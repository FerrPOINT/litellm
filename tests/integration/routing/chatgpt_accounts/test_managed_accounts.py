import json
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx
import jwt
import pytest
import yaml

from litellm.proxy.chatgpt_accounts.models import AccountBindings, AccountRecord, AccountsState
from litellm.proxy.chatgpt_accounts.store import AccountStore, write_private_file
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import stop_child


class ScriptedUpstream(BaseHTTPRequestHandler):
    owners: tuple[str, ...] = ()
    bodies: tuple[dict, ...] = ()
    rejected_owners: frozenset[str] = frozenset()
    failure_status: int = 429
    retry_after: str | None = None
    structured_quota: bool = False
    partial_failure: bool = False
    stream_started = threading.Event()
    release_stream = threading.Event()

    def log_message(self, format, *args) -> None:
        pass

    def do_GET(self) -> None:
        payload: Final = json.dumps({"models": [{"slug": "gpt-6.1-terra"}, {"slug": "gpt-6-luna"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        body: Final = json.loads(self.rfile.read(int(self.headers["content-length"])))
        owner: Final = self.headers.get("ChatGPT-Account-Id", "missing-owner")
        type(self).owners = (*type(self).owners, owner)
        type(self).bodies = (*type(self).bodies, body)
        if owner in type(self).rejected_owners:
            quota: Final = type(self).failure_status == 429 or type(self).structured_quota
            failure: Final = json.dumps(
                {
                    "error": {
                        "message": "fixture rejection",
                        "type": "rate_limit_error" if quota else "invalid_request_error",
                        "code": "rate_limit_exceeded" if quota else "invalid_input",
                    }
                }
            ).encode()
            self.send_response(type(self).failure_status)
            if type(self).retry_after is not None:
                self.send_header("Retry-After", type(self).retry_after)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(failure)))
            self.end_headers()
            self.wfile.write(failure)
            return
        text: Final = "OK " + owner
        item: Final = {
            "type": "message",
            "id": "msg_fixture",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        tool: Final = {
            "id": "fc_fixture",
            "type": "function_call",
            "call_id": "call_fixture",
            "name": "echo",
            "arguments": '{"value":"ok"}',
            "status": "completed",
        }
        output: Final = tool if body.get("tools") else item
        response: Final = {
            "id": "resp_" + uuid4().hex,
            "object": "response",
            "created_at": 1800000000,
            "status": "completed",
            "model": body["model"],
            "output": [output],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "temperature": 1,
            "top_p": 1,
            "error": None,
            "incomplete_details": None,
        }
        events: Final = (
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**item, "status": "in_progress", "content": []},
            },
            {
                "type": "response.content_part.added",
                "item_id": "msg_fixture",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_fixture",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            },
            {
                "type": "response.output_text.done",
                "item_id": "msg_fixture",
                "output_index": 0,
                "content_index": 0,
                "text": text,
            },
            {"type": "response.output_item.done", "output_index": 0, "item": output},
            {"type": "response.completed", "response": response},
        )
        tool_events: Final = (
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "output_index": 0, "item": {**tool, "arguments": ""}},
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_fixture",
                "output_index": 0,
                "delta": tool["arguments"],
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_fixture",
                "output_index": 0,
                "arguments": tool["arguments"],
            },
            {"type": "response.output_item.done", "output_index": 0, "item": tool},
            {"type": "response.completed", "response": response},
        )
        selected_events: Final = (
            (
                *events[:4],
                {
                    "type": "response.failed",
                    "response": {
                        **response,
                        "status": "failed",
                        "error": {"code": "rate_limit_exceeded", "message": "fixture quota"},
                    },
                },
            )
            if type(self).partial_failure
            else tool_events
            if body.get("tools")
            else events
        )
        payload: Final = "".join(
            "event: " + event["type"] + "\ndata: " + json.dumps({**event, "sequence_number": index}) + "\n\n"
            for index, event in enumerate(selected_events)
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if "hold-stream" in json.dumps(body):
            type(self).stream_started.set()
            type(self).release_stream.wait(timeout=30)
        self.wfile.write(payload)


@contextmanager
def proxy(
    tmp_path: Path, automatic: bool = False, disable_cooldowns: bool = False, deadline: float = 600
) -> Iterator[tuple[httpx.Client, tuple[str, str], subprocess.Popen]]:
    ScriptedUpstream.owners = ()
    ScriptedUpstream.bodies = ()
    ScriptedUpstream.rejected_owners = frozenset()
    ScriptedUpstream.failure_status = 429
    ScriptedUpstream.retry_after = None
    ScriptedUpstream.structured_quota = False
    ScriptedUpstream.partial_failure = False
    ScriptedUpstream.stream_started.clear()
    ScriptedUpstream.release_stream.clear()
    store: Final = AccountStore(tmp_path / "accounts")
    ids: Final = (uuid4(), uuid4())
    records: Final = tuple(
        AccountRecord(
            id=account_id,
            label="Same label",
            email="same@example.test",
            workspace=owner,
            client_id="fixture",
            status="ready",
        )
        for account_id, owner in zip(ids, ("account-A", "account-B"), strict=True)
    )
    for record in records:
        access: Final = jwt.encode(
            {"exp": 4102444800, "https://api.openai.com/auth": {"chatgpt_account_id": record.workspace}},
            "fixture-secret-aaaaaaaaaaaaaaaaaaaaaaaa",
            algorithm="HS256",
        )
        write_private_file(
            store.auth_file(record.id),
            json.dumps(
                {
                    "access_token": access,
                    "refresh_token": "fixture-unused",
                    "account_id": record.workspace,
                    "expires_at": 4102444800,
                }
            ),
        )
    assert isinstance(
        store.update(
            lambda state: AccountsState(
                accounts=records, bindings=AccountBindings(default_account_id=ids[0], automatic_failover=automatic)
            )
        ),
        AccountsState,
    )
    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedUpstream)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port: Final = allocate_port()
    config: Final = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": route,
                        "litellm_params": {"model": "chatgpt/" + model},
                        "model_info": {"mode": "responses"},
                    }
                    for route, model in (("main-dev", "gpt-6.1-terra"), ("app-test", "gpt-6-luna"))
                ],
                "litellm_settings": {"turn_off_message_logging": True},
                "router_settings": {"num_retries": 0, "disable_cooldowns": disable_cooldowns},
                "general_settings": {"master_key": "sk-fixture-inference", "disable_spend_logs": True},
            }
        )
    )
    environment: Final = {
        **os.environ,
        "LITELLM_CHATGPT_ACCOUNTS_DIR": str(store.root),
        "LITELLM_CHATGPT_ACCOUNTS_ADMIN_KEY": "admin-" + "a" * 40,
        "CHATGPT_API_BASE": f"http://127.0.0.1:{server.server_port}",
        "CHATGPT_TOKEN_DIR": str(tmp_path / "unassigned"),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_CHATGPT_ACCOUNTS_REQUEST_TIMEOUT": str(deadline),
    }
    log: Final = tmp_path / "proxy.log"
    with log.open("w") as output:
        process: Final = subprocess.Popen(
            [
                os.sys.executable,
                "-m",
                "litellm.proxy.proxy_cli",
                "--config",
                str(config),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--num_workers",
                "1",
            ],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                timeout=45,
                headers={"Authorization": "Bearer sk-fixture-inference"},
            ) as client:
                deadline: Final = time.monotonic() + 45
                while time.monotonic() < deadline:
                    try:
                        if client.get("/health/liveliness").status_code == 200:
                            break
                    except httpx.ConnectError:
                        if process.poll() is not None:
                            pytest.fail("Proxy exited: " + log.read_text()[-6000:])
                    threading.Event().wait(0.1)
                else:
                    pytest.fail("Proxy startup deadline exceeded")
                yield client, tuple(str(item) for item in ids), process
        finally:
            stop_child(process, grace_seconds=15)
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_real_proxy_switches_without_restart_and_keeps_response_owner(tmp_path: Path) -> None:
    with proxy(tmp_path) as (client, ids, process):
        pid: Final = process.pid
        first: Final = client.post(
            "/v1/responses", json={"model": "main-dev", "input": "__PRIVATE_PROMPT_SENTINEL_6929__"}
        )
        assert first.status_code == 200, first.text
        assert first.json()["output"][0]["content"][0]["text"] == "OK account-A"
        denied: Final = client.get("/chatgpt/accounts")
        assert denied.status_code == 401
        signed_in: Final = client.post("/chatgpt/accounts/session", json={"key": "admin-" + "a" * 40})
        assert signed_in.status_code == 200
        current: Final = client.get("/chatgpt/accounts").json()
        switched: Final = client.put(
            "/chatgpt/accounts/bindings",
            json={
                "revision": current["revision"],
                "bindings": {"default_account_id": ids[1], "routes": {"app-test": ids[0]}},
            },
        )
        assert switched.status_code == 200, switched.text
        second: Final = client.post(
            "/v1/chat/completions", json={"model": "main-dev", "messages": [{"role": "user", "content": "fixture"}]}
        )
        assert second.status_code == 200, second.text
        assert second.json()["choices"][0]["message"]["content"] == "OK account-B"
        continued: Final = client.post(
            "/v1/responses",
            json={"model": "main-dev", "input": "fixture continuation", "previous_response_id": first.json()["id"]},
        )
        assert continued.status_code == 200, continued.text
        assert continued.json()["output"][0]["content"][0]["text"] == "OK account-A"
        streamed: Final = client.post(
            "/v1/chat/completions",
            json={"model": "app-test", "messages": [{"role": "user", "content": "fixture"}], "stream": True},
        )
        assert streamed.status_code == 200, streamed.text
        assert "OK account-A" in streamed.text and "[DONE]" in streamed.text
        tools: Final = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ]
        tool_response: Final = client.post(
            "/v1/chat/completions",
            json={"model": "main-dev", "messages": [{"role": "user", "content": "Call echo"}], "tools": tools},
        )
        assert tool_response.status_code == 200, tool_response.text
        call: Final = tool_response.json()["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "echo" and json.loads(call["function"]["arguments"]) == {"value": "ok"}
        tool_stream: Final = client.post(
            "/v1/chat/completions",
            json={
                "model": "main-dev",
                "messages": [{"role": "user", "content": "Call echo"}],
                "tools": tools,
                "stream": True,
            },
        )
        assert tool_stream.status_code == 200, tool_stream.text
        chunks: Final = [
            json.loads(line[6:])
            for line in tool_stream.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        argument_deltas: Final = tuple(
            call["function"].get("arguments", "")
            for chunk in chunks
            for call in chunk.get("choices", [{}])[0].get("delta", {}).get("tool_calls", [])
        )
        assert json.loads("".join(argument_deltas)) == {"value": "ok"}
        back: Final = client.put(
            "/chatgpt/accounts/bindings",
            json={"revision": switched.json()["revision"], "bindings": {"default_account_id": ids[0]}},
        )
        assert back.status_code == 200, back.text
        again: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture"})
        assert again.status_code == 200 and again.json()["output"][0]["content"][0]["text"] == "OK account-A"
        assert process.pid == pid and process.poll() is None
        assert all("chatgpt_auth_file" not in body and "api_key" not in body for body in ScriptedUpstream.bodies)
        log: Final = (tmp_path / "proxy.log").read_text()
        assert "ChatGPT accounts request_finished" in log and "reason=manual" in log
        assert "__PRIVATE_PROMPT_SENTINEL_6929__" not in log and "OK account-A" not in log
        for auth_file in (tmp_path / "accounts" / "credentials").rglob("auth.json"):
            credentials: Final = json.loads(auth_file.read_text())
            assert credentials["access_token"] not in log and credentials["refresh_token"] not in log


def test_native_responses_stream_records_selected_profile(tmp_path: Path) -> None:
    with proxy(tmp_path) as (client, ids, process):
        result: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture", "stream": True})
        assert result.status_code == 200, result.text
        assert "response.completed" in result.text and process.poll() is None
        deadline: Final = time.monotonic() + 3
        expected: Final = f"profile={ids[0]} result=success"
        while time.monotonic() < deadline:
            if expected in (tmp_path / "proxy.log").read_text():
                break
            time.sleep(0.05)
        assert expected in (tmp_path / "proxy.log").read_text()


def test_rate_limited_profile_never_rotates_to_another_chatgpt_profile(tmp_path: Path) -> None:
    with proxy(tmp_path) as (client, ids, process):
        ScriptedUpstream.rejected_owners = frozenset({"account-A"})
        for _ in range(4):
            limited: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture quota"})
            assert limited.status_code == 429, limited.text
        assert ScriptedUpstream.owners and set(ScriptedUpstream.owners) == {"account-A"}
        assert client.post("/chatgpt/accounts/session", json={"key": "admin-" + "a" * 40}).status_code == 200
        current: Final = client.get("/chatgpt/accounts").json()
        assert current["accounts"][0]["status"] == "rate_limited", current
        switched: Final = client.put(
            "/chatgpt/accounts/bindings",
            json={"revision": current["revision"], "bindings": {"default_account_id": ids[1]}},
        )
        assert switched.status_code == 200, switched.text
        working: Final = client.post(
            "/v1/responses", json={"model": "main-dev", "input": "fixture after manual switch"}
        )
        assert working.status_code == 200, working.text
        assert working.json()["output"][0]["content"][0]["text"] == "OK account-B"
        assert process.poll() is None


def test_switch_during_stream_keeps_original_profile_and_accepts_parallel_request(tmp_path: Path) -> None:
    with proxy(tmp_path) as (client, ids, process):
        assert client.post("/chatgpt/accounts/session", json={"key": "admin-" + "a" * 40}).status_code == 200
        with ThreadPoolExecutor(max_workers=1) as pool:
            streamed: Final = pool.submit(
                client.post,
                "/v1/chat/completions",
                json={"model": "main-dev", "messages": [{"role": "user", "content": "hold-stream"}], "stream": True},
            )
            try:
                assert ScriptedUpstream.stream_started.wait(timeout=15), "Upstream did not receive the stream"
                current: Final = client.get("/chatgpt/accounts").json()
                switched: Final = client.put(
                    "/chatgpt/accounts/bindings",
                    json={"revision": current["revision"], "bindings": {"default_account_id": ids[1]}},
                )
                assert switched.status_code == 200, switched.text
                parallel: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "parallel request"})
                assert parallel.status_code == 200, parallel.text
                assert parallel.json()["output"][0]["content"][0]["text"] == "OK account-B"
            finally:
                ScriptedUpstream.release_stream.set()
            completed: Final = streamed.result(timeout=30)
            assert completed.status_code == 200, completed.text
            assert "OK account-A" in completed.text and "OK account-B" not in completed.text
        assert process.poll() is None


@pytest.mark.parametrize("status", [401, 402, 403, 429, 503])
@pytest.mark.parametrize("stream", [False, True])
def test_automatic_failover_repeats_request_and_stays_on_working_profile(
    tmp_path: Path, status: int, stream: bool
) -> None:
    with proxy(tmp_path, automatic=True) as (client, ids, process):
        ScriptedUpstream.failure_status = status
        ScriptedUpstream.rejected_owners = frozenset({"account-A"})
        first: Final = client.post(
            "/v1/chat/completions",
            json={"model": "main-dev", "messages": [{"role": "user", "content": "fixture"}], "stream": stream},
        )
        assert first.status_code == 200, first.text
        assert "OK account-B" in first.text
        assert ScriptedUpstream.owners == ("account-A", "account-B")
        again: Final = client.post("/v1/responses", json={"model": "app-test", "input": "fixture"})
        assert again.status_code == 200, again.text
        assert ScriptedUpstream.owners[-1] == "account-B"
        assert client.post("/chatgpt/accounts/session", json={"key": "admin-" + "a" * 40}).status_code == 200
        state: Final = client.get("/chatgpt/accounts").json()
        assert state["active_accounts"]["default"] == ids[1]
        ScriptedUpstream.rejected_owners = frozenset({"account-B"})
        checked: Final = client.post(f"/chatgpt/accounts/{ids[0]}/check")
        assert checked.status_code == 200, checked.text
        reversed_request: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture"})
        assert reversed_request.status_code == 200, reversed_request.text
        assert ScriptedUpstream.owners[-2:] == ("account-B", "account-A")
        ScriptedUpstream.rejected_owners = frozenset({"account-A", "account-B"})
        exhausted: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture"})
        assert exhausted.status_code == 503, exhausted.text
        count: Final = len(ScriptedUpstream.owners)
        waiting: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture"})
        assert waiting.status_code == 503
        assert len(ScriptedUpstream.owners) == count
        assert process.poll() is None


def test_automatic_failover_does_not_repeat_invalid_requests_on_other_accounts(tmp_path: Path) -> None:
    with proxy(tmp_path, automatic=True) as (client, ids, process):
        ScriptedUpstream.failure_status = 400
        ScriptedUpstream.rejected_owners = frozenset({"account-A"})
        response: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture"})
        assert response.status_code == 400, response.text
        assert ScriptedUpstream.owners == ("account-A",)
        assert client.post("/chatgpt/accounts/session", json={"key": "admin-" + "a" * 40}).status_code == 200
        state: Final = client.get("/chatgpt/accounts").json()
        assert state["active_accounts"]["default"] == ids[0]
        assert all(account["unavailable_reason"] is None for account in state["accounts"])
        ScriptedUpstream.rejected_owners = frozenset()
        valid: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture"})
        assert valid.status_code == 200, valid.text
        assert ScriptedUpstream.owners == ("account-A", "account-A")
        assert process.poll() is None


def test_one_request_never_retries_a_failed_account_even_if_its_reset_is_immediate(tmp_path: Path) -> None:
    with proxy(tmp_path, automatic=True, disable_cooldowns=True) as (client, ids, process):
        ScriptedUpstream.rejected_owners = frozenset({"account-A"})
        ScriptedUpstream.retry_after = "0"
        response: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture"})
        assert response.status_code == 200, response.text
        assert ScriptedUpstream.owners == ("account-A", "account-B")
        assert process.poll() is None and len(ids) == 2


def test_provider_quota_code_can_fail_over_even_when_http_status_is_400(tmp_path: Path) -> None:
    with proxy(tmp_path, automatic=True) as (client, ids, process):
        ScriptedUpstream.failure_status = 400
        ScriptedUpstream.structured_quota = True
        ScriptedUpstream.rejected_owners = frozenset({"account-A"})
        response: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture"})
        assert response.status_code == 200, response.text
        assert ScriptedUpstream.owners == ("account-A", "account-B")
        assert process.poll() is None and len(ids) == 2


def test_account_retry_after_does_not_delay_the_next_account(tmp_path: Path) -> None:
    with proxy(tmp_path, automatic=True) as (client, ids, process):
        ScriptedUpstream.retry_after = "600"
        ScriptedUpstream.rejected_owners = frozenset({"account-A"})
        response: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture"})
        assert response.status_code == 200, response.text
        assert ScriptedUpstream.owners == ("account-A", "account-B")
        assert process.poll() is None and len(ids) == 2


def test_recovered_account_is_used_after_the_working_account_fails_without_manual_check(tmp_path: Path) -> None:
    with proxy(tmp_path, automatic=True) as (client, ids, process):
        ScriptedUpstream.retry_after = "0"
        ScriptedUpstream.rejected_owners = frozenset({"account-A"})
        first: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture"})
        assert first.status_code == 200, first.text
        assert ScriptedUpstream.owners == ("account-A", "account-B")
        sticky: Final = client.post("/v1/responses", json={"model": "app-test", "input": "fixture"})
        assert sticky.status_code == 200 and ScriptedUpstream.owners[-1] == "account-B"
        ScriptedUpstream.rejected_owners = frozenset({"account-B"})
        recovered: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture"})
        assert recovered.status_code == 200, recovered.text
        assert ScriptedUpstream.owners[-2:] == ("account-B", "account-A")
        assert process.poll() is None and len(ids) == 2


def test_request_deadline_cancels_the_whole_chain(tmp_path: Path) -> None:
    with proxy(tmp_path, automatic=True, deadline=1) as (client, ids, process):
        response: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "hold-stream"})
        ScriptedUpstream.release_stream.set()
        assert response.status_code == 504, response.text
        assert response.json()["error"]["code"] == "chatgpt_request_deadline"
        assert ScriptedUpstream.owners == ("account-A",)
        assert process.poll() is None and len(ids) == 2


def test_a_partial_stream_is_not_replayed_on_another_account(tmp_path: Path) -> None:
    with proxy(tmp_path, automatic=True) as (client, ids, process):
        ScriptedUpstream.partial_failure = True
        response: Final = client.post("/v1/responses", json={"model": "main-dev", "input": "fixture", "stream": True})
        assert response.status_code == 200, response.text
        assert "OK account-A" in response.text
        assert "response.failed" in response.text or '"error"' in response.text
        assert ScriptedUpstream.owners == ("account-A",)
        assert process.poll() is None and len(ids) == 2
