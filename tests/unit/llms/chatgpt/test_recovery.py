import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final
from unittest.mock import patch

import httpx
import jwt
import pytest
import respx

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import CHATGPT_OAUTH_TOKEN_URL, GetAccessTokenError
from litellm.llms.chatgpt.recovery import apost_with_refresh, post_with_refresh
from litellm.proxy.chatgpt_accounts.store import write_private_file
from litellm.router import Router


def token(value: str) -> str:
    return jwt.encode({"exp": 4102444800, "sub": value}, "fixture-secret-aaaaaaaaaaaaaaaaaaaaaaaa", algorithm="HS256")


def authenticator(tmp_path: Path) -> Authenticator:
    path: Final = tmp_path / "auth.json"
    write_private_file(
        path,
        json.dumps({"access_token": token("old"), "refresh_token": "test-refresh", "expires_at": 4102444800}),
    )
    return Authenticator(auth_file=str(path))


@respx.mock
def test_parallel_rejected_tokens_refresh_once_and_failure_keeps_file(tmp_path: Path) -> None:
    auth: Final = authenticator(tmp_path)
    old: Final = auth.get_access_token()
    refreshed: Final = token("new")
    endpoint: Final = respx.post(CHATGPT_OAUTH_TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": refreshed, "id_token": refreshed, "refresh_token": "rotated"}
        )
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        results: Final = tuple(pool.map(lambda _: auth.refresh_access_token(rejected_access_token=old), range(4)))
    assert results == (refreshed,) * 4
    assert endpoint.call_count == 1
    before: Final = Path(auth.auth_file).read_bytes()
    endpoint.mock(return_value=httpx.Response(400, json={"error": "invalid_grant"}))
    with pytest.raises(GetAccessTokenError):
        auth.refresh_access_token(rejected_access_token=refreshed)
    assert Path(auth.auth_file).read_bytes() == before
    assert Path(auth.auth_file).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
@pytest.mark.asyncio
async def test_invalidated_request_retries_once_with_new_header(tmp_path: Path, asynchronous: bool) -> None:
    auth: Final = authenticator(tmp_path)
    old: Final = auth.get_access_token()
    refreshed: Final = token("new")
    refresh: Final = respx.post(CHATGPT_OAUTH_TOKEN_URL).mock(
        return_value=httpx.Response(200, json={"access_token": refreshed, "id_token": refreshed})
    )
    upstream: Final = respx.post("https://fixture.test/responses").mock(
        side_effect=[
            httpx.Response(401, json={"error": {"code": "token_invalidated"}}),
            httpx.Response(200, json={"id": "resp-ok"}),
        ]
    )
    headers: Final = {"Authorization": "Bearer " + old, "ChatGPT-Account-Id": "fixture-account"}

    def send(updated):
        response: Final = httpx.post("https://fixture.test/responses", headers=updated, json={"input": "fixture"})
        response.raise_for_status()
        return response

    async def asend(updated):
        async with httpx.AsyncClient() as client:
            response: Final = await client.post(
                "https://fixture.test/responses", headers=updated, json={"input": "fixture"}
            )
            response.raise_for_status()
            return response

    response: Final = (
        await apost_with_refresh(asend, headers, auth) if asynchronous else post_with_refresh(send, headers, auth)
    )
    assert response.json()["id"] == "resp-ok"
    assert upstream.call_count == 2 and refresh.call_count == 1
    assert upstream.calls[1].request.headers["authorization"] == "Bearer " + refreshed
    assert upstream.calls[1].request.content == upstream.calls[0].request.content


@pytest.mark.parametrize("status,code", [(429, "token_invalidated"), (403, "token_invalidated"), (401, "other")])
@respx.mock
def test_other_errors_do_not_refresh(tmp_path: Path, status: int, code: str) -> None:
    auth: Final = authenticator(tmp_path)
    refresh: Final = respx.post(CHATGPT_OAUTH_TOKEN_URL)
    upstream: Final = respx.post("https://fixture.test/responses").mock(
        return_value=httpx.Response(status, json={"error": {"code": code}})
    )

    def send(headers):
        response: Final = httpx.post("https://fixture.test/responses", headers=headers)
        response.raise_for_status()
        return response

    with pytest.raises(httpx.HTTPStatusError):
        post_with_refresh(send, {"Authorization": "Bearer " + auth.get_access_token()}, auth)
    assert upstream.call_count == 1 and refresh.call_count == 0


@respx.mock
def test_failed_credential_replace_keeps_previous_file_and_redacts_tokens(tmp_path: Path, caplog) -> None:
    auth: Final = authenticator(tmp_path)
    old: Final = auth.get_access_token()
    before: Final = Path(auth.auth_file).read_bytes()
    refreshed: Final = token("replacement-secret")
    respx.post(CHATGPT_OAUTH_TOKEN_URL).mock(
        return_value=httpx.Response(200, json={"access_token": refreshed, "id_token": refreshed})
    )
    with patch("os.replace", side_effect=OSError("fixture storage unavailable")):
        with pytest.raises(GetAccessTokenError) as caught:
            auth.refresh_access_token(rejected_access_token=old)
    assert caught.value.status_code == 503
    assert Path(auth.auth_file).read_bytes() == before
    assert refreshed not in caplog.text and "test-refresh" not in caplog.text


@pytest.mark.parametrize("chat,stream", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.asyncio
@respx.mock
async def test_public_provider_recovers_invalidated_token_for_streaming_tools(
    tmp_path: Path, monkeypatch, chat: bool, stream: bool
) -> None:
    monkeypatch.setenv("CHATGPT_API_BASE", "https://fixture.test")
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    auth: Final = authenticator(tmp_path)
    refreshed: Final = token("replacement")
    refresh: Final = respx.post(CHATGPT_OAUTH_TOKEN_URL).mock(
        return_value=httpx.Response(200, json={"access_token": refreshed, "id_token": refreshed})
    )
    item: Final = {
        "id": "fc_fixture",
        "type": "function_call",
        "call_id": "call_fixture",
        "name": "echo",
        "arguments": '{"value":"ok"}',
        "status": "completed",
    }
    response: Final = {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1800000000,
        "model": "gpt-6-luna",
        "status": "completed",
        "output": [item],
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    events: Final = (
        {"type": "response.created", "response": {**response, "output": [], "status": "in_progress"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {**item, "arguments": ""}},
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_fixture",
            "output_index": 0,
            "delta": item["arguments"],
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_fixture",
            "output_index": 0,
            "arguments": item["arguments"],
        },
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": response},
    )
    sse: Final = "".join(
        "data: " + json.dumps({**event, "sequence_number": index}) + "\n\n" for index, event in enumerate(events)
    )
    upstream: Final = respx.post("https://fixture.test/responses").mock(
        side_effect=[
            httpx.Response(401, json={"error": {"code": "token_invalidated"}}),
            httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}),
        ]
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "fixture-tools",
                "litellm_params": {"model": "chatgpt/gpt-6-luna", "chatgpt_auth_file": auth.auth_file},
                "model_info": {"mode": "responses"},
            }
        ],
        num_retries=0,
    )
    result: Final = (
        await router.acompletion(
            model="fixture-tools", messages=[{"role": "user", "content": "Call echo"}], stream=stream
        )
        if chat
        else await router.aresponses(model="fixture-tools", input="Call echo", stream=stream)
    )
    if stream:
        chunks: Final = tuple([chunk.model_dump_json() async for chunk in result])
        assert any("echo" in chunk for chunk in chunks)
        assert any("value" in chunk for chunk in chunks)
    elif chat:
        assert result.choices[0].message.tool_calls[0].function.arguments == '{"value":"ok"}'
    else:
        assert result.output[0].arguments == '{"value":"ok"}'
    assert upstream.call_count == 2 and refresh.call_count == 1
    assert upstream.calls[0].request.content == upstream.calls[1].request.content
    assert upstream.calls[1].request.headers["authorization"] == "Bearer " + refreshed


@respx.mock
def test_refresh_is_independent_for_two_profiles(tmp_path: Path) -> None:
    first: Final = authenticator(tmp_path / "A")
    second: Final = authenticator(tmp_path / "B")
    for auth, refresh in ((first, "refresh-A"), (second, "refresh-B")):
        write_private_file(
            Path(auth.auth_file),
            json.dumps({"access_token": token("old"), "refresh_token": refresh, "expires_at": 4102444800}),
        )

    def refresh_response(request: httpx.Request) -> httpx.Response:
        value: Final = json.loads(request.content)["refresh_token"]
        return httpx.Response(
            200, json={"access_token": token(value), "id_token": token(value), "refresh_token": value}
        )

    endpoint: Final = respx.post(CHATGPT_OAUTH_TOKEN_URL).mock(side_effect=refresh_response)
    rejected: Final = first.get_access_token()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results: Final = tuple(
            pool.map(
                lambda auth: auth.refresh_access_token(rejected_access_token=rejected), (first, first, second, second)
            )
        )
    assert results == (token("refresh-A"), token("refresh-A"), token("refresh-B"), token("refresh-B"))
    assert endpoint.call_count == 2
    assert first.get_access_token() == token("refresh-A")
    assert second.get_access_token() == token("refresh-B")


@respx.mock
def test_other_provider_keeps_opaque_header_bytes_without_oauth_refresh(tmp_path: Path) -> None:
    auth: Final = authenticator(tmp_path)
    refresh: Final = respx.post(CHATGPT_OAUTH_TOKEN_URL)
    upstream: Final = respx.post("https://fixture.test/responses").mock(
        return_value=httpx.Response(200, json={"id": "fixture"})
    )
    headers: Final = {"X-Provider-Opaque": b"\xff"}

    def send(values):
        response: Final = httpx.post("https://fixture.test/responses", headers=values)
        response.raise_for_status()
        return response

    from litellm.llms.chatgpt.recovery import request_authenticator

    response: Final = post_with_refresh(
        send, headers, request_authenticator("openai", {"chatgpt_auth_file": auth.auth_file}, headers)
    )
    assert response.json() == {"id": "fixture"}
    assert dict(upstream.calls[0].request.headers.raw)[b"X-Provider-Opaque"] == b"\xff"
    assert refresh.call_count == 0
