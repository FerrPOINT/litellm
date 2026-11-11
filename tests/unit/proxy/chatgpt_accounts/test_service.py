import json
from pathlib import Path
from typing import Final
from uuid import UUID, uuid4

import pytest

from litellm.proxy.chatgpt_accounts.models import (
    AccountBindings,
    AccountFailure,
    AccountRecord,
    AccountsState,
    ProbeResult,
)
from litellm.proxy.chatgpt_accounts.service import AccountService, deployment_id
from litellm.proxy.chatgpt_accounts.store import AccountStore, write_private_file
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router import Router


class ReadyProbe:
    async def check(self, auth_file: Path, models: tuple[str, ...]) -> ProbeResult:
        return ProbeResult(models=models)


class DeniedProbe:
    async def check(self, auth_file: Path, models: tuple[str, ...]) -> ProbeResult:
        return ProbeResult(error=AccountFailure(code="model_denied", message="No model access", status=403))


class PartialModelProbe:
    def __init__(self, unavailable: str) -> None:
        self.unavailable = unavailable

    async def check(self, auth_file: Path, models: tuple[str, ...]) -> ProbeResult:
        return ProbeResult(
            models=tuple(model for model in models if model != self.unavailable), unavailable_models=(self.unavailable,)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", ("optional-model", "gpt-6.1-terra"))
async def test_unavailable_direct_model_does_not_block_healthy_routes(
    service: AccountService, unavailable: str
) -> None:
    router: Final = Router(
        model_list=[
            {"model_name": "main-dev", "litellm_params": {"model": "chatgpt/gpt-6.1-terra"}},
            {"model_name": "app-test", "litellm_params": {"model": "chatgpt/gpt-6-luna"}},
            {"model_name": "optional", "litellm_params": {"model": "chatgpt/optional-model"}},
        ]
    )
    partial: Final = AccountService(service.store, router, PartialModelProbe(unavailable))
    target: Final = account(partial)
    result: Final = await partial.bind(
        AccountBindings(default_account_id=target, automatic_failover=True), partial.store.read().revision
    )
    if unavailable == "optional-model":
        assert isinstance(result, AccountsState)
        assert partial.selection("main-dev", None) == target
        assert partial.selection("app-test", None) == target
        assert isinstance(partial.selection("optional", None), AccountFailure)
        assert partial.store.read().accounts[0].deployment_outages["chatgpt/optional-model"].code == "model_unavailable"
    else:
        assert isinstance(result, AccountFailure)
        assert result.code == "model_unavailable"
        assert partial.store.read().bindings.default_account_id is None


def account(service: AccountService, label: str = "Personal") -> UUID:
    account_id: Final = uuid4()
    result: Final = service.save_account(
        AccountRecord(
            id=account_id,
            label=label,
            email="same@example.test",
            client_id="test-client",
            status="ready",
        )
    )
    assert isinstance(result, AccountsState)
    write_private_file(service.store.auth_file(account_id), json.dumps({"access_token": str(account_id)}))
    return account_id


@pytest.mark.asyncio
async def test_switching_keeps_credentials_and_response_owner(service: AccountService) -> None:
    first: Final = account(service)
    second: Final = account(service)
    before_first: Final = service.store.auth_file(first).read_bytes()
    before_second: Final = service.store.auth_file(second).read_bytes()
    first_binding: Final = await service.bind(AccountBindings(default_account_id=first), service.store.read().revision)
    assert isinstance(first_binding, AccountsState)
    assert service.selection("main-dev", None) == first
    old_model_id: Final = deployment_id(first, service.templates[0])
    response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        model_id=old_model_id,
        custom_llm_provider="chatgpt",
        response_id="resp_fixture",
    )
    second_binding: Final = await service.bind(
        AccountBindings(default_account_id=second), service.store.read().revision
    )
    assert isinstance(second_binding, AccountsState)
    assert service.selection("main-dev", None) == second
    assert service.selection("main-dev", response_id) == first
    assert service.store.auth_file(first).read_bytes() == before_first
    assert service.store.auth_file(second).read_bytes() == before_second
    restarted: Final = AccountService(AccountStore(service.store.root), service.router, ReadyProbe())
    assert restarted.selection("main-dev", None) == second
    assert restarted.selection("main-dev", response_id) == first


@pytest.mark.asyncio
async def test_route_override_is_independent_of_global_selection(service: AccountService) -> None:
    first: Final = account(service)
    second: Final = account(service)
    result: Final = await service.bind(
        AccountBindings(default_account_id=first, routes={"app-test": second}),
        service.store.read().revision,
    )
    assert isinstance(result, AccountsState)
    assert service.selection("main-dev", None) == first
    assert service.selection("app-test", None) == second


@pytest.mark.asyncio
async def test_denied_activation_does_not_change_binding(service: AccountService) -> None:
    first: Final = account(service)
    second: Final = account(service)
    initial: Final = await service.bind(AccountBindings(default_account_id=first), service.store.read().revision)
    assert isinstance(initial, AccountsState)
    denied: Final = AccountService(service.store, service.router, DeniedProbe())
    result: Final = await denied.bind(AccountBindings(default_account_id=second), service.store.read().revision)
    assert isinstance(result, AccountFailure)
    assert result.code == "model_denied"
    assert service.selection("main-dev", None) == first


@pytest.mark.asyncio
async def test_selected_account_cannot_be_disabled(service: AccountService) -> None:
    first: Final = account(service)
    bound: Final = await service.bind(AccountBindings(default_account_id=first), service.store.read().revision)
    assert isinstance(bound, AccountsState)
    rejected: Final = service.disable(first)
    assert isinstance(rejected, AccountFailure)
    assert rejected.code == "account_in_use"
    assert service.record(first).enabled


@pytest.mark.asyncio
async def test_switch_then_disable_preserves_stream_but_rejects_new_continuation(service: AccountService) -> None:
    first: Final = account(service)
    second: Final = account(service)
    original: Final = await service.bind(AccountBindings(default_account_id=first), service.store.read().revision)
    assert isinstance(original, AccountsState)
    pinned: Final = service.selection("main-dev", None)
    changed: Final = await service.bind(AccountBindings(default_account_id=second), service.store.read().revision)
    assert isinstance(changed, AccountsState)
    disabled: Final = service.disable(first)
    assert isinstance(disabled, AccountsState)
    assert pinned == first
    assert service.router.get_deployment(deployment_id(first, service.templates[0])) is not None
    owner: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        model_id=deployment_id(first, service.templates[0]),
        custom_llm_provider="chatgpt",
        response_id="resp_owner",
    )
    result: Final = service.selection("main-dev", owner)
    assert isinstance(result, AccountFailure)
    assert result.code == "account_unavailable"


def test_stale_revision_cannot_overwrite_newer_settings(service: AccountService) -> None:
    revision: Final = service.store.read().revision
    first: Final = account(service)
    result: Final = service.store.update(lambda state: state, expected_revision=revision)
    assert isinstance(result, AccountFailure)
    assert result.code == "state_changed"
    assert service.record(first) is not None


def test_private_files_reject_symlink_and_preserve_target(tmp_path: Path) -> None:
    target: Final = tmp_path / "outside.json"
    target.write_text("protected")
    link: Final = tmp_path / "auth.json"
    link.symlink_to(target)
    with pytest.raises(OSError, match="Refusing symlinked"):
        write_private_file(link, "replacement")
    assert target.read_text() == "protected"


def test_registry_roundtrip_does_not_store_credentials(service: AccountService) -> None:
    first: Final = account(service)
    state: Final = service.store.read()
    assert state.accounts[0].id == first
    assert "access_token" not in service.store.state_file.read_text()
    assert service.store.state_file.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_overrides_do_not_allow_disabling_hub_default(service: AccountService) -> None:
    first: Final = account(service)
    second: Final = account(service)
    assert isinstance(
        await service.bind(
            AccountBindings(default_account_id=first, routes={"main-dev": second, "app-test": second}),
            service.store.read().revision,
        ),
        AccountsState,
    )
    rejected: Final = service.disable(first)
    assert isinstance(rejected, AccountFailure) and rejected.code == "account_in_use"
    absent_default: Final = await service.bind(
        AccountBindings(routes={"main-dev": second, "app-test": second}), service.store.read().revision
    )
    assert isinstance(absent_default, AccountFailure) and absent_default.code == "account_required"


@pytest.mark.asyncio
async def test_automatic_failover_stays_on_working_account_and_returns_when_it_fails(service) -> None:
    from dataclasses import dataclass

    import httpx
    from openai import RateLimitError

    from litellm.proxy.chatgpt_accounts.availability import classify_outage

    @dataclass
    class Clock:
        now: float = 1000

        def __call__(self) -> float:
            return self.now

    clock: Final = Clock()
    managed: Final = AccountService(service.store, service.router, ReadyProbe(), clock)
    first: Final = account(managed)
    second: Final = account(managed)
    state: Final = await managed.bind(
        AccountBindings(default_account_id=first, automatic_failover=True), managed.store.read().revision
    )
    assert isinstance(state, AccountsState)
    managed.activate(first, "default", state.revision)
    failure: Final = RateLimitError(
        "fixture",
        response=httpx.Response(
            429, headers={"retry-after": "60"}, request=httpx.Request("POST", "https://fixture.test")
        ),
        body={},
    )
    outage: Final = classify_outage(failure, clock())
    assert outage is not None
    managed.record_outage(first, outage)
    managed.record_success(first, 999)
    assert managed.record(first).unavailable_reason == "quota"
    assert managed.selection("main-dev", None) == second
    managed.activate(second, "default", state.revision)
    blocked: Final = managed.disable(second)
    assert isinstance(blocked, AccountFailure) and blocked.status == 409
    clock.now = 1061
    assert managed.selection("app-test", None) == second
    reverse: Final = classify_outage(failure, clock())
    assert reverse is not None
    managed.record_outage(second, reverse)
    assert managed.selection("main-dev", None) == first
    managed.activate(first, "default", state.revision)
    managed.record_success(first, clock())
    assert managed.record(first).status == "ready"
    assert managed.store.read().revision == state.revision
    assert managed.store.read().last_switches["default"].reason == "quota"


@pytest.mark.asyncio
async def test_automatic_failover_does_not_retarget_response_continuations(service) -> None:
    first: Final = account(service)
    second: Final = account(service)
    state: Final = await service.bind(
        AccountBindings(default_account_id=first, automatic_failover=True), service.store.read().revision
    )
    assert isinstance(state, AccountsState)
    response: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        "chatgpt", deployment_id(first, service.templates[0]), "resp_fixture"
    )
    service.activate(second, "default", state.revision)
    assert service.selection("main-dev", None) == second
    assert service.selection("main-dev", response) == first


@pytest.mark.asyncio
async def test_deployment_outage_keeps_the_same_accounts_other_model_available(service: AccountService) -> None:
    import httpx
    from openai import APIStatusError

    from litellm.proxy.chatgpt_accounts.availability import classify_outage

    first: Final = account(service)
    second: Final = account(service)
    state: Final = await service.bind(
        AccountBindings(default_account_id=first, automatic_failover=True), service.store.read().revision
    )
    assert isinstance(state, AccountsState)
    denied: Final = APIStatusError(
        "model denied", response=httpx.Response(403, request=httpx.Request("POST", "https://fixture.test")), body={}
    )
    outage: Final = classify_outage(denied, service.clock())
    assert outage is not None
    model: Final = service.templates[0].litellm_params.model
    service.record_outage(first, outage, model)
    assert service.selection("main-dev", None) == second
    assert service.selection("app-test", None) == first
    service.record_success(first, 0, model)
    assert service.selection("main-dev", None) == second
    service.record_success(first, service.clock(), model)
    assert service.selection("main-dev", None) == first


@pytest.mark.asyncio
async def test_selecting_a_working_account_does_not_require_a_broken_reserve_to_log_in(service: AccountService) -> None:
    first: Final = account(service)
    second: Final = account(service)
    service.update_account(first, {"status": "login_required"})

    class WorkingAccountProbe:
        async def check(self, auth_file: Path, models: tuple[str, ...]) -> ProbeResult:
            if auth_file == service.store.auth_file(second):
                return ProbeResult(models=models)
            return ProbeResult(error=AccountFailure(code="login_required", message="Sign in again", status=401))

    managed: Final = AccountService(service.store, service.router, WorkingAccountProbe())
    bound: Final = await managed.bind(
        AccountBindings(default_account_id=second, automatic_failover=True), managed.store.read().revision
    )
    assert isinstance(bound, AccountsState)
    assert managed.selection("main-dev", None) == second
