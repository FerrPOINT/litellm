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
