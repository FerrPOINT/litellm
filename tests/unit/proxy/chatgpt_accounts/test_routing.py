from typing import Final

import httpx
import pytest
from fastapi import HTTPException
from openai import BadRequestError, RateLimitError
from test_service import account

from litellm.proxy.chatgpt_accounts.availability import classify_outage
from litellm.proxy.chatgpt_accounts.models import ACCOUNT_METADATA_KEY, AccountBindings, AccountsState
from litellm.proxy.chatgpt_accounts.routing import AccountsRoutingHook
from litellm.proxy.chatgpt_accounts.service import AccountService, deployment_id


@pytest.mark.asyncio
async def test_snapshot_survives_switch_and_client_metadata_is_not_authoritative(service) -> None:
    first: Final = account(service)
    second: Final = account(service)
    assert isinstance(
        await service.bind(AccountBindings(default_account_id=first), service.store.read().revision), AccountsState
    )
    hook: Final = AccountsRoutingHook(service)
    request: Final = await hook.async_pre_call_hook(
        None, None, {"model": "main-dev", "metadata": {ACCOUNT_METADATA_KEY: str(second)}}, "aresponses"
    )
    assert request is not None
    assert isinstance(
        await service.bind(AccountBindings(default_account_id=second), service.store.read().revision), AccountsState
    )
    selected: Final = await hook.async_filter_deployments("main-dev", service.router.get_model_list(), None, request)
    assert selected
    assert all(item["model_info"]["managed_chatgpt_account_id"] == str(first) for item in selected)
    new_request: Final = await hook.async_pre_call_hook(None, None, {"model": "main-dev"}, "aresponses")
    assert new_request["metadata"][ACCOUNT_METADATA_KEY] == str(second)


@pytest.mark.asyncio
async def test_final_account_failure_reports_earliest_retry_without_hiding_invalid_requests(
    service: AccountService,
) -> None:
    fixed: Final = AccountService(service.store, service.router, service.probe, clock=lambda: 1000)
    first: Final = account(fixed)
    second: Final = account(fixed)
    assert isinstance(
        await fixed.bind(
            AccountBindings(default_account_id=first, automatic_failover=True), fixed.store.read().revision
        ),
        AccountsState,
    )
    hook: Final = AccountsRoutingHook(fixed)
    request: Final = await hook.async_pre_call_hook(None, None, {"model": "main-dev"}, "aresponses")
    assert request is not None
    error: Final = RateLimitError(
        "fixture quota",
        response=httpx.Response(
            429, headers={"Retry-After": "60"}, request=httpx.Request("POST", "https://fixture.test")
        ),
        body={},
    )
    outage: Final = classify_outage(error, fixed.clock())
    assert outage is not None
    fixed.record_outage(first, outage)
    assert await hook.async_post_call_failure_hook(request, error, None) is None
    fixed.record_outage(second, outage.model_copy(update={"retry_at": 1120, "until": 1120}))
    exhausted: Final = await hook.async_post_call_failure_hook(request, error, None)
    assert isinstance(exhausted, HTTPException)
    assert exhausted.status_code == 503 and exhausted.headers == {"Retry-After": "60"}
    invalid: Final = BadRequestError(
        "fixture malformed history",
        response=httpx.Response(400, request=httpx.Request("POST", "https://fixture.test")),
        body={},
    )
    assert await hook.async_post_call_failure_hook(request, invalid, None) is None
    assert await hook.async_post_call_failure_hook({"model": "unmanaged"}, error, None) is None
    assert await hook.async_post_call_failure_hook({**request, "model": "unmanaged"}, error, None) is None
    manual: Final = {
        "model": "main-dev",
        "metadata": {"_managed_chatgpt_route": "main-dev", "_managed_chatgpt_automatic": False},
    }
    assert await hook.async_post_call_failure_hook(manual, error, None) is None


@pytest.mark.asyncio
async def test_retry_chooses_an_untried_account_even_if_the_first_is_ready(service) -> None:
    first: Final = account(service)
    second: Final = account(service)
    assert isinstance(
        await service.bind(
            AccountBindings(default_account_id=first, automatic_failover=True), service.store.read().revision
        ),
        AccountsState,
    )
    hook: Final = AccountsRoutingHook(service)
    request: Final = await hook.async_pre_call_hook(None, None, {"model": "main-dev"}, "aresponses")
    assert request is not None
    deployments: Final = service.router.get_model_list(model_name="main-dev")
    retried: Final = await hook.async_filter_deployments(
        "main-dev",
        deployments,
        None,
        {**request, "_retry_skipped_deployment_ids": (deployment_id(first, service.templates[0]),)},
    )
    assert retried
    assert all(item["model_info"]["managed_chatgpt_account_id"] == str(second) for item in retried)
    independent: Final = await hook.async_filter_deployments("main-dev", deployments, None, request)
    assert independent
    assert all(item["model_info"]["managed_chatgpt_account_id"] == str(first) for item in independent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [{"api_key": "other"}, {"chatgpt_auth_file": "/tmp/stolen"}, {"extra_headers": {"ChatGPT-Account-Id": "other"}}],
)
async def test_caller_cannot_override_credentials(service, payload) -> None:
    hook: Final = AccountsRoutingHook(service)
    with pytest.raises(HTTPException) as caught:
        await hook.async_pre_call_hook(None, None, {"model": "main-dev", **payload}, "aresponses")
    assert caught.value.status_code == 400
    assert await hook.async_pre_call_hook(None, None, {"model": "ollama-fallback", **payload}, "aresponses") is None


@pytest.mark.asyncio
async def test_failure_status_uses_provider_identity_and_real_retry_header(service) -> None:
    first: Final = account(service)
    second: Final = account(service)
    assert isinstance(
        await service.bind(AccountBindings(default_account_id=first), service.store.read().revision), AccountsState
    )
    revision: Final = service.store.read().revision
    hook: Final = AccountsRoutingHook(service)
    request: Final = await hook.async_pre_call_hook(None, None, {"model": "main-dev"}, "aresponses")
    response: Final = httpx.Response(
        429,
        headers={"retry-after": "Fri, 01 Jan 2100 00:00:00 GMT"},
        request=httpx.Request("POST", "https://fixture.test"),
    )
    error: Final = RateLimitError("fixture quota", response=response, body={})
    await hook.async_post_call_failure_deployment_hook(
        {"metadata": request["metadata"], "custom_llm_provider": "openai"}, error, None
    )
    assert service.record(first).status == "ready"
    await hook.async_post_call_failure_deployment_hook(
        {"metadata": {**request["metadata"], "model_info": {"id": deployment_id(first, service.templates[0])}}},
        error,
        None,
    )
    assert service.record(first).status == "rate_limited"
    assert service.record(first).retry_at == 4102444800
    assert service.selection("main-dev", None) == first
    assert service.store.read().revision == revision
    switched: Final = await service.bind(AccountBindings(default_account_id=second), revision)
    assert isinstance(switched, AccountsState)
    assert service.selection("main-dev", None) == second
