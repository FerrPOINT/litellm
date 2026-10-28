from collections.abc import Mapping
from uuid import UUID

import httpx
import pytest
from openai import APIStatusError

from litellm.exceptions import RateLimitError
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.proxy.chatgpt_accounts.availability import classify_outage, next_attempt
from litellm.proxy.chatgpt_accounts.models import AccountOutage, AccountRecord


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"retry-after": "0"}, 1000),
        ({"retry-after": "60"}, 1060),
        ({"retry-after": "Thu, 01 Jan 1970 00:20:00 GMT"}, 1200),
        ({"retry-after": "Thu, 01 Jan 1970 00:00:00 GMT"}, 0),
        ({"x-codex-primary-used-percent": "100", "x-codex-primary-reset-at": "2000"}, 2000),
    ],
)
def test_mapped_provider_error_preserves_observed_recovery(headers: Mapping[str, str], expected: int) -> None:
    with pytest.raises(RateLimitError) as caught:
        exception_type(
            model="fixture-model",
            custom_llm_provider="chatgpt",
            original_exception=BaseLLMException(429, "fixture rate limit", headers=dict(headers)),
        )
    outage = classify_outage(caught.value, 1000)
    assert outage is not None and outage.retry_at == expected and outage.until == expected


@pytest.mark.parametrize(
    "status,reason",
    [
        (401, "authentication"),
        (402, "payment"),
        (403, "access_denied"),
        (429, "quota"),
        (500, "provider_unavailable"),
    ],
)
def test_account_failures_have_a_bounded_recovery_policy(status: int, reason: str) -> None:
    error = APIStatusError(
        "fixture", response=httpx.Response(status, request=httpx.Request("POST", "https://fixture.test")), body={}
    )
    outage = classify_outage(error, 1000)
    assert outage is not None and outage.reason == reason
    assert outage.retry_at is None
    assert outage.until is None if status == 401 else outage.until is not None and outage.until > 1000


@pytest.mark.parametrize("status", [400, 404, 413, 422])
def test_request_errors_do_not_mark_an_account_unavailable(status: int) -> None:
    error = APIStatusError(
        "fixture", response=httpx.Response(status, request=httpx.Request("POST", "https://fixture.test")), body={}
    )
    assert classify_outage(error, 1000) is None


def test_observed_retry_after_is_separate_from_the_local_unknown_reset_policy() -> None:
    error = APIStatusError(
        "fixture",
        response=httpx.Response(
            429, headers={"retry-after": "60"}, request=httpx.Request("POST", "https://fixture.test")
        ),
        body={},
    )
    outage = classify_outage(error, 1000)
    assert outage is not None and outage.retry_at == 1060 and outage.until == 1060


@pytest.mark.parametrize("code,scope", [("subscription_expired", "account"), ("model_not_allowed", "deployment")])
def test_forbidden_model_does_not_disable_other_models_on_the_account(code: str, scope: str) -> None:
    failure = APIStatusError(
        "fixture",
        response=httpx.Response(403, request=httpx.Request("POST", "https://fixture.test")),
        body={"error": {"code": code}},
    )
    outage = classify_outage(failure, 1000)
    assert outage is not None and outage.scope == scope


def test_retry_time_waits_for_both_account_and_model_and_never_claims_reauthentication_will_reset() -> None:
    account = AccountRecord(
        id=UUID(int=1),
        label="Fixture",
        client_id="fixture",
        status="rate_limited",
        unavailable_reason="quota",
        unavailable_until=2000,
        deployment_outages={
            "chatgpt/fixture": AccountOutage(
                reason="provider_unavailable", status="error", code="provider_503", retry_at=None, until=1100
            )
        },
    )
    assert next_attempt(account, 1000, "chatgpt/fixture") == 2000
    assert next_attempt(account.model_copy(update={"status": "login_required"}), 1000, "chatgpt/fixture") is None


@pytest.mark.parametrize("error", [{"resets_at": 2500}, {"resets_in_seconds": 1500}])
def test_provider_quota_reset_is_preserved_without_guessing_a_window(error: Mapping[str, int]) -> None:
    response = httpx.Response(429, request=httpx.Request("POST", "https://fixture.test"))
    outage = classify_outage(APIStatusError("fixture", response=response, body={"error": error}), 1000)
    assert outage is not None and outage.retry_at == 2500 and outage.until == 2500


@pytest.mark.parametrize("secondary_used,expected", [(40, 2000), (100, 3000)])
def test_only_exhausted_windows_delay_recovery(secondary_used: int, expected: int) -> None:
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://fixture.test"),
        headers={
            "x-codex-primary-used-percent": "100",
            "x-codex-primary-reset-at": "2000",
            "x-codex-secondary-used-percent": str(secondary_used),
            "x-codex-secondary-reset-at": "3000",
        },
    )
    outage = classify_outage(APIStatusError("fixture", response=response, body={}), 1000)
    assert outage is not None and outage.retry_at == expected and outage.until == expected


def test_invalid_reset_values_are_unknown_instead_of_breaking_failover() -> None:
    response = httpx.Response(429, request=httpx.Request("POST", "https://fixture.test"))
    outage = classify_outage(
        APIStatusError("fixture", response=response, body={"resets_at": "nan", "resets_in_seconds": "inf"}), 1000
    )
    assert outage is not None and outage.retry_at is None and outage.until == 1600
