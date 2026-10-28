import math
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from typing import Final

import httpx
from openai import APIConnectionError, APIStatusError, AuthenticationError, PermissionDeniedError, RateLimitError
from pydantic import TypeAdapter

from litellm.exceptions import ContentPolicyViolationError, PaymentRequiredError
from litellm.litellm_core_utils.exception_mapping_utils import get_exception_response_headers

from .models import AccountOutage, AccountRecord

_OBJECT_MAP: Final = TypeAdapter(Mapping[str, object])
_HEADER: Final = TypeAdapter[str | None](str | None)


def available(account: AccountRecord, now: float, model: str | None = None) -> bool:
    deployment: Final = account.deployment_outages.get(model) if model else None
    return (
        account.enabled
        and account.status != "login_required"
        and (
            account.unavailable_reason is None
            or (account.unavailable_until is not None and account.unavailable_until <= now)
        )
        and (deployment is None or (deployment.until is not None and deployment.until <= now))
    )


def next_attempt(account: AccountRecord, now: float, model: str | None = None) -> float | None:
    if not account.enabled or account.status == "login_required":
        return None
    deployment: Final = account.deployment_outages.get(model) if model else None
    deadlines: Final = tuple(
        value
        for value in (account.unavailable_until, deployment.until if deployment else None)
        if value is not None and value > now
    )
    return max(deadlines) if deadlines else None


def retry_at(response: httpx.Response | None, now: float | None = None) -> float | None:
    return _retry_after(response.headers if response is not None else None, now)


def _retry_after(headers: httpx.Headers | None, now: float | None = None) -> float | None:
    value: Final = TypeAdapter[str | None](str | None).validate_python(
        headers.get("retry-after") if headers is not None else None
    )
    if value is None:
        return None
    if value.isdecimal():
        return (time.time() if now is None else now) + int(value)
    try:
        return parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _number(value: object) -> float | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    try:
        result: Final = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, OverflowError):
        return None


def _unwrap_error(document: Mapping[str, object]) -> Mapping[str, object]:
    nested: Final = document.get("error")
    return _OBJECT_MAP.validate_python(nested) if isinstance(nested, Mapping) else document


def _error_body(exception: Exception, response: httpx.Response | None) -> Mapping[str, object]:
    body: Final = getattr(exception, "body", None)
    if isinstance(body, Mapping) and body:
        return _unwrap_error(_OBJECT_MAP.validate_python(body))
    if response is None:
        return {}
    try:
        return _unwrap_error(_OBJECT_MAP.validate_json(response.content))
    except (ValueError, httpx.ResponseNotRead):
        return {}


def _window_reset(headers: httpx.Headers | None, window: str) -> float | None:
    if headers is None:
        return None
    used: Final = _number(_HEADER.validate_python(headers.get(f"x-codex-{window}-used-percent")))
    reset: Final = _number(_HEADER.validate_python(headers.get(f"x-codex-{window}-reset-at")))
    return reset if used is not None and used >= 100 and reset is not None and reset > 0 else None


def _quota_reset(headers: httpx.Headers | None, body: Mapping[str, object], now: float) -> float | None:
    absolute: Final = _number(body.get("resets_at"))
    seconds: Final = _number(body.get("resets_in_seconds"))
    payload: Final = (
        absolute
        if absolute is not None and absolute > 0
        else now + seconds
        if seconds is not None and seconds >= 0
        else None
    )
    known: Final = tuple(
        value
        for value in (payload, _window_reset(headers, "primary"), _window_reset(headers, "secondary"))
        if value is not None
    )
    return max(known) if known else None


def classify_outage(exception: Exception, now: float) -> AccountOutage | None:
    if isinstance(exception, ContentPolicyViolationError):
        return None
    response: Final = exception.response if isinstance(exception, (APIStatusError, httpx.HTTPStatusError)) else None
    headers: Final = get_exception_response_headers(exception)
    raw_status: Final = getattr(exception, "status_code", None)
    http_status: Final = (
        response.status_code if response is not None else raw_status if isinstance(raw_status, int) else None
    )
    status: Final = (
        429
        if isinstance(exception, RateLimitError)
        else 402
        if isinstance(exception, PaymentRequiredError)
        else 401
        if isinstance(exception, AuthenticationError)
        else 403
        if isinstance(exception, PermissionDeniedError)
        else http_status
    )
    body: Final = _error_body(exception, response)
    observed: Final = tuple(
        value
        for value in (_retry_after(headers, now), _quota_reset(headers, body, now) if status == 429 else None)
        if value is not None
    )
    reset: Final = max(observed) if observed else None
    if status == 401:
        return AccountOutage(
            reason="authentication", status="login_required", code="provider_401", retry_at=reset, until=None
        )
    if status in (402, 403):
        code: Final = body.get("code")
        payment: Final = status == 402 or code in (
            "insufficient_quota",
            "billing_hard_limit_reached",
            "subscription_expired",
        )
        return AccountOutage(
            reason="payment" if payment else "access_denied",
            status="error",
            code=f"provider_{status}",
            retry_at=reset,
            until=reset if reset is not None else now + 600,
            scope="account" if payment else "deployment",
        )
    if status == 429:
        return AccountOutage(
            reason="quota",
            status="rate_limited",
            code="provider_429",
            retry_at=reset,
            until=reset if reset is not None else now + 600,
        )
    if (
        status == 408
        or (status is not None and status >= 500)
        or isinstance(exception, (APIConnectionError, httpx.TransportError))
    ):
        return AccountOutage(
            reason="provider_unavailable",
            status="error",
            code=f"provider_{status}" if status else "connection_failed",
            retry_at=reset,
            until=reset if reset is not None else now + 30,
            scope="deployment",
        )
    return None
