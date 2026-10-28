import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from typing import Final

import httpx
from openai import APIConnectionError, APIStatusError, AuthenticationError, PermissionDeniedError, RateLimitError
from pydantic import TypeAdapter

from litellm.exceptions import ContentPolicyViolationError, PaymentRequiredError

from .models import AccountOutage, AccountRecord

_OBJECT_MAP: Final = TypeAdapter(Mapping[str, object])


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
    value: Final = TypeAdapter[str | None](str | None).validate_python(
        response.headers.get("retry-after") if response is not None else None
    )
    if value is None:
        return None
    if value.isdecimal():
        return (time.time() if now is None else now) + int(value)
    try:
        return parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def classify_outage(exception: Exception, now: float) -> AccountOutage | None:
    if isinstance(exception, ContentPolicyViolationError):
        return None
    response: Final = exception.response if isinstance(exception, (APIStatusError, httpx.HTTPStatusError)) else None
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
    reset: Final = retry_at(response, now)
    if status == 401:
        return AccountOutage(
            reason="authentication", status="login_required", code="provider_401", retry_at=reset, until=None
        )
    if status in (402, 403):
        body: Final = getattr(exception, "body", None)
        object_body: Final = _OBJECT_MAP.validate_python(body) if isinstance(body, Mapping) else None
        error: Final = object_body.get("error", object_body) if object_body is not None else None
        code: Final = _OBJECT_MAP.validate_python(error).get("code") if isinstance(error, Mapping) else None
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
            until=reset or now + 600,
            scope="account" if payment else "deployment",
        )
    if status == 429:
        return AccountOutage(
            reason="quota", status="rate_limited", code="provider_429", retry_at=reset, until=reset or now + 600
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
            until=reset or now + 30,
            scope="deployment",
        )
    return None
