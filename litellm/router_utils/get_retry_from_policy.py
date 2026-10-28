"""Resolve how many retries a RetryPolicy grants for a given exception."""

from collections.abc import Callable, Mapping
from itertools import chain
from types import MappingProxyType
from typing import Final

from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    InternalServerError,
    PaymentRequiredError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.types.router import RetryPolicy

_RETRIES_BY_EXCEPTION_TYPE: Final[Mapping[type, Callable[[RetryPolicy], int | None]]] = MappingProxyType(
    {
        PaymentRequiredError: lambda policy: policy.PaymentRequiredErrorRetries,
        PermissionDeniedError: lambda policy: policy.PermissionDeniedErrorRetries,
        AuthenticationError: lambda policy: policy.AuthenticationErrorRetries,
        Timeout: lambda policy: policy.TimeoutErrorRetries,
        RateLimitError: lambda policy: policy.RateLimitErrorRetries,
        ContentPolicyViolationError: lambda policy: policy.ContentPolicyViolationErrorRetries,
        BadRequestError: lambda policy: policy.BadRequestErrorRetries,
        ServiceUnavailableError: lambda policy: policy.ServiceUnavailableErrorRetries,
        InternalServerError: lambda policy: policy.InternalServerErrorRetries,
    }
)


def _retries_for_http_status(exception: Exception, policy: RetryPolicy) -> int | None:
    if isinstance(exception, ContentPolicyViolationError):
        return None
    status_code: Final = getattr(exception, "status_code", None)
    match status_code:
        case 401:
            return policy.AuthenticationErrorRetries
        case 402:
            return policy.PaymentRequiredErrorRetries
        case 403:
            return policy.PermissionDeniedErrorRetries
        case 404:
            return policy.NotFoundErrorRetries
        case 408:
            return policy.TimeoutErrorRetries
        case 429:
            return policy.RateLimitErrorRetries
        case 500 | 502 | 504:
            return policy.InternalServerErrorRetries
        case 503:
            return policy.ServiceUnavailableErrorRetries
        case _:
            return None


def resolve_retry_policy(
    retry_policy: RetryPolicy | Mapping[str, int | None] | None,
    model_group: str | None,
    model_group_retry_policy: Mapping[str, RetryPolicy | Mapping[str, int | None]] | None,
) -> RetryPolicy | None:
    selected: Final = (
        model_group_retry_policy[model_group]
        if model_group_retry_policy is not None and model_group is not None and model_group in model_group_retry_policy
        else retry_policy
    )
    if isinstance(selected, Mapping):
        return RetryPolicy.model_validate(selected)
    return selected


def get_num_retries_from_retry_policy(
    exception: Exception,
    retry_policy: RetryPolicy | Mapping[str, int | None] | None = None,
    model_group: str | None = None,
    model_group_retry_policy: Mapping[str, RetryPolicy | Mapping[str, int | None]] | None = None,
) -> int | None:
    """Prefer explicitly configured HTTP status retries, then the exception's most specific class."""
    policy: Final = resolve_retry_policy(retry_policy, model_group, model_group_retry_policy)
    if policy is None:
        return None
    by_class: Final = (
        _RETRIES_BY_EXCEPTION_TYPE[cls](policy) for cls in type(exception).__mro__ if cls in _RETRIES_BY_EXCEPTION_TYPE
    )
    configured: Final = chain((_retries_for_http_status(exception, policy),), by_class)
    return next((retries for retries in configured if retries is not None), policy.DefaultRetries)


def reset_retry_policy() -> RetryPolicy:
    return RetryPolicy()
