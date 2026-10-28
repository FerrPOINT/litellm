from collections.abc import Mapping, Sequence
from typing import Final
from uuid import UUID

import httpx
from fastapi import HTTPException
from pydantic import TypeAdapter

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.chatgpt.common_utils import is_chatgpt_oauth_key
from litellm.types.utils import CallTypes

from .availability import classify_outage, retry_at
from .models import ACCOUNT_ID_FIELD, ACCOUNT_METADATA_KEY, AccountFailure
from .service import AccountService, accounts_logger, deployment_id

_OBJECT_MAP: Final = TypeAdapter(dict[str, object])
_RETRY_IDS: Final = TypeAdapter(tuple[str, ...])
_CREDENTIAL_FIELDS: Final = frozenset(
    {
        "chatgpt_auth_file",
        "api_key",
        "api_base",
        "custom_llm_provider",
        "client",
        "deployment",
        "specific_deployment",
        "model_info",
        "litellm_params",
        "extra_body",
        "num_retries",
        "fallbacks",
        "disable_fallbacks",
        "retry_policy",
        "model_group_retry_policy",
        "_retry_skipped_deployment_ids",
    }
)


def _mapping(value: object) -> Mapping[str, object]:
    return _OBJECT_MAP.validate_python(value) if isinstance(value, Mapping) else {}


def _metadata(data: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(data.get("litellm_metadata") or data.get("metadata"))


def _account_failure(failure: AccountFailure) -> HTTPException:
    return HTTPException(
        failure.status,
        detail=failure.model_dump(),
        headers={"Retry-After": str(failure.retry_after)} if failure.retry_after is not None else None,
    )


class AccountsRoutingHook(CustomLogger):
    def __init__(self, service: AccountService) -> None:
        self.turn_off_message_logging = True
        self.message_logging = False
        self.service = service

    async def async_pre_call_hook(
        self,
        user_api_key_dict: object,
        cache: object,
        data: Mapping[str, object],
        call_type: str,
    ) -> dict[str, object] | None:  # mutable-ok: CustomLogger requires a mutable request result
        model: Final = data.get("model")
        if not isinstance(model, str):
            return None
        if model not in self.service.routes():
            deployment: Final = self.service.router.get_deployment(model_id=model)
            if model.startswith("chatgpt/") or (deployment and deployment.litellm_params.model.startswith("chatgpt/")):
                raise HTTPException(
                    400, detail={"code": "managed_route_required", "message": "Choose a configured ChatGPT route."}
                )
            return None
        if any(key in data for key in _CREDENTIAL_FIELDS):
            raise HTTPException(
                400, detail={"code": "managed_credentials", "message": "Account credentials are server-managed."}
            )
        extra_headers: Final = _mapping(data.get("extra_headers"))
        if any(key.lower() in {"authorization", "chatgpt-account-id"} for key in extra_headers):
            raise HTTPException(
                400, detail={"code": "managed_credentials", "message": "Account headers are server-managed."}
            )
        proxy_headers: Final = _mapping(_mapping(data.get("proxy_server_request")).get("headers"))
        authorization: Final = proxy_headers.get("authorization")
        if isinstance(authorization, str) and is_chatgpt_oauth_key(authorization):
            raise HTTPException(
                400, detail={"code": "managed_credentials", "message": "Use the gateway inference key."}
            )
        previous: Final = data.get("previous_response_id")
        snapshot: Final = self.service.store.read()
        selected: Final = self.service.selection(model, previous if isinstance(previous, str) else None, snapshot)
        automatic: Final = snapshot.bindings.automatic_failover and not previous
        if isinstance(selected, AccountFailure):
            raise _account_failure(selected)
        metadata: Final = {
            **{key: value for key, value in _metadata(data).items() if key != ACCOUNT_METADATA_KEY},
            ACCOUNT_METADATA_KEY: str(selected),
            "_managed_chatgpt_route": model,
            "_managed_chatgpt_scope": self.service.scope(snapshot.bindings, model),
            "_managed_chatgpt_revision": snapshot.revision,
            "_managed_chatgpt_automatic": bool(automatic),
            "_managed_chatgpt_started_at": self.service.clock(),
        }
        attempts: Final = max(0, sum(account.enabled for account in snapshot.accounts) - 1)
        retry_options: Final[Mapping[str, object]] = (
            {
                "num_retries": attempts,
                "disable_fallbacks": True,
                "fallbacks": [],
                "model_group_retry_policy": {
                    model: {
                        "DefaultRetries": attempts,
                        "BadRequestErrorRetries": 0,
                        "PaymentRequiredErrorRetries": attempts,
                        "AuthenticationErrorRetries": attempts,
                        "PermissionDeniedErrorRetries": attempts,
                        "RateLimitErrorRetries": attempts,
                        "TimeoutErrorRetries": attempts,
                        "InternalServerErrorRetries": attempts,
                        "ServiceUnavailableErrorRetries": attempts,
                        "NotFoundErrorRetries": 0,
                        "ContentPolicyViolationErrorRetries": 0,
                        "retry_same_deployment": False,
                    }
                },
            }
            if automatic
            else {"num_retries": 0, "disable_fallbacks": True, "fallbacks": ()}
            if snapshot.bindings.automatic_failover
            else {}
        )
        return {**data, **retry_options, "metadata": metadata, "litellm_metadata": metadata}

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: Sequence[Mapping[str, object]],
        messages: object,
        request_kwargs: Mapping[str, object] | None = None,
        parent_otel_span: object = None,
    ) -> list[dict[str, object]]:  # mutable-ok: Router requires mutable deployment dictionaries
        if model not in self.service.routes():
            return [dict(deployment) for deployment in healthy_deployments]
        account_id: Final = _metadata(request_kwargs or {}).get(ACCOUNT_METADATA_KEY)
        if not isinstance(account_id, str):
            raise HTTPException(
                409, detail={"code": "account_not_resolved", "message": "No managed account was selected."}
            )
        metadata: Final = _metadata(request_kwargs or {})
        skipped: Final = _RETRY_IDS.validate_python((request_kwargs or {}).get("_retry_skipped_deployment_ids") or ())
        candidates: Final = tuple(
            deployment
            for deployment in healthy_deployments
            if _mapping(deployment.get("model_info")).get("id") not in skipped
        )
        healthy: Final = frozenset(
            UUID(value)
            for deployment in candidates
            if isinstance(value := _mapping(deployment.get("model_info")).get(ACCOUNT_ID_FIELD), str)
        )
        scope: Final = metadata.get("_managed_chatgpt_scope")
        selected: Final = (
            self.service.retry_selection(
                UUID(account_id), scope if isinstance(scope, str) else "default", healthy, model
            )
            if metadata.get("_managed_chatgpt_automatic") is True
            else UUID(account_id)
        )
        if isinstance(selected, AccountFailure):
            raise _account_failure(selected)
        return [
            dict(deployment)
            for deployment in candidates
            if _mapping(deployment.get("model_info")).get(ACCOUNT_ID_FIELD) == str(selected)
        ]

    async def async_filter_listed_models(
        self,
        user_api_key_dict: object,
        model_names: Sequence[str],
    ) -> Sequence[str]:
        return model_names

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, object],
        original_exception: Exception,
        user_api_key_dict: object,
        traceback_str: str | None = None,
    ) -> HTTPException | None:
        metadata: Final = _metadata(request_data)
        route: Final = metadata.get("_managed_chatgpt_route")
        if metadata.get("_managed_chatgpt_automatic") is not True or not isinstance(route, str):
            return None
        if request_data.get("model") != route or route not in self.service.routes():
            return None
        if classify_outage(original_exception, self.service.clock()) is None:
            return None
        selected: Final = self.service.selection(route, None)
        return _account_failure(selected) if isinstance(selected, AccountFailure) else None

    def profile_for_attempt(self, kwargs: Mapping[str, object]) -> UUID | None:
        parameters: Final = _mapping(kwargs.get("litellm_params"))
        metadata: Final = _metadata(kwargs) or _metadata(parameters)
        route: Final = metadata.get("_managed_chatgpt_route")
        if not isinstance(route, str) or route not in self.service.routes():
            return None
        template: Final = next(item for item in self.service.templates if item.model_name == route)
        actual_id: Final = _mapping(metadata.get("model_info")).get("id")
        return next(
            (item.id for item in self.service.store.read().accounts if deployment_id(item.id, template) == actual_id),
            None,
        )

    def _log_result(self, kwargs: Mapping[str, object], result: str) -> UUID | None:
        profile_id: Final = self.profile_for_attempt(kwargs)
        if profile_id is None:
            return None
        metadata: Final = _metadata(kwargs) or _metadata(_mapping(kwargs.get("litellm_params")))
        route: Final = metadata.get("_managed_chatgpt_route")
        template: Final = next(item for item in self.service.templates if item.model_name == route)
        accounts_logger.info(
            "ChatGPT accounts request_finished route=%s model=%s profile=%s result=%s",
            route,
            template.litellm_params.model,
            profile_id,
            result,
        )
        return profile_id

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        profile_id: Final = self.profile_for_attempt(kwargs)
        metadata: Final = _metadata(kwargs)
        scope: Final = metadata.get("_managed_chatgpt_scope")
        revision: Final = metadata.get("_managed_chatgpt_revision")
        if profile_id is None or metadata.get("_managed_chatgpt_automatic") is not True:
            return
        if isinstance(scope, str) and isinstance(revision, int):
            self.service.activate(profile_id, scope, revision)

    async def async_post_call_success_deployment_hook(
        self,
        request_data: Mapping[str, object],
        response: object,
        call_type: CallTypes | None,
    ) -> None:
        profile_id: Final = self._log_result(request_data, "success")
        if profile_id is None:
            return
        started: Final = _metadata(request_data).get("_managed_chatgpt_started_at")
        route: Final = _metadata(request_data).get("_managed_chatgpt_route")
        model: Final = next(
            (item.litellm_params.model for item in self.service.templates if item.model_name == route), None
        )
        self.service.record_success(
            profile_id, float(started) if isinstance(started, (float, int)) else self.service.clock(), model
        )

    async def async_post_call_failure_deployment_hook(
        self,
        request_data: Mapping[str, object],
        exception: Exception,
        call_type: CallTypes | None,
        fallback_depth: int | None = None,
    ) -> None:
        profile_id: Final = self._log_result(request_data, "failure")
        if profile_id is None:
            return
        outage: Final = classify_outage(exception, self.service.clock())
        if outage is not None:
            route: Final = _metadata(request_data).get("_managed_chatgpt_route")
            model: Final = next(
                (item.litellm_params.model for item in self.service.templates if item.model_name == route), None
            )
            self.service.record_outage(profile_id, outage, model)

    @staticmethod
    def retry_at(response: httpx.Response | None) -> float | None:
        return retry_at(response)
