import time
from collections.abc import Mapping, Sequence
from email.utils import parsedate_to_datetime
from typing import Final
from uuid import UUID

import httpx
from fastapi import HTTPException
from openai import APIStatusError
from pydantic import TypeAdapter

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.chatgpt.common_utils import is_chatgpt_oauth_key
from litellm.types.utils import CallTypes

from .models import ACCOUNT_ID_FIELD, ACCOUNT_METADATA_KEY, AccountFailure
from .service import AccountService, accounts_logger, deployment_id

_OBJECT_MAP: Final = TypeAdapter(dict[str, object])
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
    }
)


def _mapping(value: object) -> Mapping[str, object]:
    return _OBJECT_MAP.validate_python(value) if isinstance(value, Mapping) else {}


def _metadata(data: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(data.get("litellm_metadata") or data.get("metadata"))


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
        selected: Final = self.service.selection(model, previous if isinstance(previous, str) else None)
        if isinstance(selected, AccountFailure):
            raise HTTPException(selected.status, detail=selected.model_dump())
        metadata: Final = {
            **{key: value for key, value in _metadata(data).items() if key != ACCOUNT_METADATA_KEY},
            ACCOUNT_METADATA_KEY: str(selected),
            "_managed_chatgpt_route": model,
        }
        return {**data, "metadata": metadata, "litellm_metadata": metadata}

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
        return [
            dict(deployment)
            for deployment in healthy_deployments
            if _mapping(deployment.get("model_info")).get(ACCOUNT_ID_FIELD) == account_id
        ]

    async def async_filter_listed_models(
        self,
        user_api_key_dict: object,
        model_names: Sequence[str],
    ) -> Sequence[str]:
        return model_names

    def _log_result(self, kwargs: Mapping[str, object], result: str) -> UUID | None:
        parameters: Final = _mapping(kwargs.get("litellm_params"))
        metadata: Final = _metadata(kwargs) or _metadata(parameters)
        route: Final = metadata.get("_managed_chatgpt_route")
        account_id: Final = metadata.get(ACCOUNT_METADATA_KEY)
        if route not in self.service.routes() or not isinstance(account_id, str):
            return
        try:
            profile_id: Final = UUID(account_id)
        except ValueError:
            return None
        if self.service.record(profile_id) is None:
            return None
        template: Final = next(item for item in self.service.templates if item.model_name == route)
        actual_id: Final = _mapping(metadata.get("model_info")).get("id")
        if actual_id != deployment_id(profile_id, template):
            return None
        accounts_logger.info(
            "ChatGPT accounts request_finished route=%s model=%s profile=%s result=%s",
            route,
            template.litellm_params.model,
            account_id,
            result,
        )
        return profile_id

    async def async_post_call_success_deployment_hook(
        self,
        request_data: Mapping[str, object],
        response: object,
        call_type: CallTypes | None,
    ) -> None:
        profile_id: Final = self._log_result(request_data, "success")
        if profile_id is None:
            return
        account: Final = self.service.record(profile_id)
        if account is not None and account.enabled and account.status != "ready":
            self.service.update_account(
                profile_id, {"status": "ready", "error_code": None, "retry_at": None}, advance_revision=False
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
        account: Final = self.service.record(profile_id)
        if account is None or not account.enabled:
            return
        response: Final = exception.response if isinstance(exception, (APIStatusError, httpx.HTTPStatusError)) else None
        status: Final = response.status_code if response is not None else None
        self.service.update_account(
            profile_id,
            {
                "status": "login_required" if status == 401 else "rate_limited" if status == 429 else "error",
                "error_code": f"provider_{status}" if status else "provider_error",
                "retry_at": self.retry_at(response),
            },
            advance_revision=False,
        )

    @staticmethod
    def retry_at(response: httpx.Response | None) -> float | None:
        value: Final = TypeAdapter[str | None](str | None).validate_python(
            response.headers.get("retry-after") if response is not None else None
        )
        if value is None:
            return None
        if value.isdecimal():
            return time.time() + int(value)
        try:
            return parsedate_to_datetime(value).timestamp()
        except (TypeError, ValueError, OverflowError):
            return None
