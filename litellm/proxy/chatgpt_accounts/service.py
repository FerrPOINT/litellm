import asyncio
import hashlib
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol
from uuid import UUID

import httpx
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.llms.chatgpt.authenticator import get_cached_authenticator
from litellm.llms.chatgpt.common_utils import GetAccessTokenError, get_chatgpt_default_headers
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router import Router
from litellm.types.router import Deployment

from .models import ACCOUNT_ID_FIELD, AccountBindings, AccountFailure, AccountRecord, AccountsState, ProbeResult
from .store import AccountStore

accounts_logger: Final = verbose_proxy_logger.getChild("chatgpt_accounts")


class CatalogModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    slug: str


class ModelCatalog(BaseModel):
    model_config = ConfigDict(frozen=True)
    models: tuple[CatalogModel, ...]


class ProbeOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    output: tuple[object, ...]


class ManagedDeployment(BaseModel):
    model_config = ConfigDict(frozen=True)
    managed_chatgpt_account_id: str | None = None


class AccountProbe(Protocol):
    async def check(self, auth_file: Path, models: tuple[str, ...]) -> ProbeResult: ...


class OpenAIAccountProbe:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def check(self, auth_file: Path, models: tuple[str, ...]) -> ProbeResult:
        import litellm

        authenticator: Final = get_cached_authenticator(str(auth_file))
        try:
            token: Final = await asyncio.to_thread(authenticator.get_access_token)
            account_id: Final = await asyncio.to_thread(authenticator.get_account_id)
            response: Final = await self.client.get(
                authenticator.get_api_base().rstrip("/") + "/models",
                headers=get_chatgpt_default_headers(token, account_id),
                params={"client_version": "0.162.0"},
            )
            response.raise_for_status()
            catalog: Final = ModelCatalog.model_validate(response.json())
            slugs: Final = tuple(item.slug for item in catalog.models)
            required: Final = models or slugs[:1]
            if not required:
                return ProbeResult(
                    error=AccountFailure(code="no_models", message="The account has no available models.")
                )
            for model in required:
                if (
                    not isinstance(
                        result := await litellm.aresponses(
                            model="chatgpt/" + model,
                            chatgpt_auth_file=str(auth_file),
                            input=[{"role": "user", "content": [{"type": "input_text", "text": "Reply exactly OK"}]}],
                            reasoning={"effort": "high"},
                            timeout=45,
                        ),
                        BaseModel,
                    )
                    or not ProbeOutput.model_validate(result.model_dump()).output
                ):
                    return ProbeResult(
                        error=AccountFailure(code="empty_response", message="The model returned no output.")
                    )
            return ProbeResult(models=slugs)
        except (httpx.HTTPError, OpenAIError, ValueError, OSError, GetAccessTokenError) as exc:
            status: Final = getattr(exc, "status_code", None)
            return ProbeResult(
                error=AccountFailure(
                    code="login_required" if status == 401 else "rate_limited" if status == 429 else "probe_failed",
                    message=(
                        "Sign in to this ChatGPT account again."
                        if status == 401
                        else "ChatGPT rate limit reached; no alternate ChatGPT account was selected."
                        if status == 429
                        else "This ChatGPT account cannot access a required model."
                        if status == 403
                        else "ChatGPT model verification failed; check provider availability and the configured model."
                    ),
                    status=status if isinstance(status, int) and 400 <= status < 600 else 502,
                )
            )


def deployment_id(account_id: UUID, template: Deployment) -> str:
    value: Final = f"{account_id}:{template.model_name}:{template.litellm_params.model}"
    return "chatgpt-account-" + hashlib.sha256(value.encode()).hexdigest()[:32]


def selected_account(bindings: AccountBindings, route: str) -> UUID | None:
    return bindings.routes.get(route) or bindings.default_account_id


class AccountService:
    def __init__(self, store: AccountStore, router: Router, probe: AccountProbe) -> None:
        self.store = store
        self.router = router
        self.probe = probe
        self.templates = tuple(
            item
            for item in TypeAdapter(tuple[Deployment, ...]).validate_python(router.get_model_list() or ())
            if self.is_template(item)
        )
        self.register_deployments(self.store.read())

    @staticmethod
    def is_template(item: Deployment) -> bool:
        return (
            item.litellm_params.model.startswith("chatgpt/")
            and not ManagedDeployment.model_validate(item.model_info.model_dump()).managed_chatgpt_account_id
        )

    def account_deployment(self, account: AccountRecord, template: Deployment) -> Deployment:
        return template.model_copy(
            update={
                "litellm_params": template.litellm_params.model_copy(
                    update={
                        "chatgpt_auth_file": str(self.store.auth_file(account.id)),
                        "api_key": None,
                    }
                ),
                "model_info": template.model_info.model_copy(
                    update={
                        "id": deployment_id(account.id, template),
                        ACCOUNT_ID_FIELD: str(account.id),
                    }
                ),
            }
        )

    def register_deployments(self, state: AccountsState) -> None:
        for account in state.accounts:
            for template in self.templates:
                self.router.add_deployment(self.account_deployment(account, template))

    def routes(self) -> tuple[str, ...]:
        return tuple(template.model_name for template in self.templates)

    def record(self, account_id: UUID) -> AccountRecord | None:
        return next((item for item in self.store.read().accounts if item.id == account_id), None)

    def save_account(self, account: AccountRecord) -> AccountsState | AccountFailure:
        def replace(state: AccountsState) -> AccountsState:
            accounts: Final = tuple(item for item in state.accounts if item.id != account.id) + (account,)
            return state.model_copy(update={"accounts": accounts})

        result: Final = self.store.update(replace)
        if isinstance(result, AccountsState):
            self.register_deployments(result)
        return result

    async def check(self, account_id: UUID, models: tuple[str, ...] = ()) -> AccountRecord | AccountFailure:
        account: Final = self.record(account_id)
        if account is None:
            return AccountFailure(code="account_not_found", message="Account not found.", status=404)
        result: Final = await self.probe.check(self.store.auth_file(account_id), models)
        updates: Final = {
            "status": (
                "ready"
                if result.error is None
                else "login_required"
                if result.error.code == "login_required"
                else "rate_limited"
                if result.error.code == "rate_limited"
                else "error"
            ),
            "models": result.models if result.error is None else account.models,
            "last_checked_at": time.time(),
            "error_code": result.error.code if result.error else None,
        }
        saved: Final = self.update_account(account_id, updates, advance_revision=False)
        return result.error if result.error else saved

    async def bind(self, bindings: AccountBindings, revision: int) -> AccountsState | AccountFailure:
        current: Final = self.store.read()
        if bindings.default_account_id is None:
            return AccountFailure(code="account_required", message="Select a hub default account.")
        if current.revision != revision:
            return AccountFailure(code="state_changed", message="Settings changed. Reload and retry.", status=409)
        if any(route not in ("main-dev", "app-test") or route not in self.routes() for route in bindings.routes):
            return AccountFailure(code="unknown_route", message="The route is not managed by ChatGPT accounts.")
        for template in self.templates:
            if (target := selected_account(bindings, template.model_name)) is None:
                return AccountFailure(code="account_required", message="Select an account for every ChatGPT route.")
            if (account := self.record(target)) is None or not account.enabled:
                return AccountFailure(
                    code="account_unavailable", message="The selected account is not enabled.", status=409
                )
        account_ids: Final = frozenset(selected_account(bindings, template.model_name) for template in self.templates)
        for account_id in account_ids:
            if account_id is None:
                continue
            if (failure := await self.verify_binding(bindings, account_id)) is not None:
                return failure
        result: Final = self.store.update(
            lambda state: state.model_copy(update={"bindings": bindings}),
            expected_revision=revision,
        )
        if isinstance(result, AccountsState):
            accounts_logger.info(
                "ChatGPT accounts bindings_changed revision=%s reason=manual default_profile=%s routes=%s",
                result.revision,
                bindings.default_account_id,
                tuple(f"{route}={profile}" for route, profile in bindings.routes.items()),
            )
        return result

    async def verify_binding(self, bindings: AccountBindings, account_id: UUID) -> AccountFailure | None:
        required: Final = tuple(
            dict.fromkeys(
                template.litellm_params.model.removeprefix("chatgpt/")
                for template in self.templates
                if selected_account(bindings, template.model_name) == account_id
            )
        )
        result: Final = await self.probe.check(self.store.auth_file(account_id), required)
        return result.error

    def update_account(
        self, account_id: UUID, updates: Mapping[str, object], *, advance_revision: bool = True
    ) -> AccountRecord | AccountFailure:
        def update(state: AccountsState) -> AccountsState | AccountFailure:
            if not any(item.id == account_id for item in state.accounts):
                return AccountFailure(code="account_not_found", message="Account not found.", status=404)
            return state.model_copy(
                update={
                    "accounts": tuple(
                        item.model_copy(update=updates) if item.id == account_id else item for item in state.accounts
                    )
                }
            )

        result: Final = self.store.update(update, advance_revision=advance_revision)
        if isinstance(result, AccountFailure):
            return result
        return next(item for item in result.accounts if item.id == account_id)

    def rename(self, account_id: UUID, label: str) -> AccountRecord | AccountFailure:
        return self.update_account(account_id, {"label": label})

    def disable(self, account_id: UUID) -> AccountsState | AccountFailure:
        def update(state: AccountsState) -> AccountsState | AccountFailure:
            if state.bindings.default_account_id == account_id or any(
                selected_account(state.bindings, route) == account_id for route in self.routes()
            ):
                return AccountFailure(
                    code="account_in_use", message="Reassign its routes before disconnecting.", status=409
                )
            if not any(account.id == account_id for account in state.accounts):
                return AccountFailure(code="account_not_found", message="Account not found.", status=404)
            return state.model_copy(
                update={
                    "accounts": tuple(
                        account.model_copy(update={"enabled": False, "status": "disabled"})
                        if account.id == account_id
                        else account
                        for account in state.accounts
                    )
                }
            )

        return self.store.update(update)

    def selection(self, route: str, previous_response_id: str | None) -> UUID | AccountFailure:
        state: Final = self.store.read()
        previous_model_id: Final = (
            ResponsesAPIRequestUtils.get_model_id_from_response_id(previous_response_id)
            if previous_response_id
            else None
        )
        owners: Final = tuple(
            account.id
            for account in state.accounts
            if any(deployment_id(account.id, template) == previous_model_id for template in self.templates)
        )
        if previous_response_id and len(owners) != 1:
            return AccountFailure(code="unknown_response_owner", message="Start a new conversation.", status=409)
        target: Final = owners[0] if previous_response_id else selected_account(state.bindings, route)
        account: Final = next((item for item in state.accounts if item.id == target), None)
        if account is None or not account.enabled:
            return AccountFailure(code="account_unavailable", message="Choose an enabled ChatGPT account.", status=409)
        return account.id
