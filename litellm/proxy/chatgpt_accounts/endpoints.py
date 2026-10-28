import hmac
import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Final, TypeAlias
from uuid import UUID

import httpx
import jwt
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from litellm.llms.chatgpt.authenticator import chatgpt_auth_logger

from .login import LoginOperations
from .models import AccountBindings, AccountFailure, AccountRecord, AccountsState, LoginView
from .routing import AccountsRoutingHook
from .service import AccountService, OpenAIAccountProbe, accounts_logger
from .store import AccountStore

_COOKIE: Final = "litellm_chatgpt_accounts_admin"
Label: TypeAlias = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class LoginRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    key: str = Field(min_length=1, max_length=512)


class NoOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AdminSession(BaseModel):
    model_config = ConfigDict(frozen=True)
    expires_at: int


class LabelRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    label: Label


class AccountLoginRequest(LabelRequest):
    account_id: UUID | None = None


class BindingsRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    revision: int = Field(ge=0)
    bindings: AccountBindings


class RouteView(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    model: str
    context_length: int | None = None


class AccountsView(AccountsState):
    routes: tuple[RouteView, ...]
    quota_source: str | None = None


class RequestApplication(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    app: FastAPI


class Controller:
    def __init__(self, service: AccountService, logins: LoginOperations, admin_key: str) -> None:
        self.service = service
        self.logins = logins
        self.admin_key = admin_key


@asynccontextmanager
async def accounts_lifespan(app: FastAPI) -> AsyncGenerator[None]:
    import litellm
    from litellm.proxy import proxy_server

    directory: Final = os.environ.get("LITELLM_CHATGPT_ACCOUNTS_DIR")
    if not directory:
        yield
        return
    admin_key: Final = os.environ.get("LITELLM_CHATGPT_ACCOUNTS_ADMIN_KEY", "")
    if len(admin_key) < 32:
        raise RuntimeError("Managed ChatGPT accounts require a separate admin key of at least 32 characters")
    if proxy_server.llm_router is None:
        raise RuntimeError("Managed ChatGPT accounts require a configured Router")
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        service: Final = AccountService(
            AccountStore(Path(directory)),
            proxy_server.llm_router,
            OpenAIAccountProbe(client),
        )
        logins: Final = LoginOperations(service, client)
        hook: Final = AccountsRoutingHook(service)
        app.state.chatgpt_accounts = Controller(service, logins, admin_key)  # rebind-ok: lifespan owns FastAPI state
        litellm.callbacks = [*litellm.callbacks, hook]
        account_log_level: Final = accounts_logger.level
        auth_log_level: Final = chatgpt_auth_logger.level
        accounts_logger.setLevel(logging.INFO)
        chatgpt_auth_logger.setLevel(logging.INFO)
        try:
            yield
        finally:
            await logins.close()
            accounts_logger.setLevel(account_log_level)
            chatgpt_auth_logger.setLevel(auth_log_level)
            litellm.callbacks = [callback for callback in litellm.callbacks if callback is not hook]
            del app.state.chatgpt_accounts  # rebind-ok: lifespan releases its FastAPI state


router: Final = APIRouter(prefix="/chatgpt/accounts", tags=["chatgpt accounts"], lifespan=accounts_lifespan)


def controller(request: Request) -> Controller:
    value: Final = getattr(RequestApplication.model_validate(request.scope).app.state, "chatgpt_accounts", None)
    if not isinstance(value, Controller):
        raise HTTPException(404, detail="Managed ChatGPT accounts are not enabled")
    return value


def require_admin(request: Request, value: Annotated[Controller, Depends(controller)]) -> Controller:
    cookie: Final = request.cookies.get(_COOKIE)
    try:
        claims: Final = jwt.decode(
            cookie or "",
            value.admin_key,
            algorithms=["HS256"],
            audience="chatgpt-accounts-admin",
            issuer="litellm",
            options={"require": ["exp", "iat", "aud", "iss"]},
        )
        if claims.get("scope") != "chatgpt-accounts:manage":
            raise HTTPException(403, detail="Administrative session required")
    except jwt.InvalidTokenError as exc:
        raise HTTPException(401, detail="Administrative session required") from exc
    origin: Final = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, detail="Cross-origin account management is not allowed")
    return value


Admin: TypeAlias = Annotated[Controller, Depends(require_admin)]


def public_failure(failure: AccountFailure) -> HTTPException:
    return HTTPException(failure.status, detail={"code": failure.code, "message": failure.message})


@router.post(
    "/session",
    response_model=AdminSession,
    description="Create a one-hour administrative cookie using the separate account management key.",
)
async def login_session(
    body: LoginRequest, request: Request, response: Response, value: Annotated[Controller, Depends(controller)]
) -> AdminSession:
    origin: Final = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, detail="Cross-origin sign-in is not allowed")
    if not hmac.compare_digest(body.key.encode(), value.admin_key.encode()):
        raise HTTPException(401, detail="Invalid administrative key")
    now: Final = int(time.time())
    token: Final = jwt.encode(
        {
            "iat": now,
            "exp": now + 3600,
            "aud": "chatgpt-accounts-admin",
            "iss": "litellm",
            "scope": "chatgpt-accounts:manage",
        },
        value.admin_key,
        algorithm="HS256",
    )
    response.set_cookie(
        _COOKIE,
        token,
        max_age=3600,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path="/chatgpt",
    )
    return AdminSession(expires_at=now + 3600)


@router.delete("/session", status_code=204)
async def logout_session(response: Response, value: Admin) -> None:
    response.delete_cookie(_COOKIE, path="/chatgpt")


@router.get("", response_model=AccountsView)
async def list_accounts(value: Admin) -> AccountsView:
    state: Final = value.service.store.read()
    return AccountsView(
        version=state.version,
        revision=state.revision,
        accounts=state.accounts,
        bindings=state.bindings,
        active_accounts=state.active_accounts,
        last_switches=state.last_switches,
        routes=tuple(
            RouteView.model_validate(
                {
                    "name": item.model_name,
                    "model": item.litellm_params.model,
                    "context_length": item.model_info.model_dump().get("max_input_tokens"),
                }
            )
            for item in value.service.templates
        ),
    )


@router.post(
    "/logins",
    response_model=LoginView,
    description="Start a pending ChatGPT device sign-in without changing current assignments.",
)
async def start_login(body: AccountLoginRequest, value: Admin) -> LoginView:
    result: Final = await value.logins.start(body.label, body.account_id)
    if isinstance(result, AccountFailure):
        raise public_failure(result)
    return result


@router.get("/logins/{login_id}", response_model=LoginView)
async def login_status(login_id: UUID, value: Admin) -> LoginView:
    result: Final = value.logins.views.get(login_id)
    if result is None:
        raise HTTPException(
            404, detail={"code": "login_not_found", "message": "Sign-in expired or the application restarted."}
        )
    return result


@router.delete("/logins/{login_id}", response_model=LoginView)
async def cancel_login(login_id: UUID, value: Admin) -> LoginView:
    result: Final = await value.logins.cancel(login_id)
    if isinstance(result, AccountFailure):
        raise public_failure(result)
    return result


@router.put("/bindings", response_model=AccountsState)
async def set_bindings(body: BindingsRequest, value: Admin) -> AccountsState:
    result: Final = await value.service.bind(body.bindings, body.revision)
    if isinstance(result, AccountFailure):
        raise public_failure(result)
    return result


@router.post(
    "/{account_id}/check",
    response_model=AccountRecord,
    description="Check the stored authorization and model access for one profile.",
)
async def check_account(account_id: UUID, value: Admin, _body: NoOptions | None = None) -> AccountRecord:
    result: Final = await value.service.check(account_id)
    if isinstance(result, AccountFailure):
        raise public_failure(result)
    return result


@router.patch("/{account_id}", response_model=AccountRecord)
async def rename_account(account_id: UUID, body: LabelRequest, value: Admin) -> AccountRecord:
    result: Final = value.service.rename(account_id, body.label)
    if isinstance(result, AccountFailure):
        raise public_failure(result)
    return result


@router.post(
    "/{account_id}/disconnect",
    response_model=AccountsState,
    description="Disable an unassigned profile while allowing in-flight requests to finish.",
)
async def disconnect_account(account_id: UUID, value: Admin, _body: NoOptions | None = None) -> AccountsState:
    result: Final = value.service.disable(account_id)
    if isinstance(result, AccountFailure):
        raise public_failure(result)
    return result
