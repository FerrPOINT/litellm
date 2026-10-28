import asyncio
import time
from collections.abc import Mapping
from typing import Final
from uuid import UUID, uuid4

import httpx
from filelock import FileLock
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import (
    CHATGPT_AUTH_BASE,
    CHATGPT_CLIENT_ID,
    CHATGPT_DEVICE_CODE_URL,
    CHATGPT_DEVICE_TOKEN_URL,
    CHATGPT_DEVICE_VERIFY_URL,
    CHATGPT_OAUTH_TOKEN_URL,
    GetAccessTokenError,
    decode_jwt_claims,
)

from .models import AccountFailure, AccountRecord, LoginView
from .service import AccountService
from .store import write_private_file


class DeviceCode(BaseModel):
    model_config = ConfigDict(frozen=True)
    device_auth_id: str
    user_code: str = Field(validation_alias=AliasChoices("user_code", "usercode"))
    interval: int = Field(default=5, ge=1, le=60)
    expires_in: int = Field(default=900, ge=1, le=1800)


class AuthorizationCode(BaseModel):
    model_config = ConfigDict(frozen=True)
    authorization_code: str
    code_verifier: str


class OAuthTokens(BaseModel):
    model_config = ConfigDict(frozen=True)
    access_token: str
    refresh_token: str
    id_token: str


class LoginOperations:
    def __init__(self, service: AccountService, client: httpx.AsyncClient) -> None:
        self.service = service
        self.client = client
        self.views: Mapping[UUID, LoginView] = {}
        self.tasks: Mapping[UUID, asyncio.Task[None]] = {}

    async def start(self, label: str, account_id: UUID | None = None) -> LoginView | AccountFailure:
        if sum(view.status == "pending" for view in self.views.values()) >= 4:
            return AccountFailure(code="too_many_logins", message="Finish or cancel an existing sign-in.", status=429)
        if len(self.service.store.read().accounts) >= 32 and account_id is None:
            return AccountFailure(code="too_many_accounts", message="The account registry is full.", status=409)
        if account_id is not None and self.service.record(account_id) is None:
            return AccountFailure(code="account_not_found", message="Account not found.", status=404)
        try:
            response: Final = await self.client.post(CHATGPT_DEVICE_CODE_URL, json={"client_id": CHATGPT_CLIENT_ID})
            response.raise_for_status()
            device: Final = DeviceCode.model_validate(response.json())
        except (httpx.HTTPError, ValueError, OSError, GetAccessTokenError):
            return AccountFailure(
                code="login_start_failed", message="Could not start ChatGPT sign-in. Retry later.", status=502
            )
        login_id: Final = uuid4()
        view: Final = LoginView(
            id=login_id,
            account_id=account_id or uuid4(),
            verification_url=CHATGPT_DEVICE_VERIFY_URL,
            user_code=device.user_code,
            expires_at=time.time() + device.expires_in,
        )
        self.views = {**dict(tuple(self.views.items())[-31:]), login_id: view}
        self.tasks = {
            **{key: task for key, task in self.tasks.items() if not task.done()},
            login_id: asyncio.create_task(self._run(view, device, label)),
        }
        return view

    async def cancel(self, login_id: UUID) -> LoginView | AccountFailure:
        view: Final = self.views.get(login_id)
        if view is None:
            return AccountFailure(
                code="login_not_found", message="Sign-in expired or the application restarted.", status=404
            )
        if view.status != "pending":
            return view
        cancelled: Final = view.model_copy(update={"status": "cancelled", "user_code": ""})
        self.views = {**self.views, login_id: cancelled}
        task: Final = self.tasks.get(login_id)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return cancelled

    async def close(self) -> None:
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    async def _run(self, view: LoginView, device: DeviceCode, label: str) -> None:
        try:
            while time.time() < view.expires_at:
                if (
                    response := await self.client.post(
                        CHATGPT_DEVICE_TOKEN_URL,
                        json={"device_auth_id": device.device_auth_id, "user_code": device.user_code},
                    )
                ).status_code == 200:
                    if (
                        result := await self._complete(view, AuthorizationCode.model_validate(response.json()), label)
                    ) is None:
                        self.views = {
                            **self.views,
                            view.id: view.model_copy(update={"status": "ready", "user_code": ""}),
                        }
                        return
                    self.views = {
                        **self.views,
                        view.id: view.model_copy(
                            update={
                                "status": "failed" if result else "ready",
                                "user_code": "",
                                "error_code": result.code if result else None,
                            }
                        ),
                    }
                    return
                if response.status_code not in (403, 404, 429):
                    response.raise_for_status()
                await asyncio.sleep(max(5, device.interval))
            self.views = {**self.views, view.id: view.model_copy(update={"status": "expired", "user_code": ""})}
        except asyncio.CancelledError:
            return
        except (httpx.HTTPError, ValueError, OSError, GetAccessTokenError):
            self.views = {
                **self.views,
                view.id: view.model_copy(
                    update={
                        "status": "failed",
                        "user_code": "",
                        "error_code": "login_failed",
                    }
                ),
            }

        finally:
            candidate: Final = self.service.store.root / "pending" / str(view.id) / "auth.json"
            candidate.unlink(missing_ok=True)

    async def _complete(self, view: LoginView, code: AuthorizationCode, label: str) -> AccountFailure | None:
        response: Final = await self.client.post(
            CHATGPT_OAUTH_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CHATGPT_CLIENT_ID,
                "code": code.authorization_code,
                "code_verifier": code.code_verifier,
                "redirect_uri": CHATGPT_AUTH_BASE + "/deviceauth/callback",
            },
        )
        response.raise_for_status()
        tokens: Final = OAuthTokens.model_validate(response.json())
        candidate: Final = self.service.store.root / "pending" / str(view.id) / "auth.json"
        auth: Final = Authenticator(auth_file=str(candidate))
        auth.save_tokens(tokens.model_dump())
        probe: Final = await self.service.probe.check(candidate, ())
        if probe.error:
            return probe.error
        identity: Final = auth.get_account_id()
        if not identity:
            return AccountFailure(code="identity_missing", message="ChatGPT did not identify the account.", status=401)
        claims: Final = decode_jwt_claims(tokens.id_token)
        subject: Final = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            return AccountFailure(code="identity_missing", message="ChatGPT did not identify the user.", status=401)
        existing: Final = self.service.record(view.account_id)
        if existing is not None and (existing.workspace != identity or existing.subject != subject):
            return AccountFailure(
                code="account_mismatch", message="Sign in to the account selected for reconnection.", status=409
            )
        email: Final = claims.get("email")
        current_view: Final = self.views.get(view.id)
        if current_view is None or current_view.status != "pending":
            return AccountFailure(code="login_cancelled", message="Sign-in was cancelled.", status=409)
        destination: Final = self.service.store.auth_file(view.account_id)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with FileLock(str(destination) + ".lock"):
            write_private_file(destination, candidate.read_text())
        result: Final = self.service.save_account(
            AccountRecord(
                id=view.account_id,
                label=existing.label if existing else label,
                email=email if isinstance(email, str) else None,
                workspace=identity,
                subject=subject,
                client_id=CHATGPT_CLIENT_ID,
                status="ready",
                models=probe.models,
                last_checked_at=time.time(),
            )
        )
        candidate.unlink(missing_ok=True)
        return result if isinstance(result, AccountFailure) else None
