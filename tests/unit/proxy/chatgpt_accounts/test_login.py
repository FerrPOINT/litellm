import asyncio
from pathlib import Path
from typing import Final

import httpx
import jwt
import pytest
import respx

from litellm.llms.chatgpt.common_utils import CHATGPT_DEVICE_CODE_URL, CHATGPT_DEVICE_TOKEN_URL, CHATGPT_OAUTH_TOKEN_URL
from litellm.proxy.chatgpt_accounts.login import LoginOperations
from litellm.proxy.chatgpt_accounts.models import LoginView
from test_service import account


@pytest.mark.asyncio
@respx.mock
async def test_add_account_preserves_active_profile_and_publishes_private_credentials(service) -> None:
    first: Final = account(service)
    before: Final = service.store.auth_file(first).read_bytes()
    access: Final = jwt.encode(
        {
            "exp": 4102444800,
            "email": "same@example.test",
            "sub": "fixture-user-B",
            "https://api.openai.com/auth": {"chatgpt_account_id": "workspace-B"},
        },
        "fixture-secret-aaaaaaaaaaaaaaaaaaaaaaaa",
        algorithm="HS256",
    )
    respx.post(CHATGPT_DEVICE_CODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={"device_auth_id": "fixture-device", "user_code": "CODE-FIXTURE", "expires_in": 900, "interval": 5},
        )
    )
    respx.post(CHATGPT_DEVICE_TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"authorization_code": "fixture-code", "code_verifier": "fixture-verifier"}
        )
    )
    respx.post(CHATGPT_OAUTH_TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": access, "refresh_token": "fixture-refresh", "id_token": access}
        )
    )
    async with httpx.AsyncClient() as client:
        logins: Final = LoginOperations(service, client)
        started: Final = await logins.start("Personal")
        assert isinstance(started, LoginView)
        await asyncio.gather(*logins.tasks.values())
        completed: Final = logins.views[started.id]
        assert completed.status == "ready", completed.error_code
        record: Final = service.record(started.account_id)
        assert record is not None and record.workspace == "workspace-B"
        assert record.email == "same@example.test"
        assert service.store.auth_file(first).read_bytes() == before
        assert service.store.auth_file(started.account_id).stat().st_mode & 0o777 == 0o600
        assert not tuple((service.store.root / "pending").rglob("auth.json"))
        assert "access_token" not in completed.model_dump_json()
        await logins.close()


@pytest.mark.asyncio
@respx.mock
async def test_cancel_sign_in_does_not_publish_account(service) -> None:
    first: Final = account(service)
    before: Final = service.store.state_file.read_bytes()
    respx.post(CHATGPT_DEVICE_CODE_URL).mock(
        return_value=httpx.Response(
            200, json={"device_auth_id": "fixture-device", "user_code": "CODE-FIXTURE", "interval": 5}
        )
    )
    async with httpx.AsyncClient() as client:
        logins: Final = LoginOperations(service, client)
        started: Final = await logins.start("Cancelled")
        assert isinstance(started, LoginView)
        cancelled: Final = await logins.cancel(started.id)
        assert isinstance(cancelled, LoginView) and cancelled.status == "cancelled"
        assert cancelled.user_code == ""
        assert service.store.state_file.read_bytes() == before
        assert service.record(first) is not None
        assert service.record(started.account_id) is None
        await logins.close()


@pytest.mark.asyncio
@respx.mock
async def test_reconnect_rejects_another_user_in_the_same_workspace(service) -> None:
    first: Final = account(service)
    service.update_account(first, {"workspace": "shared-workspace", "subject": "user-A"})
    before: Final = service.store.auth_file(first).read_bytes()
    access: Final = jwt.encode(
        {
            "exp": 4102444800,
            "sub": "user-B",
            "email": "same@example.test",
            "https://api.openai.com/auth": {"chatgpt_account_id": "shared-workspace"},
        },
        "fixture-secret-aaaaaaaaaaaaaaaaaaaaaaaa",
        algorithm="HS256",
    )
    respx.post(CHATGPT_DEVICE_CODE_URL).mock(
        return_value=httpx.Response(200, json={"device_auth_id": "fixture-device", "user_code": "CODE-FIXTURE"})
    )
    respx.post(CHATGPT_DEVICE_TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"authorization_code": "fixture-code", "code_verifier": "fixture-verifier"}
        )
    )
    respx.post(CHATGPT_OAUTH_TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": access, "id_token": access, "refresh_token": "fixture-new"}
        )
    )
    async with httpx.AsyncClient() as client:
        logins: Final = LoginOperations(service, client)
        started: Final = await logins.start("Personal", first)
        assert isinstance(started, LoginView)
        await asyncio.gather(*logins.tasks.values())
        assert logins.views[started.id].error_code == "account_mismatch"
        assert service.record(first).subject == "user-A"
        assert service.store.auth_file(first).read_bytes() == before
        assert not tuple((service.store.root / "pending").rglob("auth.json"))
        await logins.close()
