from pathlib import Path
from typing import Final

import httpx
import pytest
from fastapi import FastAPI

from litellm.proxy.chatgpt_accounts.endpoints import Controller, router
from litellm.proxy.chatgpt_accounts.login import LoginOperations


@pytest.mark.asyncio
async def test_inference_key_cannot_manage_accounts_and_session_rejects_csrf(service) -> None:
    app: Final = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient() as upstream:
        app.state.chatgpt_accounts = Controller(service, LoginOperations(service, upstream), "admin-" + "a" * 40)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://pilot") as client:
            denied: Final = await client.get("/chatgpt/accounts", headers={"Authorization": "Bearer inference-key"})
            assert denied.status_code == 401
            wrong: Final = await client.post("/chatgpt/accounts/session", json={"key": "inference-key"})
            assert wrong.status_code == 401
            logged_in: Final = await client.post("/chatgpt/accounts/session", json={"key": "admin-" + "a" * 40})
            assert logged_in.status_code == 200
            cookie: Final = logged_in.headers["set-cookie"].lower()
            assert "httponly" in cookie and "samesite=strict" in cookie
            listed: Final = await client.get("/chatgpt/accounts")
            assert listed.status_code == 200
            assert [route["name"] for route in listed.json()["routes"]] == ["main-dev", "app-test"]
            csrf: Final = await client.put(
                "/chatgpt/accounts/bindings", headers={"origin": "http://evil"}, json={"revision": 0, "bindings": {}}
            )
            assert csrf.status_code == 403
            signed_out: Final = await client.delete("/chatgpt/accounts/session")
            assert signed_out.status_code == 204
            expired: Final = await client.get("/chatgpt/accounts")
            assert expired.status_code == 401
