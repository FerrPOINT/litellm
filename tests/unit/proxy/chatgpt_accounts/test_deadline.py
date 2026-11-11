import asyncio
from typing import Final

import httpx
import pytest
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from litellm.proxy.chatgpt_accounts.deadline import AccountsRequestDeadline


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "elapsed", "expected_status"),
    [(None, 601, 200), ("6000", 601, 200), ("7200", 7000, 200), ("6000", 6001, 504), ("2", 3, 504)],
)
async def test_inference_uses_the_configured_whole_request_deadline(
    monkeypatch: pytest.MonkeyPatch, configured: str | None, elapsed: float, expected_status: int
) -> None:
    monkeypatch.setenv("LITELLM_CHATGPT_ACCOUNTS_DIR", "fixture-accounts")
    monkeypatch.delenv("LITELLM_CHATGPT_ACCOUNTS_REQUEST_TIMEOUT", raising=False)
    if configured is not None:
        monkeypatch.setenv("LITELLM_CHATGPT_ACCOUNTS_REQUEST_TIMEOUT", configured)
    clock = 100000.0
    loop: Final = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "time", lambda: clock)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal clock
        clock += elapsed
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await JSONResponse({"completed": True})(scope, receive, send)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=AccountsRequestDeadline(app)), base_url="http://fixture.test"
    ) as client:
        response: Final = await client.post("/v1/responses", json={"input": "fixture"})
    assert response.status_code == expected_status
    if expected_status == 504:
        assert response.json()["error"]["code"] == "chatgpt_request_deadline"
    else:
        assert response.json() == {"completed": True}


@pytest.mark.parametrize("configured", ["0", "-1", "nan", "inf"])
def test_invalid_request_deadline_is_rejected(monkeypatch: pytest.MonkeyPatch, configured: str) -> None:
    monkeypatch.setenv("LITELLM_CHATGPT_ACCOUNTS_REQUEST_TIMEOUT", configured)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse({"completed": True})(scope, receive, send)

    with pytest.raises(ValueError, match="positive"):
        AccountsRequestDeadline(app)
