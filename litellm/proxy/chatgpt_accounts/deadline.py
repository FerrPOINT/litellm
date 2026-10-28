import asyncio
import os
from typing import Final

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class AccountsRequestDeadline:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.seconds = min(600.0, float(os.environ.get("LITELLM_CHATGPT_ACCOUNTS_REQUEST_TIMEOUT", "600")))
        if self.seconds <= 0:
            raise ValueError("ChatGPT accounts request timeout must be positive")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not os.environ.get("LITELLM_CHATGPT_ACCOUNTS_DIR") or scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope["path"] not in ("/v1/responses", "/responses", "/v1/chat/completions", "/chat/completions"):
            await self.app(scope, receive, send)
            return
        started: Final = ResponseStarted(send)
        try:
            async with asyncio.timeout(self.seconds):
                await self.app(scope, receive, started)
        except TimeoutError:
            if started.value:
                raise
            response: Final = JSONResponse(
                {
                    "error": {
                        "code": "chatgpt_request_deadline",
                        "message": "The ChatGPT request deadline was exceeded.",
                    }
                },
                status_code=504,
            )
            await response(scope, receive, send)


class ResponseStarted:
    def __init__(self, send: Send) -> None:
        self.send = send
        self.value = False

    async def __call__(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            self.value = True
        await self.send(message)
