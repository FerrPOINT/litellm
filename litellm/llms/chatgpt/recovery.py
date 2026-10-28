import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter

from .authenticator import Authenticator, get_cached_authenticator, get_chatgpt_auth_file
from .common_utils import JsonObject

_JSON: Final = TypeAdapter(JsonObject)


def request_authenticator(
    provider: str,
    params: Mapping[str, object],
    headers: Mapping[object, object],
) -> Authenticator | None:
    if provider != "chatgpt":
        return None
    auth_file: Final = get_chatgpt_auth_file(params)
    auth: Final = get_cached_authenticator(auth_file) if auth_file else Authenticator()
    header: Final = headers.get("Authorization")
    token: Final = header.removeprefix("Bearer ") if isinstance(header, str) else ""
    if auth_file is None and not auth.is_saved_access_token(token):
        return None
    return auth


def is_invalidated(error: httpx.HTTPStatusError) -> bool:
    if error.response.status_code != 401:
        return False
    try:
        payload: Final = _JSON.validate_python(error.response.json())
        detail: Final = payload.get("error")
        return isinstance(detail, Mapping) and detail.get("code") == "token_invalidated"
    except (ValueError, TypeError):
        return False


def post_with_refresh(
    send: Callable[[Mapping[object, object]], httpx.Response],
    headers: Mapping[object, object],
    auth: Authenticator | None,
) -> httpx.Response:
    try:
        return send(headers)
    except httpx.HTTPStatusError as error:
        if auth is None or not is_invalidated(error):
            raise
        header: Final = headers.get("Authorization")
        rejected: Final = header.removeprefix("Bearer ") if isinstance(header, str) else ""
        token: Final = auth.refresh_access_token(rejected_access_token=rejected)
        return send({**headers, "Authorization": "Bearer " + token})


async def apost_with_refresh(
    send: Callable[[Mapping[object, object]], Awaitable[httpx.Response]],
    headers: Mapping[object, object],
    auth: Authenticator | None,
) -> httpx.Response:
    try:
        return await send(headers)
    except httpx.HTTPStatusError as error:
        if auth is None or not is_invalidated(error):
            raise
        header: Final = headers.get("Authorization")
        rejected: Final = header.removeprefix("Bearer ") if isinstance(header, str) else ""
        token: Final = await asyncio.to_thread(auth.refresh_access_token, rejected_access_token=rejected)
        return await send({**headers, "Authorization": "Bearer " + token})
