from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.chatgpt_accounts.service import AccountService
from litellm.proxy.chatgpt_accounts.store import AccountStore
from litellm.router import Router
from test_service import ReadyProbe


@pytest.fixture
def service(tmp_path: Path) -> AccountService:
    router: Final = Router(
        model_list=[
            {"model_name": "main-dev", "litellm_params": {"model": "chatgpt/gpt-6.1-terra"}},
            {"model_name": "app-test", "litellm_params": {"model": "chatgpt/gpt-6-luna"}},
        ],
        num_retries=0,
    )
    return AccountService(AccountStore(tmp_path), router, ReadyProbe())
