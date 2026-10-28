from collections.abc import Mapping
from typing import Final, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AccountRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    label: str = Field(min_length=1, max_length=120)
    email: str | None = None
    workspace: str | None = None
    subject: str | None = None
    client_id: str
    enabled: bool = True
    status: Literal["ready", "login_required", "rate_limited", "error", "disabled"] = "login_required"
    models: tuple[str, ...] = ()
    last_checked_at: float | None = None
    error_code: str | None = None
    retry_at: float | None = None


class AccountBindings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default_account_id: UUID | None = None
    routes: Mapping[str, UUID | None] = Field(default_factory=dict)


class AccountsState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    revision: int = 0
    accounts: tuple[AccountRecord, ...] = ()
    bindings: AccountBindings = Field(default_factory=AccountBindings)


class AccountFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    message: str
    status: int = 400


class LoginView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    account_id: UUID
    verification_url: str
    user_code: str
    expires_at: float
    status: Literal["pending", "ready", "failed", "cancelled", "expired"] = "pending"
    error_code: str | None = None


class ProbeResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    models: tuple[str, ...] = ()
    error: AccountFailure | None = None


ACCOUNT_METADATA_KEY: Final = "_managed_chatgpt_account"
ACCOUNT_ID_FIELD: Final = "managed_chatgpt_account_id"
