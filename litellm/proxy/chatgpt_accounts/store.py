import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Final
from uuid import UUID

from filelock import FileLock

from .models import AccountFailure, AccountsState


def write_private_file(path: Path, content: str) -> None:
    if path.is_symlink():
        raise OSError("Refusing symlinked account state")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, candidate = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(candidate, path)
        directory: Final = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(candidate):
            os.unlink(candidate)


class AccountStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.state_file = self.root / "accounts.json"
        self.lock = FileLock(str(self.root / "accounts.lock"))

    def auth_file(self, account_id: UUID) -> Path:
        path: Final = self.root / "credentials" / str(account_id) / "auth.json"
        if path.is_symlink() or path.parent.is_symlink() or path.parent.parent.is_symlink():
            raise OSError("Refusing symlinked account credentials")
        return path

    def read(self) -> AccountsState:
        if not self.state_file.exists():
            return AccountsState()
        if self.state_file.is_symlink():
            raise OSError("Refusing symlinked account registry")
        return AccountsState.model_validate_json(self.state_file.read_text(encoding="utf-8"))

    def update(
        self,
        transform: Callable[[AccountsState], AccountsState | AccountFailure],
        expected_revision: int | None = None,
        advance_revision: bool = True,
    ) -> AccountsState | AccountFailure:
        with self.lock:
            current: Final = self.read()
            if expected_revision is not None and current.revision != expected_revision:
                return AccountFailure(
                    code="state_changed", message="Account settings changed. Reload and retry.", status=409
                )
            updated: Final = transform(current)
            if isinstance(updated, AccountFailure):
                return updated
            next_state: Final = updated.model_copy(update={"revision": current.revision + int(advance_revision)})
            write_private_file(self.state_file, next_state.model_dump_json(indent=2))
            return next_state
