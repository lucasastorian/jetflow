"""E2B Sandbox executor with session persistence"""

from typing import Optional, Any
from e2b_code_interpreter import Sandbox


class E2BSandboxExecutor:
    """Manages E2B sandbox lifecycle. Patterns: ephemeral (create->kill) or session-based (query paused->resume or create with auto-pause)."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        persistent: bool = False,
        timeout: int = 300,
        api_key: Optional[str] = None,
        _sandbox_id: Optional[str] = None,
    ):
        self._sandbox_id_override = _sandbox_id
        self.session_id = session_id
        self.user_id = user_id
        self.persistent = persistent
        self.timeout = timeout
        self.api_key = api_key
        self._sandbox: Optional[Sandbox] = None
        self._last_sandbox_id: Optional[str] = None

        if persistent and not session_id and not _sandbox_id:
            raise ValueError("persistent=True requires session_id")

    def __start__(self) -> None:
        if self._sandbox is not None:
            return

        kwargs = {}
        if self.api_key:
            kwargs['api_key'] = self.api_key

        if self._sandbox_id_override:
            self._sandbox = Sandbox.connect(sandbox_id=self._sandbox_id_override, timeout=self.timeout, **kwargs)
            self._last_sandbox_id = self._sandbox.sandbox_id
            return

        if self.persistent and self.session_id:
            from e2b_code_interpreter import SandboxQuery, SandboxState

            metadata = {'session_id': self.session_id}
            if self.user_id:
                metadata['user_id'] = self.user_id

            query = SandboxQuery(state=[SandboxState.PAUSED], metadata=metadata)
            paginator = Sandbox.list(query=query, **kwargs)
            sandboxes = paginator.next_items()

            if len(sandboxes) > 1:
                raise ValueError(f"Multiple paused sandboxes found for session_id={self.session_id}. Found {len(sandboxes)} sandboxes.")

            if sandboxes:
                self._sandbox = Sandbox.connect(sandbox_id=sandboxes[0].sandbox_id, timeout=self.timeout, **kwargs)
                self._last_sandbox_id = self._sandbox.sandbox_id
                return

            create_metadata = {'session_id': self.session_id}
            if self.user_id:
                create_metadata['user_id'] = self.user_id

            self._sandbox = Sandbox.beta_create(auto_pause=True, timeout=self.timeout, metadata=create_metadata, **kwargs)
            self._last_sandbox_id = self._sandbox.sandbox_id
            return

        self._sandbox = Sandbox.create(timeout=self.timeout, **kwargs)
        self._last_sandbox_id = self._sandbox.sandbox_id

    def run_code(self, code: str) -> Any:
        if not self._sandbox:
            raise RuntimeError("Sandbox not started. Call __start__() first.")
        execution = self._sandbox.run_code(code)
        return execution

    def __stop__(self) -> None:
        if not self._sandbox:
            return

        try:
            if self.persistent:
                self._sandbox.beta_pause()
            else:
                self._sandbox.kill()
        finally:
            self._sandbox = None

    @property
    def sandbox(self) -> Optional[Sandbox]:
        return self._sandbox
