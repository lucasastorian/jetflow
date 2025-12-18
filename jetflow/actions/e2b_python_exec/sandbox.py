"""E2B Sandbox with session persistence and file operations."""

from __future__ import annotations

from typing import Optional, Any, List, Union
from e2b_code_interpreter import Sandbox

from jetflow.actions.utils import FileInfo
from jetflow.actions.e2b_python_exec.storage import BaseStorage


class E2BSandbox:
    """Manages E2B sandbox lifecycle with file operations."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        persistent: bool = False,
        timeout: int = 300,
        api_key: Optional[str] = None,
        _sandbox_id: Optional[str] = None,
        template: Optional[str] = None,
        storage: Optional["BaseStorage"] = None,
    ):
        self._sandbox_id_override = _sandbox_id
        self.session_id = session_id
        self.user_id = user_id
        self.persistent = persistent
        self.timeout = timeout
        self.api_key = api_key
        self.template = template
        self.storage = storage
        self._sandbox: Optional[Sandbox] = None

        if persistent and not session_id and not _sandbox_id:
            raise ValueError("persistent=True requires session_id")

    def start(self) -> None:
        """Initialize sandbox."""
        if self._sandbox is not None:
            return

        kwargs = {'api_key': self.api_key} if self.api_key else {}
        if self.template:
            kwargs['template'] = self.template

        if self._sandbox_id_override:
            self._sandbox = Sandbox.connect(sandbox_id=self._sandbox_id_override, timeout=self.timeout, **kwargs)
            self._mount_storage()
            return

        if self.persistent and self.session_id:
            self._sandbox = self._resume_or_create_persistent(**kwargs)
            self._mount_storage()
            return

        self._sandbox = Sandbox.create(timeout=self.timeout, **kwargs)
        self._mount_storage()

    def _mount_storage(self) -> None:
        """Mount cloud storage if configured."""
        if self.storage and self._sandbox:
            self.storage.mount(self._sandbox)

    def _resume_or_create_persistent(self, **kwargs) -> Sandbox:
        """Resume paused sandbox or create new with auto-pause."""
        from e2b_code_interpreter import SandboxQuery, SandboxState

        metadata = {'session_id': self.session_id}
        if self.user_id:
            metadata['user_id'] = self.user_id

        query = SandboxQuery(state=[SandboxState.PAUSED], metadata=metadata)
        sandboxes = Sandbox.list(query=query, **kwargs).next_items()

        if len(sandboxes) > 1:
            raise ValueError(f"Multiple paused sandboxes for session_id={self.session_id}")

        if sandboxes:
            return Sandbox.connect(sandbox_id=sandboxes[0].sandbox_id, timeout=self.timeout, **kwargs)

        create_metadata = {'session_id': self.session_id}
        if self.user_id:
            create_metadata['user_id'] = self.user_id
        return Sandbox.beta_create(auto_pause=True, timeout=self.timeout, metadata=create_metadata, **kwargs)

    def stop(self) -> None:
        """Cleanup sandbox."""
        if not self._sandbox:
            return
        try:
            self._sandbox.beta_pause() if self.persistent else self._sandbox.kill()
        finally:
            self._sandbox = None

    def run_code(self, code: str) -> Any:
        """Execute code in sandbox."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")
        return self._sandbox.run_code(code)

    def read_file(self, path: str, format: str = 'text') -> Union[str, bytes]:
        """Read file from sandbox."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")
        content = self._sandbox.files.read(path)
        return content.encode('utf-8') if format == 'bytes' and not isinstance(content, bytes) else content

    def write_file(self, path: str, content: Union[str, bytes]) -> None:
        """Write file to sandbox."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")
        self._sandbox.files.write(path, content)

    def list_files(self, path: str = '/home/user') -> List[FileInfo]:
        """List files in sandbox directory."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")
        return [
            FileInfo(
                name=e.name,
                path=getattr(e, 'path', f"{path}/{e.name}"),
                type=getattr(e, 'type', 'dir' if getattr(e, 'is_dir', False) else 'file'),
                size=getattr(e, 'size', 0)
            )
            for e in self._sandbox.files.list(path)
        ]

    def make_dir(self, path: str) -> None:
        """Create directory in sandbox."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")
        self._sandbox.files.make_dir(path)

    def delete_file(self, path: str) -> None:
        """Delete file or directory from sandbox."""
        if not self._sandbox:
            raise RuntimeError("Sandbox not started")
        self.run_code(f"import os,shutil;p='{path}';shutil.rmtree(p) if os.path.isdir(p) else os.remove(p) if os.path.exists(p) else None")

    @property
    def is_started(self) -> bool:
        """Check if sandbox is started."""
        return self._sandbox is not None
