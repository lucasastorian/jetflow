"""E2B Code Interpreter - Cloud-based Python execution with session persistence

Requires: pip install jetflow[e2b]

Example with S3 storage for market data:
    from jetflow.actions.e2b_python_exec import E2BPythonExec, S3Storage

    exec = E2BPythonExec(
        template="my-s3-template",  # Custom template with s3fs installed
        storage=S3Storage(
            bucket="market-data",
            access_key_id="AKIA...",
            secret_access_key="...",
        )
    )
    # Agent can read files from /home/user/bucket/
"""

try:
    from jetflow.actions.e2b_python_exec.action import E2BPythonExec, PythonExec
    from jetflow.actions.e2b_python_exec.sandbox import E2BSandbox
    from jetflow.actions.e2b_python_exec.storage import (
        BaseStorage,
        S3Storage,
        R2Storage,
        GCSStorage,
    )
    from jetflow.actions.utils import FileInfo

    __all__ = [
        "E2BPythonExec",
        "PythonExec",
        "E2BSandbox",
        "FileInfo",
        "BaseStorage",
        "S3Storage",
        "R2Storage",
        "GCSStorage",
    ]
except ImportError as e:
    raise ImportError(
        "E2B code interpreter requires e2b SDK. Install with: pip install jetflow[e2b]"
    ) from e
