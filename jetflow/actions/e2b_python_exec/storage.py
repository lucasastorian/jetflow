"""Cloud storage mounting for E2B sandboxes.

Enables mounting S3, GCS, or R2 buckets to sandboxes for persistent data access.
Requires a custom E2B template with the appropriate FUSE tools installed:
- S3/R2: s3fs-fuse
- GCS: gcsfuse
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from e2b_code_interpreter import Sandbox


@dataclass
class BaseStorage:
    """Base class for cloud storage configuration."""
    bucket: str
    mount_path: str = "/home/user/bucket"

    def mount(self, sandbox: "Sandbox") -> None:
        """Mount the storage bucket to the sandbox."""
        raise NotImplementedError


@dataclass
class S3Storage(BaseStorage):
    """Amazon S3 storage configuration.

    Requires template with s3fs installed:
        template = Template().from_image("e2bdev/code-interpreter:latest").apt_install(["s3fs"])

    Usage:
        exec = E2BPythonExec(
            template="my-s3-template",
            storage=S3Storage(
                bucket="my-bucket",
                access_key_id="AKIA...",
                secret_access_key="...",
            )
        )
    """
    access_key_id: str = None
    secret_access_key: str = None
    region: Optional[str] = None

    def mount(self, sandbox: "Sandbox") -> None:
        sandbox.files.make_dir(self.mount_path)
        sandbox.files.write("/root/.passwd-s3fs", f"{self.access_key_id}:{self.secret_access_key}")
        sandbox.commands.run("sudo chmod 600 /root/.passwd-s3fs")

        flags = "-o allow_other -o use_path_request_style"
        if self.region:
            flags += f" -o endpoint={self.region}"

        sandbox.commands.run(f"sudo s3fs {self.bucket} {self.mount_path} {flags}")


@dataclass
class R2Storage(BaseStorage):
    """Cloudflare R2 storage configuration.

    Requires template with s3fs installed (same as S3).

    Usage:
        exec = E2BPythonExec(
            template="my-s3-template",
            storage=R2Storage(
                bucket="my-bucket",
                account_id="abc123",
                access_key_id="...",
                secret_access_key="...",
            )
        )
    """
    account_id: str = None
    access_key_id: str = None
    secret_access_key: str = None

    def mount(self, sandbox: "Sandbox") -> None:
        sandbox.files.make_dir(self.mount_path)
        sandbox.files.write("/root/.passwd-s3fs", f"{self.access_key_id}:{self.secret_access_key}")
        sandbox.commands.run("sudo chmod 600 /root/.passwd-s3fs")

        endpoint = f"https://{self.account_id}.r2.cloudflarestorage.com"
        sandbox.commands.run(
            f"sudo s3fs {self.bucket} {self.mount_path} -o url={endpoint} -o allow_other -o use_path_request_style"
        )


@dataclass
class GCSStorage(BaseStorage):
    """Google Cloud Storage configuration.

    Requires template with gcsfuse installed:
        template = (
            Template()
            .from_image("e2bdev/code-interpreter:latest")
            .apt_install(["gnupg", "lsb-release"])
            .run_cmd("lsb_release -c -s > /tmp/lsb_release")
            .run_cmd('GCSFUSE_REPO=$(cat /tmp/lsb_release) && echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt gcsfuse-$GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list')
            .run_cmd("curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc")
            .apt_install(["gcsfuse"])
        )

    Usage:
        exec = E2BPythonExec(
            template="my-gcs-template",
            storage=GCSStorage(
                bucket="my-bucket",
                service_account_key='{"type": "service_account", ...}',
            )
        )
    """
    service_account_key: str = None  # JSON string

    def mount(self, sandbox: "Sandbox") -> None:
        sandbox.files.make_dir(self.mount_path)
        sandbox.files.write("/home/user/gcs-key.json", self.service_account_key)

        sandbox.commands.run(
            f"sudo gcsfuse -o allow_other --file-mode=777 --dir-mode=777 "
            f"--key-file /home/user/gcs-key.json {self.bucket} {self.mount_path}"
        )
