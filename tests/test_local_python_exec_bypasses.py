"""Test that common sandbox bypasses are blocked"""

import os
import pytest
from jetflow.actions.local_python_exec import LocalPythonExec, PythonExec


@pytest.fixture
def executor():
    """Create local Python executor"""
    return LocalPythonExec(extract_charts=False)


class TestBypassProtection:
    """Test that common bypass techniques are blocked"""

    def test_builtins_open_bypass_blocked(self, executor):
        """Test that import builtins; builtins.open() is blocked"""
        repo_path = os.path.abspath('bypass_builtins.txt')

        result = executor(PythonExec(code=f"""
import builtins
try:
    with builtins.open(r'{repo_path}', 'w') as f:
        f.write('bypassed')
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError:
    result = 'blocked'
"""))

        assert 'blocked' in result.content
        assert not os.path.exists('bypass_builtins.txt')

    def test_io_open_bypass_blocked(self, executor):
        """Test that import io; io.open() is blocked"""
        repo_path = os.path.abspath('bypass_io.txt')

        result = executor(PythonExec(code=f"""
import io
try:
    with io.open(r'{repo_path}', 'w') as f:
        f.write('bypassed')
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError:
    result = 'blocked'
"""))

        assert 'blocked' in result.content
        assert not os.path.exists('bypass_io.txt')

    def test_os_open_bypass_blocked(self, executor):
        """Test that os.open() is blocked"""
        repo_path = os.path.abspath('bypass_os.txt')

        result = executor(PythonExec(code=f"""
import os
try:
    fd = os.open(r'{repo_path}', os.O_WRONLY | os.O_CREAT)
    os.close(fd)
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError:
    result = 'blocked'
"""))

        assert 'blocked' in result.content
        assert not os.path.exists('bypass_os.txt')

    def test_pathlib_bypass_blocked(self, executor):
        """Test that pathlib.Path().write_text() is blocked"""
        repo_path = os.path.abspath('bypass_pathlib.txt')

        result = executor(PythonExec(code=f"""
from pathlib import Path
try:
    Path(r'{repo_path}').write_text('bypassed')
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError:
    result = 'blocked'
"""))

        assert 'blocked' in result.content
        assert not os.path.exists('bypass_pathlib.txt')

    def test_os_remove_blocked(self, executor):
        """Test that os.remove() outside temp dir is blocked"""
        # Create a test file in repo
        test_file = 'test_to_remove.txt'
        with open(test_file, 'w') as f:
            f.write('test')

        try:
            abs_path = os.path.abspath(test_file)

            result = executor(PythonExec(code=f"""
import os
try:
    os.remove(r'{abs_path}')
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError:
    result = 'blocked'
"""))

            assert 'blocked' in result.content
            # Verify file still exists
            assert os.path.exists(test_file)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_pathlib_unlink_blocked(self, executor):
        """Test that Path.unlink() outside temp dir is blocked"""
        test_file = 'test_to_unlink.txt'
        with open(test_file, 'w') as f:
            f.write('test')

        try:
            abs_path = os.path.abspath(test_file)

            result = executor(PythonExec(code=f"""
from pathlib import Path
try:
    Path(r'{abs_path}').unlink()
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError:
    result = 'blocked'
"""))

            assert 'blocked' in result.content
            assert os.path.exists(test_file)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_os_mkdir_outside_temp_blocked(self, executor):
        """Test that os.mkdir() outside temp dir is blocked"""
        dir_path = os.path.abspath('malicious_dir')

        result = executor(PythonExec(code=f"""
import os
try:
    os.mkdir(r'{dir_path}')
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError:
    result = 'blocked'
"""))

        assert 'blocked' in result.content
        assert not os.path.exists('malicious_dir')

    def test_os_rename_outside_temp_blocked(self, executor):
        """Test that os.rename() outside temp dir is blocked"""
        src = os.path.abspath('src_file.txt')
        dst = os.path.abspath('dst_file.txt')

        result = executor(PythonExec(code=f"""
import os
try:
    os.rename(r'{src}', r'{dst}')
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError:
    result = 'blocked'
"""))

        assert 'blocked' in result.content


class TestAllowedOperations:
    """Test that operations within temp dir are allowed"""

    def test_write_in_temp_dir_allowed(self, executor):
        """Test that writes within temp dir succeed"""
        result = executor(PythonExec(code="""
with open('allowed.txt', 'w') as f:
    f.write('success')
result = 'written'
"""))

        assert 'written' in result.content

    def test_os_operations_in_temp_allowed(self, executor):
        """Test that os operations within temp dir succeed"""
        result = executor(PythonExec(code="""
import os

os.mkdir('testdir')
with open('testdir/file.txt', 'w') as f:
    f.write('test')
os.rename('testdir/file.txt', 'testdir/renamed.txt')
os.remove('testdir/renamed.txt')
os.rmdir('testdir')

result = 'success'
"""))

        assert 'success' in result.content

    def test_pathlib_operations_in_temp_allowed(self, executor):
        """Test that pathlib operations within temp dir succeed"""
        result = executor(PythonExec(code="""
from pathlib import Path

p = Path('test.txt')
p.write_text('hello')
content = p.read_text()
p.unlink()

result = content
"""))

        assert 'hello' in result.content


class TestBlockedModules:
    """Test that dangerous modules are blocked"""

    def test_subprocess_blocked(self, executor):
        """Test that subprocess import is blocked"""
        result = executor(PythonExec(code="""
try:
    import subprocess
    result = 'SHOULD_NOT_REACH_HERE'
except ImportError as e:
    result = f'blocked: {e}'
"""))
        assert 'blocked' in result.content

    def test_os_system_blocked(self, executor):
        """Test that os.system is blocked"""
        result = executor(PythonExec(code="""
import os
try:
    os.system('echo test')
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError as e:
    result = f'blocked: {e}'
"""))
        assert 'blocked' in result.content

    def test_os_popen_blocked(self, executor):
        """Test that os.popen is blocked"""
        result = executor(PythonExec(code="""
import os
try:
    os.popen('echo test')
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError as e:
    result = f'blocked: {e}'
"""))
        assert 'blocked' in result.content

    def test_socket_blocked(self, executor):
        """Test that socket import is blocked"""
        result = executor(PythonExec(code="""
try:
    import socket
    result = 'SHOULD_NOT_REACH_HERE'
except ImportError as e:
    result = f'blocked: {e}'
"""))
        assert 'blocked' in result.content

    def test_ctypes_blocked(self, executor):
        """Test that ctypes import is blocked"""
        result = executor(PythonExec(code="""
try:
    import ctypes
    result = 'SHOULD_NOT_REACH_HERE'
except ImportError as e:
    result = f'blocked: {e}'
"""))
        assert 'blocked' in result.content

    def test_pickle_blocked(self, executor):
        """Test that pickle import is blocked"""
        result = executor(PythonExec(code="""
try:
    import pickle
    result = 'SHOULD_NOT_REACH_HERE'
except ImportError as e:
    result = f'blocked: {e}'
"""))
        assert 'blocked' in result.content

    def test_tarfile_blocked(self, executor):
        """Test that tarfile import is blocked"""
        result = executor(PythonExec(code="""
try:
    import tarfile
    result = 'SHOULD_NOT_REACH_HERE'
except ImportError as e:
    result = f'blocked: {e}'
"""))
        assert 'blocked' in result.content

    def test_multiprocessing_blocked(self, executor):
        """Test that multiprocessing import is blocked"""
        result = executor(PythonExec(code="""
try:
    import multiprocessing
    result = 'SHOULD_NOT_REACH_HERE'
except ImportError as e:
    result = f'blocked: {e}'
"""))
        assert 'blocked' in result.content


class TestQuotaEnforcement:
    """Test that file size quotas are enforced during writes"""

    def test_large_write_via_open_blocked(self):
        """Test that writing a large file via open() is blocked"""
        executor = LocalPythonExec(extract_charts=False, max_file_size_mb=1)

        result = executor(PythonExec(code="""
try:
    with open('large.txt', 'w') as f:
        f.write('x' * (2 * 1024 * 1024))  # 2MB
    result = 'SHOULD_NOT_REACH_HERE'
except PermissionError as e:
    result = f'blocked: {e}'
"""))
        assert 'blocked' in result.content
