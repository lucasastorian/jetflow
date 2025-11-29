"""Tests for LocalPythonExec file API"""

import pytest
import os

from jetflow.actions.local_python_exec import LocalPythonExec, PythonExec, FileInfo


class TestFileAPI:
    """Tests for the file API methods"""

    @pytest.fixture
    def executor(self):
        """Create executor with session started"""
        exec_instance = LocalPythonExec()
        exec_instance.__start__()
        yield exec_instance
        exec_instance.__stop__()

    def test_write_and_read_file(self, executor):
        """Test writing and reading a text file"""
        executor.write_file('test.txt', 'Hello, World!')
        content = executor.read_file('test.txt')
        assert content == 'Hello, World!'

    def test_write_and_read_bytes(self, executor):
        """Test writing and reading binary content"""
        binary_data = b'\x00\x01\x02\x03\xff'
        executor.write_file('binary.bin', binary_data)
        content = executor.read_file('binary.bin', format='bytes')
        assert content == binary_data

    def test_list_files_empty(self, executor):
        """Test listing empty directory"""
        files = executor.list_files()
        assert isinstance(files, list)
        # May have some matplotlib config files, but should work

    def test_list_files_with_content(self, executor):
        """Test listing directory after creating files"""
        executor.write_file('file1.txt', 'content1')
        executor.write_file('file2.txt', 'content2')

        files = executor.list_files()
        names = [f.name for f in files]

        assert 'file1.txt' in names
        assert 'file2.txt' in names

    def test_make_dir(self, executor):
        """Test creating a directory"""
        executor.make_dir('subdir')
        executor.write_file('subdir/nested.txt', 'nested content')

        content = executor.read_file('subdir/nested.txt')
        assert content == 'nested content'

    def test_delete_file(self, executor):
        """Test deleting a file"""
        executor.write_file('to_delete.txt', 'delete me')
        assert executor.read_file('to_delete.txt') == 'delete me'

        executor.delete_file('to_delete.txt')

        with pytest.raises(FileNotFoundError):
            executor.read_file('to_delete.txt')

    def test_delete_directory(self, executor):
        """Test deleting a directory"""
        executor.make_dir('dir_to_delete')
        executor.write_file('dir_to_delete/file.txt', 'content')

        executor.delete_file('dir_to_delete')

        with pytest.raises(FileNotFoundError):
            executor.list_files('dir_to_delete')

    def test_file_info_attributes(self, executor):
        """Test FileInfo object attributes"""
        executor.write_file('info_test.txt', 'hello')

        files = executor.list_files()
        info_file = next((f for f in files if f.name == 'info_test.txt'), None)

        assert info_file is not None
        assert info_file.type == 'file'
        assert info_file.size == 5
        assert 'info_test.txt' in info_file.path

    def test_file_info_to_dict(self, executor):
        """Test FileInfo.to_dict()"""
        executor.write_file('dict_test.txt', 'test')

        files = executor.list_files()
        info_file = next((f for f in files if f.name == 'dict_test.txt'), None)

        d = info_file.to_dict()
        assert d['name'] == 'dict_test.txt'
        assert d['type'] == 'file'
        assert 'path' in d
        assert 'size' in d

    def test_nested_directory_listing(self, executor):
        """Test listing files in nested directory"""
        executor.make_dir('nested/deep/path')
        executor.write_file('nested/deep/path/file.txt', 'deep content')

        files = executor.list_files('nested/deep/path')
        assert len(files) == 1
        assert files[0].name == 'file.txt'


class TestFileAPISecurityBoundaries:
    """Tests for file API security boundaries"""

    @pytest.fixture
    def executor(self):
        """Create executor with session started"""
        exec_instance = LocalPythonExec()
        exec_instance.__start__()
        yield exec_instance
        exec_instance.__stop__()

    def test_read_outside_temp_dir_blocked(self, executor):
        """Test that reading outside temp directory is blocked"""
        with pytest.raises(PermissionError):
            executor.read_file('/etc/passwd')

    def test_write_outside_temp_dir_blocked(self, executor):
        """Test that writing outside temp directory is blocked"""
        with pytest.raises(PermissionError):
            executor.write_file('/tmp/evil.txt', 'malicious')

    def test_path_traversal_blocked(self, executor):
        """Test that path traversal attacks are blocked"""
        with pytest.raises(PermissionError):
            executor.read_file('../../../etc/passwd')

    def test_list_outside_temp_dir_blocked(self, executor):
        """Test that listing outside temp directory is blocked"""
        with pytest.raises(PermissionError):
            executor.list_files('/etc')

    def test_delete_outside_temp_dir_blocked(self, executor):
        """Test that deleting outside temp directory is blocked"""
        with pytest.raises(PermissionError):
            executor.delete_file('/tmp/some_file')

    def test_make_dir_outside_temp_dir_blocked(self, executor):
        """Test that creating dir outside temp directory is blocked"""
        with pytest.raises(PermissionError):
            executor.make_dir('/tmp/evil_dir')


class TestFileAPIQuotas:
    """Tests for file API quota enforcement"""

    @pytest.fixture
    def executor_small_quota(self):
        """Create executor with small quota for testing"""
        exec_instance = LocalPythonExec(max_total_size_mb=1, max_file_size_mb=0.5)
        exec_instance.__start__()
        yield exec_instance
        exec_instance.__stop__()

    def test_file_size_limit_enforced(self, executor_small_quota):
        """Test that individual file size limit is enforced"""
        # 0.5MB = 524288 bytes, try to write 600KB
        large_content = 'x' * (600 * 1024)

        with pytest.raises(PermissionError) as exc_info:
            executor_small_quota.write_file('large.txt', large_content)

        assert 'exceeds' in str(exc_info.value)

    def test_total_quota_enforced(self, executor_small_quota):
        """Test that total quota is enforced"""
        # Write files that together exceed 1MB
        chunk = 'x' * (300 * 1024)  # 300KB each

        executor_small_quota.write_file('file1.txt', chunk)
        executor_small_quota.write_file('file2.txt', chunk)
        executor_small_quota.write_file('file3.txt', chunk)  # 900KB total

        # This should exceed quota
        with pytest.raises(PermissionError) as exc_info:
            executor_small_quota.write_file('file4.txt', chunk)

        assert 'Quota exceeded' in str(exc_info.value)


class TestFileAPIWithoutSession:
    """Tests for file API when session is not started"""

    def test_read_without_session_raises(self):
        """Test that read_file raises RuntimeError without session"""
        from jetflow.actions.local_python_exec.sandbox import LocalSandbox
        sandbox = LocalSandbox()
        # Don't call start()

        with pytest.raises(RuntimeError) as exc_info:
            sandbox.read_file('test.txt')

        assert 'not started' in str(exc_info.value)

    def test_write_without_session_raises(self):
        """Test that write_file raises RuntimeError without session"""
        from jetflow.actions.local_python_exec.sandbox import LocalSandbox
        sandbox = LocalSandbox()

        with pytest.raises(RuntimeError) as exc_info:
            sandbox.write_file('test.txt', 'content')

        assert 'not started' in str(exc_info.value)

    def test_list_without_session_raises(self):
        """Test that list_files raises RuntimeError without session"""
        from jetflow.actions.local_python_exec.sandbox import LocalSandbox
        sandbox = LocalSandbox()

        with pytest.raises(RuntimeError) as exc_info:
            sandbox.list_files()

        assert 'not started' in str(exc_info.value)


class TestFileAPICodeInteraction:
    """Tests for interaction between file API and code execution"""

    @pytest.fixture
    def executor(self):
        """Create executor with session started"""
        exec_instance = LocalPythonExec()
        exec_instance.__start__()
        yield exec_instance
        exec_instance.__stop__()

    def test_code_can_read_api_written_file(self, executor):
        """Test that code execution can read files written via API"""
        executor.write_file('data.csv', 'a,b,c\n1,2,3\n4,5,6')

        result = executor(PythonExec(code="""
import pandas as pd
df = pd.read_csv('data.csv')
result = df.shape
"""))

        assert '(2, 3)' in result.content

    def test_api_can_read_code_written_file(self, executor):
        """Test that API can read files written by code execution"""
        executor(PythonExec(code="""
with open('output.txt', 'w') as f:
    f.write('written by code')
"""))

        content = executor.read_file('output.txt')
        assert content == 'written by code'

    def test_api_list_shows_code_created_files(self, executor):
        """Test that list_files shows files created by code execution"""
        executor(PythonExec(code="""
import os
os.makedirs('code_dir', exist_ok=True)
with open('code_dir/file.txt', 'w') as f:
    f.write('test')
"""))

        files = executor.list_files('code_dir')
        assert len(files) == 1
        assert files[0].name == 'file.txt'
