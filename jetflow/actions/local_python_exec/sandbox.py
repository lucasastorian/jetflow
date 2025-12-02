"""Local sandbox for Python code execution with security restrictions."""

import ast
import builtins
import io
import math
import collections
import os
import shutil
import signal
import sys
import tempfile
from typing import Optional, Dict, Any, List, Union

try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY_PANDAS = True
except ImportError:
    np = None
    pd = None
    HAS_NUMPY_PANDAS = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    sns = None
    HAS_MATPLOTLIB = False

from jetflow.actions.utils import FileInfo

BLOCKED_MODULES = frozenset({
    'subprocess', 'pty', 'multiprocessing', 'concurrent',
    'socket', 'socketserver', 'http', 'urllib', 'ftplib', 'smtplib', 'poplib', 'imaplib', 'nntplib', 'telnetlib',
    'asyncio', 'selectors', 'select',
    'tarfile', 'zipfile', 'gzip', 'bz2', 'lzma', 'zipimport',
    'sqlite3', 'dbm', 'shelve',
    'ctypes', 'cffi', '_ctypes',
    'importlib', 'pkgutil', 'modulefinder',
    'code', 'codeop', 'compile', 'compileall', 'py_compile',
    'gc', 'inspect', 'dis', 'traceback', 'linecache',
    'pickle', 'marshal', 'copyreg',
    'resource', 'sysconfig', 'platform',
    'webbrowser', 'cgi', 'cgitb',
    'mmap', 'fcntl', 'termios', 'tty', 'rlcompleter',
    'pwd', 'grp', 'spwd', 'crypt',
    'ssl', 'hashlib', 'hmac', 'secrets',
})


class TimeoutError(Exception):
    """Raised when code execution exceeds timeout limit."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution exceeded timeout limit")


def _check_write_path(path: str, temp_dir: str):
    """Validate path is within temp directory."""
    abs_path = os.path.realpath(os.path.abspath(path))
    abs_temp = os.path.realpath(os.path.abspath(temp_dir))
    if not abs_path.startswith(abs_temp):
        raise PermissionError(f"Write access denied outside temp directory: {path}")


class LocalSandbox:
    """Sandboxed environment for local Python code execution."""

    def __init__(self, max_total_size_mb: int = 50, max_file_size_mb: int = 10):
        self.max_total_size = max_total_size_mb * 1024 * 1024
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.temp_dir: Optional[str] = None
        self.namespace: Dict[str, Any] = {}
        self._original_mpl_config: Optional[str] = None
        self._original_limits = None
        self._original_cwd: Optional[str] = None
        self._patched_modules: Dict[str, Any] = {}

    def start(self) -> None:
        """Initialize sandbox environment."""
        if self.temp_dir:
            return

        self.temp_dir = tempfile.mkdtemp(prefix='jetflow_exec_')
        self._original_cwd = os.getcwd()
        self._setup_namespace()
        self._setup_matplotlib()

    def stop(self) -> None:
        """Cleanup sandbox environment."""
        if not self.temp_dir:
            return

        self._restore_matplotlib()
        if self.temp_dir:
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass
            self.temp_dir = None

    def _setup_namespace(self) -> None:
        """Initialize execution namespace with safe builtins."""
        safe_builtins = self._make_safe_builtins()
        self.namespace = {'__builtins__': safe_builtins}
        self.namespace['__builtins__']['__import__'] = self._make_safe_import(safe_builtins)

    def _setup_matplotlib(self) -> None:
        """Configure matplotlib for sandbox."""
        if not HAS_MATPLOTLIB:
            return
        self._original_mpl_config = os.environ.get('MPLCONFIGDIR')
        os.environ['MPLCONFIGDIR'] = self.temp_dir

        # Inject savefig tracking to capture chart IDs
        self._original_savefig = plt.Figure.savefig

        def _tracked_savefig(fig_self, fname, *args, **kwargs):
            # Extract filename without path or extension
            filename = os.path.basename(str(fname))
            if '.' in filename:
                filename = filename.rsplit('.', 1)[0]
            # Store on figure for later extraction
            fig_self._jetflow_chart_id = filename
            return self._original_savefig(fig_self, fname, *args, **kwargs)

        plt.Figure.savefig = _tracked_savefig

    def _restore_matplotlib(self) -> None:
        """Restore matplotlib configuration."""
        if not HAS_MATPLOTLIB:
            return

        # Restore original savefig
        if hasattr(self, '_original_savefig'):
            plt.Figure.savefig = self._original_savefig

        if self._original_mpl_config is not None:
            os.environ['MPLCONFIGDIR'] = self._original_mpl_config
        elif 'MPLCONFIGDIR' in os.environ:
            del os.environ['MPLCONFIGDIR']

    def run_code(self, code: str, timeout: int = 5) -> Dict[str, Any]:
        """Execute code in sandbox with restrictions."""
        ephemeral = self.temp_dir is None
        if ephemeral:
            self.start()

        from jetflow.actions.local_python_exec.utils import preprocess_code, diff_namespace, ASTGuard

        code = preprocess_code(code)
        result = {'output': '', 'error': None, 'result': None, 'charts': None}

        # Setup restricted modules
        restricted_open = self._make_restricted_open()
        self.namespace['__builtins__']['open'] = restricted_open
        original_builtins_open = builtins.open
        original_io_open = io.open
        builtins.open = restricted_open
        io.open = restricted_open

        restricted_os = self._make_restricted_os()
        restricted_pathlib = self._make_restricted_pathlib()
        restricted_shutil = self._make_restricted_shutil()

        self.namespace['__builtins__']['os'] = restricted_os
        self.namespace['__builtins__']['pathlib'] = restricted_pathlib
        self.namespace['__builtins__']['shutil'] = restricted_shutil

        original_sys_modules = {}
        for modname in ('os', 'pathlib', 'shutil'):
            if modname in sys.modules:
                original_sys_modules[modname] = sys.modules[modname]
        sys.modules['os'] = restricted_os
        sys.modules['pathlib'] = restricted_pathlib
        sys.modules['shutil'] = restricted_shutil

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        old_handler = self._setup_timeout(timeout)
        self._original_limits = self._setup_resource_limits()

        pre_exec_figs = self._get_figure_nums() if HAS_MATPLOTLIB else set()

        try:
            os.chdir(self.temp_dir)
            sys.stdout, sys.stderr = stdout_capture, stderr_capture

            parsed = ast.parse(code, mode='exec')
            guard = ASTGuard(safe_builtins={k: v for k, v in self.namespace['__builtins__'].items() if k != '__import__'})
            guard.visit(parsed)

            before_ns = dict(self.namespace)
            exec_result = self._execute_ast(parsed, before_ns)
            result['result'] = exec_result

        except TimeoutError:
            result['error'] = f"Timeout: Execution exceeded {timeout} seconds"
        except MemoryError:
            result['error'] = "Memory Error: Execution exceeded 512MB limit"
        except PermissionError as e:
            result['error'] = f"Permission Error: {str(e)}"
        except SyntaxError as e:
            result['error'] = f"Syntax Error: {e.msg} at line {e.lineno}"
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            if len(tb) > 500:
                tb = "..." + tb[-500:]
            result['error'] = f"{str(e)}\n{tb}"
        finally:
            self._restore_resource_limits()
            self._cleanup_timeout(old_handler)
            try:
                os.chdir(self._original_cwd)
            except:
                pass
            sys.stdout, sys.stderr = old_stdout, old_stderr
            builtins.open = original_builtins_open
            io.open = original_io_open

            for modname, mod in original_sys_modules.items():
                sys.modules[modname] = mod
            for modname in ('os', 'pathlib', 'shutil'):
                if modname not in original_sys_modules and modname in sys.modules:
                    del sys.modules[modname]

        result['output'] = stdout_capture.getvalue()
        result['stderr'] = stderr_capture.getvalue()

        if HAS_MATPLOTLIB and not result['error']:
            post_exec_figs = self._get_figure_nums()
            new_figs = post_exec_figs - pre_exec_figs
            if new_figs:
                result['new_figures'] = new_figs

        if ephemeral:
            self.stop()

        return result

    def _execute_ast(self, parsed: ast.Module, before_ns: Dict) -> Any:
        """Execute parsed AST and return result."""
        from jetflow.actions.local_python_exec.utils import diff_namespace

        result = None
        if parsed.body:
            last_node = parsed.body[-1]

            if isinstance(last_node, ast.Expr):
                if len(parsed.body) > 1:
                    statements = ast.Module(body=parsed.body[:-1], type_ignores=[])
                    exec(compile(statements, '<string>', 'exec'), self.namespace)
                expr = ast.Expression(body=last_node.value)
                result = eval(compile(expr, '<string>', 'eval'), self.namespace)
            else:
                exec(compile(parsed, '<string>', 'exec'), self.namespace)
                for candidate in ("result", "out", "data", "summary"):
                    if candidate in self.namespace and candidate not in before_ns:
                        result = self.namespace[candidate]
                        break
                if result is None:
                    diff = diff_namespace(before_ns, self.namespace)
                    if diff["added"] or diff["modified"]:
                        result = diff

        return result

    def _get_figure_nums(self) -> set:
        """Get current matplotlib figure numbers."""
        if not HAS_MATPLOTLIB:
            return set()
        return set(str(i) for i in plt.get_fignums())

    def read_file(self, path: str, format: str = 'text') -> Union[str, bytes]:
        """Read a file from the sandbox."""
        if not self.temp_dir:
            raise RuntimeError("Sandbox not started. Call start() first.")

        full_path = os.path.join(self.temp_dir, path) if not os.path.isabs(path) else path
        real_path = os.path.realpath(full_path)
        real_temp = os.path.realpath(self.temp_dir)

        if not real_path.startswith(real_temp):
            raise PermissionError(f"Read access denied outside temp directory: {path}")
        if not os.path.exists(real_path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(real_path, 'rb' if format == 'bytes' else 'r') as f:
            return f.read()

    def write_file(self, path: str, content: Union[str, bytes]) -> None:
        """Write a file to the sandbox."""
        if not self.temp_dir:
            raise RuntimeError("Sandbox not started. Call start() first.")

        full_path = os.path.join(self.temp_dir, path) if not os.path.isabs(path) else path
        _check_write_path(full_path, self.temp_dir)

        content_size = len(content) if isinstance(content, bytes) else len(content.encode('utf-8'))
        if content_size > self.max_file_size:
            raise PermissionError(f"File size exceeds {self.max_file_size / 1024 / 1024:.0f}MB limit")
        self._check_quota(content_size)

        parent_dir = os.path.dirname(full_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        with open(full_path, 'wb' if isinstance(content, bytes) else 'w') as f:
            f.write(content)

    def list_files(self, path: str = '.') -> List[FileInfo]:
        """List files in the sandbox directory."""
        if not self.temp_dir:
            raise RuntimeError("Sandbox not started. Call start() first.")

        full_path = os.path.join(self.temp_dir, path) if not os.path.isabs(path) else path
        real_path = os.path.realpath(full_path)
        real_temp = os.path.realpath(self.temp_dir)

        if not real_path.startswith(real_temp):
            raise PermissionError(f"Access denied outside temp directory: {path}")
        if not os.path.exists(real_path):
            raise FileNotFoundError(f"Directory not found: {path}")
        if not os.path.isdir(real_path):
            raise NotADirectoryError(f"Not a directory: {path}")

        result = []
        for name in os.listdir(real_path):
            entry_path = os.path.join(real_path, name)
            is_dir = os.path.isdir(entry_path)
            size = 0 if is_dir else os.path.getsize(entry_path)
            result.append(FileInfo(name=name, path=os.path.relpath(entry_path, self.temp_dir), type='dir' if is_dir else 'file', size=size))
        return result

    def make_dir(self, path: str) -> None:
        """Create a directory in the sandbox."""
        if not self.temp_dir:
            raise RuntimeError("Sandbox not started. Call start() first.")

        full_path = os.path.join(self.temp_dir, path) if not os.path.isabs(path) else path
        _check_write_path(full_path, self.temp_dir)
        os.makedirs(full_path, exist_ok=True)

    def delete_file(self, path: str) -> None:
        """Delete a file or directory from the sandbox."""
        if not self.temp_dir:
            raise RuntimeError("Sandbox not started. Call start() first.")

        full_path = os.path.join(self.temp_dir, path) if not os.path.isabs(path) else path
        _check_write_path(full_path, self.temp_dir)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {path}")
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)

    def _check_quota(self, additional_size: int = 0) -> None:
        """Check if operation would exceed quota."""
        total = 0
        for dirpath, _, filenames in os.walk(self.temp_dir):
            for filename in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, filename))
                except:
                    pass
        if total + additional_size > self.max_total_size:
            raise PermissionError(f"Quota exceeded: would exceed {self.max_total_size / 1024 / 1024:.0f}MB limit")

    def _make_restricted_open(self):
        """Create restricted open() function."""
        _real_open = open
        temp_dir = self.temp_dir
        max_file_size = self.max_file_size
        check_quota = self._check_quota

        class QuotaEnforcingFile:
            """File wrapper that enforces size limits on writes."""
            def __init__(self, f, path):
                self._f = f
                self._path = path
                self._written = 0

            def write(self, data):
                size = len(data) if isinstance(data, (bytes, bytearray)) else len(data.encode('utf-8') if isinstance(data, str) else data)
                if self._written + size > max_file_size:
                    raise PermissionError(f"Write exceeds {max_file_size / 1024 / 1024:.0f}MB file size limit")
                check_quota(size)
                self._written += size
                return self._f.write(data)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self._f.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self._f, name)

        def restricted_open(file, mode='r', *args, **kwargs):
            if 'r' in mode and 'w' not in mode and 'a' not in mode and 'x' not in mode:
                return _real_open(file, mode, *args, **kwargs)
            if 'w' in mode or 'a' in mode or 'x' in mode:
                _check_write_path(str(file), temp_dir)
                check_quota(0)
                f = _real_open(file, mode, *args, **kwargs)
                return QuotaEnforcingFile(f, file)
            return _real_open(file, mode, *args, **kwargs)
        return restricted_open

    def _make_restricted_os(self):
        """Create restricted os module."""
        import os as os_real
        temp_dir = self.temp_dir

        class RestrictedOS:
            def __getattr__(self, name):
                if name in ('system', 'popen', 'spawn', 'spawnl', 'spawnle', 'spawnlp', 'spawnlpe',
                           'spawnv', 'spawnve', 'spawnvp', 'spawnvpe', 'execl', 'execle', 'execlp',
                           'execlpe', 'execv', 'execve', 'execvp', 'execvpe', 'fork', 'forkpty',
                           'kill', 'killpg', 'plock', 'startfile'):
                    raise PermissionError(f"os.{name} is blocked for security reasons")
                attr = getattr(os_real, name)
                if name == 'open':
                    _real = attr
                    def restricted(path, flags, *args, **kwargs):
                        if flags & (os_real.O_WRONLY | os_real.O_RDWR | os_real.O_CREAT | os_real.O_TRUNC | os_real.O_APPEND):
                            _check_write_path(path, temp_dir)
                        return _real(path, flags, *args, **kwargs)
                    return restricted
                elif name in ('remove', 'unlink', 'rmdir', 'mkdir', 'makedirs', 'rename', 'replace', 'symlink', 'link'):
                    _real = attr
                    def restricted(path, *args, **kwargs):
                        _check_write_path(path, temp_dir)
                        if args and isinstance(args[0], str):
                            _check_write_path(args[0], temp_dir)
                        return _real(path, *args, **kwargs)
                    return restricted
                return attr
        return RestrictedOS()

    def _make_restricted_pathlib(self):
        """Create restricted pathlib module."""
        import pathlib as pathlib_real
        temp_dir = self.temp_dir
        _RealPath = pathlib_real.Path

        class RestrictedPath(_RealPath):
            def _check(self):
                _check_write_path(str(self), temp_dir)

            def open(self, mode='r', *args, **kwargs):
                if 'w' in mode or 'a' in mode or 'x' in mode:
                    self._check()
                return super().open(mode, *args, **kwargs)

            def write_text(self, *args, **kwargs):
                self._check()
                return super().write_text(*args, **kwargs)

            def write_bytes(self, *args, **kwargs):
                self._check()
                return super().write_bytes(*args, **kwargs)

            def touch(self, *args, **kwargs):
                self._check()
                return super().touch(*args, **kwargs)

            def mkdir(self, *args, **kwargs):
                self._check()
                return super().mkdir(*args, **kwargs)

            def rmdir(self, *args, **kwargs):
                self._check()
                return super().rmdir(*args, **kwargs)

            def unlink(self, *args, **kwargs):
                self._check()
                return super().unlink(*args, **kwargs)

            def rename(self, target, *args, **kwargs):
                self._check()
                _check_write_path(str(target), temp_dir)
                return super().rename(target, *args, **kwargs)

            def replace(self, target, *args, **kwargs):
                self._check()
                _check_write_path(str(target), temp_dir)
                return super().replace(target, *args, **kwargs)

        class RestrictedPathlib:
            Path = RestrictedPath
            PosixPath = RestrictedPath
            WindowsPath = RestrictedPath

            def __getattr__(self, name):
                return getattr(pathlib_real, name)

        return RestrictedPathlib()

    def _make_restricted_shutil(self):
        """Create restricted shutil module."""
        import shutil as shutil_real
        temp_dir = self.temp_dir

        class RestrictedShutil:
            def __getattr__(self, name):
                attr = getattr(shutil_real, name)
                if name in ('copy', 'copy2', 'copyfile', 'copymode', 'copystat', 'copytree'):
                    _real = attr
                    def restricted(src, dst, *args, **kwargs):
                        _check_write_path(dst, temp_dir)
                        return _real(src, dst, *args, **kwargs)
                    return restricted
                elif name == 'move':
                    _real = attr
                    def restricted(src, dst, *args, **kwargs):
                        _check_write_path(src, temp_dir)
                        _check_write_path(dst, temp_dir)
                        return _real(src, dst, *args, **kwargs)
                    return restricted
                elif name == 'rmtree':
                    _real = attr
                    def restricted(path, *args, **kwargs):
                        _check_write_path(path, temp_dir)
                        return _real(path, *args, **kwargs)
                    return restricted
                return attr
        return RestrictedShutil()

    def _make_safe_builtins(self) -> Dict[str, Any]:
        """Create safe builtins dictionary."""
        builtins_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'len': len, 'pow': pow,
            'int': int, 'float': float, 'str': str, 'bool': bool, 'list': list, 'dict': dict,
            'tuple': tuple, 'set': set, 'frozenset': frozenset, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted, 'reversed': reversed,
            'any': any, 'all': all, 'print': print, 'type': type, 'isinstance': isinstance,
            'issubclass': issubclass, 'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
            'delattr': delattr, 'hash': hash, 'id': id, 'iter': iter, 'next': next, 'callable': callable,
            'chr': chr, 'ord': ord, 'bin': bin, 'hex': hex, 'oct': oct, 'ascii': ascii, 'repr': repr,
            'format': format, 'bytes': bytes, 'bytearray': bytearray, 'memoryview': memoryview,
            'complex': complex, 'divmod': divmod, 'slice': slice, 'property': property,
            'staticmethod': staticmethod, 'classmethod': classmethod, 'super': super, 'object': object,
            'Exception': Exception, 'BaseException': BaseException, 'ValueError': ValueError,
            'TypeError': TypeError, 'KeyError': KeyError, 'IndexError': IndexError,
            'AttributeError': AttributeError, 'ZeroDivisionError': ZeroDivisionError,
            'RuntimeError': RuntimeError, 'StopIteration': StopIteration, 'PermissionError': PermissionError,
            'FileNotFoundError': FileNotFoundError, 'OSError': OSError, 'IOError': IOError, 'ImportError': ImportError,
            'ModuleNotFoundError': ModuleNotFoundError, 'NotImplementedError': NotImplementedError,
            'math': math, 'collections': collections,
        }

        for modname in ['time', 'datetime', 'json', 're', 'random', 'statistics', 'itertools', 'functools', 'decimal', 'fractions', 'string', 'textwrap', 'copy', 'operator']:
            try:
                builtins_dict[modname] = __import__(modname)
            except ImportError:
                pass

        if HAS_NUMPY_PANDAS:
            builtins_dict.update({'np': np, 'numpy': np, 'pd': pd, 'pandas': pd})
        if HAS_MATPLOTLIB:
            builtins_dict.update({'plt': plt, 'matplotlib': matplotlib, 'sns': sns, 'seaborn': sns})

        return builtins_dict

    def _make_safe_import(self, allowed_builtins: Dict) -> callable:
        """Create import function that blocks dangerous modules."""
        builtin_import = __builtins__['__import__'] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _import(name, globals=None, locals=None, fromlist=(), level=0):
            base = (name or "").split(".")[0]
            if base in BLOCKED_MODULES:
                raise ImportError(f"Import of '{base}' is blocked for security reasons")
            if base in allowed_builtins:
                return allowed_builtins[base]
            return builtin_import(name, globals, locals, fromlist, level)
        return _import

    def _setup_timeout(self, timeout: int):
        """Setup timeout signal handler."""
        try:
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)
                return old_handler
        except (ValueError, OSError):
            pass
        return None

    def _cleanup_timeout(self, old_handler):
        """Cleanup timeout signal handler."""
        try:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
        except (ValueError, OSError):
            pass

    def _setup_resource_limits(self):
        """Set resource limits if available."""
        try:
            import resource
            original = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
            return original
        except (ImportError, ValueError, OSError):
            return None

    def _restore_resource_limits(self):
        """Restore resource limits."""
        if self._original_limits is None:
            return
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, self._original_limits)
        except (ImportError, ValueError, OSError):
            pass
