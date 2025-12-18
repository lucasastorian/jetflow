"""Cache backend implementations"""

import json
import lmdb
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class Cache(ABC):
    """Abstract base class for cache backends"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key, returns None if not found"""
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Store value by key"""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key, returns True if existed"""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached entries"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the cache (cleanup resources)"""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MemoryCache(Cache):
    """In-memory cache for testing"""

    def __init__(self):
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        self._store.clear()

    def close(self) -> None:
        pass  # Nothing to close

    def __len__(self) -> int:
        return len(self._store)


class LMDBCache(Cache):
    """LMDB-backed persistent cache

    Fast, ACID-compliant key-value store with memory-mapped access.
    Ideal for caching LLM responses.

    Args:
        path: Directory path for the LMDB database
        map_size: Maximum database size in bytes (default 1GB)
    """

    def __init__(self, path: str = ".jetflow/cache", map_size: int = 1024 * 1024 * 1024):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self._env = lmdb.open(
            str(self.path),
            map_size=map_size,
            max_dbs=0,
            sync=True,
            writemap=True,
        )

    def get(self, key: str) -> Optional[Any]:
        with self._env.begin() as txn:
            data = txn.get(key.encode())
            if data is None:
                return None
            return json.loads(data.decode())

    def set(self, key: str, value: Any) -> None:
        data = json.dumps(value, separators=(',', ':')).encode()
        with self._env.begin(write=True) as txn:
            txn.put(key.encode(), data)

    def delete(self, key: str) -> bool:
        with self._env.begin(write=True) as txn:
            return txn.delete(key.encode())

    def clear(self) -> None:
        with self._env.begin(write=True) as txn:
            txn.drop(self._env.open_db(), delete=False)

    def close(self) -> None:
        self._env.close()

    def __len__(self) -> int:
        with self._env.begin() as txn:
            return txn.stat()['entries']
