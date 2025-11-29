"""Shared utilities for actions"""


class FileInfo:
    """Information about a file in an execution environment."""

    def __init__(self, name: str, path: str, type: str, size: int = 0):
        self.name = name
        self.path = path
        self.type = type
        self.size = size

    def __repr__(self):
        return f"FileInfo(name='{self.name}', path='{self.path}', type='{self.type}', size={self.size})"

    def to_dict(self) -> dict:
        return {'name': self.name, 'path': self.path, 'type': self.type, 'size': self.size}
