"""Timer context manager utility"""

import datetime
from contextlib import contextmanager, asynccontextmanager
from typing import Optional


class Timer:
    """Timer that tracks start and end times"""

    def __init__(self):
        self.start_time: Optional[datetime.datetime] = None
        self.end_time: Optional[datetime.datetime] = None

    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else datetime.datetime.now()
        return (end - self.start_time).total_seconds()

    @staticmethod
    @contextmanager
    def measure():
        """Context manager for measuring execution time"""
        timer = Timer()
        timer.start_time = datetime.datetime.now()
        try:
            yield timer
        finally:
            timer.end_time = datetime.datetime.now()

    @staticmethod
    @asynccontextmanager
    async def measure_async():
        """Async context manager for measuring execution time"""
        timer = Timer()
        timer.start_time = datetime.datetime.now()
        try:
            yield timer
        finally:
            timer.end_time = datetime.datetime.now()
