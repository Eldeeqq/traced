"""This module contains the Queue class.

The Queue class is used to keep track of the last n items in a list.
This is used in the forgetting versions of models to keep track of the last
n samples.
"""

from collections import defaultdict
from typing import Any


class Queue:
    """Queue with a max size."""

    def __init__(self, max_size: int | None = None):
        self._queue = list()
        self.max_size = max_size
        self.counts_queue = defaultdict(lambda: 0.0)

    def add(self, item: Any) -> None:
        """Add an item to the queue."""
        if self.max_size and len(self._queue) == self.max_size:
            front = self._queue.pop(0)
            self.counts_queue[front] -= 1
            if self.counts_queue[front] == 0:
                del self.counts_queue[front]
        self._queue.append(item)
        self.counts_queue[item] += 1

    @property
    def data(self) -> dict[str, float]:
        """Get the data in the queue."""
        return self.counts_queue
