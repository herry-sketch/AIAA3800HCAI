from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .types import FrameFeatures


@dataclass
class TemporalBuffer:
    _items: list[FrameFeatures] = field(default_factory=list)

    def append(self, features: FrameFeatures) -> None:
        self._items.append(features)

    def get_window(self, seconds: float) -> list[FrameFeatures]:
        if not self._items:
            return []
        end_time = self._items[-1].timestamp
        start_time = end_time - max(seconds, 0.0)
        return [item for item in self._items if item.timestamp >= start_time]

    def prune(self, max_seconds: float = 35.0) -> None:
        if not self._items:
            return
        cutoff = self._items[-1].timestamp - max_seconds
        while self._items and self._items[0].timestamp < cutoff:
            self._items.pop(0)

    def latest(self) -> FrameFeatures | None:
        return self._items[-1] if self._items else None

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterable[FrameFeatures]:
        return iter(self._items)

