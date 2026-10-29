from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class DetectionContext:
    dataset: Any
    time_index: int
    time_value: str
    timestamp_s: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


DetectorFn = Callable[[DetectionContext, Optional[Dict[str, Any]], "TrackingLevel"], List[Dict[str, Any]]]


@dataclass
class TrackingLevel:
    name: str
    detector: DetectorFn
    parent_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    tracker: Any = None

    def detect(
        self,
        context: DetectionContext,
        parent_object: Optional[Dict[str, Any]] = None,
    ):
        objects = self.detector(context, parent_object, self)
        if objects is None:
            return []
        return objects

    def track(self, time_index, time_value, timestamp_s, objects):
        """Feed the level's tracker with every object detected at this timestep."""
        if self.tracker is not None:
            self.tracker.update(time_index, time_value, timestamp_s, objects)

    def finalize_tracks(self):
        return {} if self.tracker is None else self.tracker.finalize()
