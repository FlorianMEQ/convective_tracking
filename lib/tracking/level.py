from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class DetectionContext:
    """Runtime context passed to each level detector."""

    dataset: Any
    time_index: int
    time_value: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrackerProtocol:
    """Minimal protocol for trackers used by TrackingLevel."""

    def update(self, time_index: int, time_value: str, objects: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def finalize(self) -> Dict[str, Any]:
        raise NotImplementedError


DetectorFn = Callable[[DetectionContext, Optional[Dict[str, Any]], "TrackingLevel"], List[Dict[str, Any]]]


@dataclass
class TrackingLevel:
    """A generic level in a hierarchical detection and tracking pipeline."""

    name: str
    detector: DetectorFn
    parent_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    tracker: Optional[TrackerProtocol] = None

    def detect(
        self,
        context: DetectionContext,
        parent_object: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        objects = self.detector(context, parent_object, self)
        if objects is None:
            return []
        return objects

    def update_tracks(self, time_index: int, time_value: str, objects: List[Dict[str, Any]]) -> None:
        if self.tracker is not None:
            self.tracker.update(time_index, time_value, objects)

    def finalize_tracks(self) -> Dict[str, Any]:
        if self.tracker is None:
            return {}
        return self.tracker.finalize()
