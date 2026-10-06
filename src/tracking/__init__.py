"""Hierarchical tracking framework.

- Generic engine and tracker: :class:`HierarchicalTrackingEngine`,
  :class:`TrackingLevel`, :class:`AdvectedIoUTracker`.
- Generic detector, configured by a rule per level: :func:`detect_objects`.
"""
from .detection import detect_objects
from .engine import HierarchicalTrackingEngine
from .level import DetectionContext, TrackingLevel
from .trackers import AdvectedIoUTracker

__all__ = [
    "AdvectedIoUTracker",
    "DetectionContext",
    "HierarchicalTrackingEngine",
    "TrackingLevel",
    "detect_objects",
]
