from __future__ import annotations

from typing import Any, Dict, List, Optional

from lib.tracking import DetectionContext, TrackingLevel
from lib.fct_core_tracking import find_convective_cores_3d


def _get_legacy_step_result(context: DetectionContext, level: TrackingLevel) -> Dict[str, Any]:
    """Compute and cache one legacy detection step for the current time index."""
    cache = context.metadata.setdefault("_legacy_detection_cache", {})
    if context.time_index in cache:
        return cache[context.time_index]

    kwargs = dict(level.params.get("legacy_detection_kwargs", {}))
    result = find_convective_cores_3d(context.dataset, **kwargs)
    step = result[0] if result else {
        "time": context.time_value,
        "mcs": [],
        "conv_objects": [],
        "cores": [],
    }
    cache[context.time_index] = step
    return step


def detect_mcs(context: DetectionContext, parent: Optional[Dict[str, Any]], level: TrackingLevel) -> List[Dict[str, Any]]:
    step = _get_legacy_step_result(context, level)
    return list(step.get("mcs", []))


def detect_convective_objects(
    context: DetectionContext,
    parent: Optional[Dict[str, Any]],
    level: TrackingLevel,
) -> List[Dict[str, Any]]:
    step = _get_legacy_step_result(context, level)
    objs = list(step.get("conv_objects", []))
    if parent is None:
        return objs

    parent_id = parent.get("mcs_id") or parent.get("object_id")
    if parent_id is None:
        return []
    return [obj for obj in objs if obj.get("mcs_id") == parent_id]


def detect_cores(
    context: DetectionContext,
    parent: Optional[Dict[str, Any]],
    level: TrackingLevel,
) -> List[Dict[str, Any]]:
    step = _get_legacy_step_result(context, level)
    cores = list(step.get("cores", []))
    if parent is None:
        return cores

    parent_id = parent.get("conv_obj_id") or parent.get("object_id")
    if parent_id is None:
        return []
    return [core for core in cores if core.get("conv_obj_id") == parent_id]


def build_convective_levels(legacy_detection_kwargs: Optional[Dict[str, Any]] = None) -> List[TrackingLevel]:
    """Build a hierarchy whose detection is delegated to legacy convective code."""
    detection_kwargs = legacy_detection_kwargs or {}
    return [
        TrackingLevel(
            name="mcs",
            detector=detect_mcs,
            params={"legacy_detection_kwargs": detection_kwargs},
        ),
        TrackingLevel(
            name="convective_object",
            detector=detect_convective_objects,
            parent_name="mcs",
            params={"legacy_detection_kwargs": detection_kwargs},
        ),
        TrackingLevel(
            name="core",
            detector=detect_cores,
            parent_name="convective_object",
            params={"legacy_detection_kwargs": detection_kwargs},
        ),
    ]
