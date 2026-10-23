from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from ..src.tracking.engine import HierarchicalTrackingEngine
from ..src.tracking.level import DetectionContext, TrackingLevel
from ..src.tracking.trackers import DEFAULT_STEP_SECONDS


DetectFn = Callable[..., List[Dict[str, Any]]]
TrackFn = Callable[..., Dict[str, Any]]


@dataclass
class LegacyLevelSpec:
    """Describe how one hierarchical level maps to legacy detection outputs."""

    name: str
    result_key: str
    parent_name: Optional[str] = None
    parent_match_field: Optional[str] = None
    parent_id_field: str = "object_id"


def _get_legacy_step_result(context, level):
    cache = context.metadata.setdefault("_legacy_detection_cache", {})
    if context.time_index in cache:
        return cache[context.time_index]

    detect_fn: DetectFn = level.params["legacy_detect_fn"]
    result = detect_fn(context.dataset, **level.params.get("legacy_detection_kwargs", {}))
    step = cache[context.time_index] = result[0] if result else {"time": context.time_value}
    return step


def _build_level_detector(
    result_key,
    parent_match_field,
    parent_id_field,
):
    def detector(
        context,
        parent,
        level,
    ):
        step = _get_legacy_step_result(context, level)
        items = list(step.get(result_key, []))
        if parent is None or parent_match_field is None:
            return items

        parent_id = parent.get(parent_id_field)
        if parent_id is None:
            return []
        return [item for item in items if item.get(parent_match_field) == parent_id]

    return detector


def create_legacy_levels(
    *,
    detect_fn,
    level_specs,
    detection_kwargs = None,
):
    """Create TrackingLevel objects bound to one legacy detection function."""
    return [
        TrackingLevel(
            name=spec.name,
            parent_name=spec.parent_name,
            detector=_build_level_detector(spec.result_key, spec.parent_match_field, spec.parent_id_field),
            params={
                "legacy_detect_fn": detect_fn,
                "legacy_detection_kwargs": detection_kwargs or {},
            },
        )
        for spec in level_specs
    ]



def run_legacy_pipeline(
    *,
    datasets,
    levels,
    result_key_by_level,
    track_fns_by_level,
    track_kwargs_by_level = None,
):
    """Run engine + convert results + execute legacy tracking functions."""
    time_info = extract_time_values_and_timestamps(datasets)
    engine = HierarchicalTrackingEngine(levels=levels)
    engine_results = engine.run(
        datasets=time_info["datasets"],
        time_values=time_info["time_values"],
    )

    legacy_results = to_legacy_results_by_time(
        engine_results,
        result_key_by_level=result_key_by_level,
        timestamps_s=time_info["timestamps_s"],
    )

    kwargs_map = track_kwargs_by_level or {}
    tracks_by_level = {
        level_name: track_fn(legacy_results, **kwargs_map.get(level_name, {}))
        for level_name, track_fn in track_fns_by_level.items()
    }
    return {"results_by_time": legacy_results, "tracks_by_level": tracks_by_level}
