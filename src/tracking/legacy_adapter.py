from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from .engine import HierarchicalTrackingEngine
from .level import DetectionContext, TrackingLevel
from .trackers import DEFAULT_STEP_SECONDS


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


def _get_legacy_step_result(context: DetectionContext, level: TrackingLevel) -> Dict[str, Any]:
    cache = context.metadata.setdefault("_legacy_detection_cache", {})
    if context.time_index in cache:
        return cache[context.time_index]

    detect_fn: DetectFn = level.params["legacy_detect_fn"]
    result = detect_fn(context.dataset, **level.params.get("legacy_detection_kwargs", {}))
    step = cache[context.time_index] = result[0] if result else {"time": context.time_value}
    return step


def _build_level_detector(
    result_key: str,
    parent_match_field: Optional[str],
    parent_id_field: str,
) -> Callable[[DetectionContext, Optional[Dict[str, Any]], TrackingLevel], List[Dict[str, Any]]]:
    def detector(
        context: DetectionContext,
        parent: Optional[Dict[str, Any]],
        level: TrackingLevel,
    ) -> List[Dict[str, Any]]:
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
    detect_fn: DetectFn,
    level_specs: List[LegacyLevelSpec],
    detection_kwargs: Optional[Dict[str, Any]] = None,
) -> List[TrackingLevel]:
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


def extract_time_values_and_timestamps(
    datasets: Iterable[Any],
    fallback_step_seconds: float = DEFAULT_STEP_SECONDS,
) -> Dict[str, List[Any]]:
    """Extract time labels and timestamps from datasets."""
    dataset_list = list(datasets)
    time_values: List[str] = []
    timestamps_s: List[float] = []

    for idx, ds in enumerate(dataset_list):
        if hasattr(ds, "coords") and "time" in ds.coords:
            raw_time = ds["time"].values
            time_values.append(str(raw_time))
            timestamps_s.append(float(pd.Timestamp(raw_time).timestamp()))
        else:
            time_values.append(str(idx))
            timestamps_s.append(float(idx * fallback_step_seconds))

    return {"datasets": dataset_list, "time_values": time_values, "timestamps_s": timestamps_s}


def to_legacy_results_by_time(
    engine_results: List[Dict[str, Any]],
    *,
    result_key_by_level: Dict[str, str],
    timestamps_s: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Convert generic engine output to legacy-like results_by_time structure."""
    legacy: List[Dict[str, Any]] = []

    for step in engine_results:
        idx = int(step.get("time_index", 0))
        has_timestamp = timestamps_s is not None and idx < len(timestamps_s)
        entry: Dict[str, Any] = {
            "time": step.get("time", str(idx)),
            "timestamp_s": float(timestamps_s[idx]) if has_timestamp else idx * DEFAULT_STEP_SECONDS,
        }

        for level_name, result_key in result_key_by_level.items():
            objs = step.get("levels", {}).get(level_name, [])
            entry[result_key] = objs
            entry[f"n_{result_key}"] = len(objs)

        legacy.append(entry)

    return legacy


def run_legacy_pipeline(
    *,
    datasets: Iterable[Any],
    levels: List[TrackingLevel],
    result_key_by_level: Dict[str, str],
    track_fns_by_level: Dict[str, TrackFn],
    track_kwargs_by_level: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
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
