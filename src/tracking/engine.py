from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .level import DetectionContext, TrackingLevel


class HierarchicalTrackingEngine:
    """Runs an arbitrary number of hierarchical detection levels."""

    def __init__(self, levels):
        self.levels = levels
        self._validate_levels()
        self.results_by_time = []

    def _validate_levels(self):
        names = {level.name for level in self.levels}
        seen = set()
        for level in self.levels:
            if level.name in seen:
                raise ValueError(f"Duplicate level name: {level.name}")
            parent = level.parent_name
            if parent is not None and parent not in names:
                raise ValueError(f"Level '{level.name}' references unknown parent '{parent}'")
            if parent is not None and parent not in seen:
                raise ValueError(f"Parent '{parent}' must appear before child '{level.name}'")
            seen.add(level.name)

    def run_time_step(
        self,
        dataset,
        time_index,
        time_value = None,
        timestamp_s = None,
        metadata = None,
    ):
        time_label = str(time_index if time_value is None else time_value)
        # Copy so per-timestep caches never leak into the caller's (shared) dict.
        context = DetectionContext(
            dataset=dataset,
            time_index=time_index,
            time_value=time_label,
            timestamp_s=timestamp_s,
            metadata=dict(metadata or {}),
        )

        level_results: Dict[str, List[Dict[str, Any]]] = {}
        for level in self.levels:
            parents = [None] if level.parent_name is None else level_results.get(level.parent_name, [])

            objects: List[Dict[str, Any]] = []
            for parent in parents:
                for item in level.detect(context=context, parent_object=parent):
                    obj = dict(item)
                    obj.setdefault("level", level.name)
                    obj.setdefault("object_id", f"t{time_label}_{level.name}_{len(objects) + 1:04d}")
                    if parent is not None:
                        obj.setdefault("parent_object_id", parent.get("object_id"))
                    objects.append(obj)

            level_results[level.name] = objects
            level.track(time_index, time_label, timestamp_s, objects)

        step = {"time_index": time_index, "time": time_label, "levels": level_results}
        self.results_by_time.append(step)
        return step

    def run(
        self,
        datasets,
        time_values = None,
        timestamps_s = None,
        metadata = None,
    ):
        values = list(time_values) if time_values is not None else []
        ts_values = list(timestamps_s) if timestamps_s is not None else []

        for idx, dataset in enumerate(datasets):
            self.run_time_step(
                dataset=dataset,
                time_index=idx,
                time_value=str(values[idx] if idx < len(values) else idx),
                timestamp_s=ts_values[idx] if idx < len(ts_values) else None,
                metadata=metadata,
            )
        return self.results_by_time

    def finalize_tracks(self):
        return {level.name: level.finalize_tracks() for level in self.levels}

    def run_and_track(
        self,
        datasets,
        time_values = None,
        timestamps_s = None,
        metadata = None,
    ):
        results = self.run(datasets, time_values, timestamps_s, metadata)
        return {"results_by_time": results, "tracks_by_level": self.finalize_tracks()}
