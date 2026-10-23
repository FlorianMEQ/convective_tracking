from __future__ import annotations
from typing import Optional, List, Dict, Any
from .level import DetectionContext, TrackingLevel
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

class HierarchicalTrackingEngine:
    """Runs a N number of hierarchical detection levels."""

    def __init__(self,  levels: List[TrackingLevel] ):
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

    def run_time_step(self, dataset, time_index: int, time_value: Optional[str] = None, timestamp_s: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        time_label = str(time_index if time_value is None else time_value)
        context = DetectionContext(
            dataset=dataset,
            time_index=time_index,
            time_value=time_label,
            timestamp_s=timestamp_s,
            metadata=dict(metadata or {}),
        )

        level_results = {}
        for level in self.levels:
            parents = [None] if level.parent_name is None else level_results.get(level.parent_name, [])
            objects = []

            if len(parents) >= 2:
                with ThreadPoolExecutor(max_workers=self._get_workers(len(parents))) as executor:
                    future_to_parent = {
                        executor.submit(level.detect, context=context, parent_object=parent): parent
                        for parent in parents
                    }
                    for future in as_completed(future_to_parent):
                        parent = future_to_parent[future]
                        for item in future.result():
                            obj = dict(item)
                            obj.setdefault("level", level.name)
                            obj.setdefault("object_id", f"t{time_label}_{level.name}_{len(objects) + 1:04d}")
                            obj.setdefault("time_value", time_label)
                            if parent is not None:
                                obj.setdefault("parent_object_id", parent.get("object_id"))
                            objects.append(obj)
            else:
                for parent in parents:
                    for item in level.detect(context=context, parent_object=parent):
                        obj = dict(item)
                        obj.setdefault("level", level.name)
                        obj.setdefault("object_id", f"t{time_label}_{level.name}_{len(objects) + 1:04d}")
                        obj.setdefault("time_value", time_label)
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
        datasets: List[Any],
        time_values: Optional[List[str]] = None,
        timestamps_s: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        values = list(time_values) if time_values is not None else []
        ts_values = list(timestamps_s) if timestamps_s is not None else []

        with ThreadPoolExecutor(max_workers=self._get_workers(len(datasets))) as executor:
            futures = []
            for idx, dataset in enumerate(datasets):
                future = executor.submit(
                    self.run_time_step,
                    dataset=dataset,
                    time_index=idx,
                    time_value=str(values[idx] if idx < len(values) else idx),
                    timestamp_s=ts_values[idx] if idx < len(ts_values) else None,
                    metadata=metadata,
                )
                futures.append(future)

            for future in futures:
                future.result()

        return self.results_by_time

    def finalize_tracks(self):
        return {level.name: level.finalize_tracks() for level in self.levels}

    def run_and_track(
        self,
        datasets: List[Any],
        time_values: Optional[List[str]] = None,
        timestamps_s: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        detections_by_time = []
        for idx, dataset in enumerate(datasets):
            detections_by_time.append(
                self._detect_only(
                    dataset=dataset,
                    time_index=idx,
                    time_value=str(time_values[idx]) if time_values else str(idx),
                    timestamp_s=timestamps_s[idx] if timestamps_s else None,
                    metadata=metadata,
                )
            )

        for step in detections_by_time:
            for level in self.levels:
                level.track(
                    step["time_index"],
                    step["time"],
                    step["timestamp_s"],
                    step["levels"][level.name],
                )
            self.results_by_time.append(step)

        return {"results_by_time": self.results_by_time, "tracks_by_level": self.finalize_tracks()}

    def _detect_only(
        self,
        dataset: Any,
        time_index: int,
        time_value: str,
        timestamp_s: Optional[float],
        metadata: Optional[Dict[str, Any]],
    ):
        time_label = time_value
        context = DetectionContext(
            dataset=dataset,
            time_index=time_index,
            time_value=time_label,
            timestamp_s=timestamp_s,
            metadata=dict(metadata or {}),
        )

        level_results = {}
        for level in self.levels:
            parents = [None] if level.parent_name is None else level_results.get(level.parent_name, [])
            objects = []

            for parent in parents:
                for item in level.detect(context=context, parent_object=parent):
                    obj = dict(item)
                    obj.setdefault("level", level.name)
                    obj.setdefault("object_id", f"t{time_label}_{level.name}_{len(objects) + 1:04d}")
                    obj.setdefault("time_value", time_label)
                    if parent is not None:
                        obj.setdefault("parent_object_id", parent.get("object_id"))
                    objects.append(obj)

            level_results[level.name] = objects

        return {
            "time_index": time_index,
            "time": time_label,
            "timestamp_s": timestamp_s,
            "levels": level_results,
        }

    def _get_workers(self, n_parents):
        if n_parents <= 1:
            return 1
        return min(n_parents, os.cpu_count() or 4)
