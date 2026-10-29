from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from .level import DetectionContext, TrackingLevel


def _detect_level(level, context, parents):
    objects = []
    for parent in parents:
        for item in level.detect(context=context, parent_object=parent):
            obj = dict(item)
            obj.setdefault("level", level.name)
            obj.setdefault("object_id", f"t{context.time_value}_{level.name}_{len(objects) + 1:04d}")
            obj.setdefault("time_value", context.time_value)
            if parent is not None:
                obj.setdefault("parent_object_id", parent.get("object_id"))
            objects.append(obj)
    return objects


def _detect_step(levels, dataset, time_index, time_value, timestamp_s, metadata):
    """Detect every level of one timestep, parents first. Pure: no tracker is touched."""
    context = DetectionContext(
        dataset=dataset,
        time_index=time_index,
        time_value=time_value,
        timestamp_s=timestamp_s,
        metadata=dict(metadata or {}),
    )
    level_results = {}
    for level in levels:
        parents = [None] if level.parent_name is None else level_results.get(level.parent_name, [])
        level_results[level.name] = _detect_level(level, context, parents)
    return {"time_index": time_index, "time": time_value, "timestamp_s": timestamp_s, "levels": level_results}


class HierarchicalTrackingEngine:
    """Runs N hierarchical detection levels, then tracks each level in chronological order."""

    def __init__(self, levels: List[TrackingLevel]):
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

    def reset(self):
        """Forget previous results and tracker states."""
        self.results_by_time = []
        for level in self.levels:
            reset = getattr(level.tracker, "reset", None)
            if callable(reset):
                reset()

    def detect_time_step(self, dataset, time_index: int, time_value=None, timestamp_s=None, metadata=None):
        label = str(time_index if time_value is None else time_value)
        return _detect_step(self.levels, dataset, time_index, label, timestamp_s, metadata)

    def _track(self, step):
        for level in self.levels:
            level.track(step["time_index"], step["time"], step["timestamp_s"], step["levels"][level.name])
        self.results_by_time.append(step)

    def run_time_step(self, dataset, time_index: int, time_value=None, timestamp_s=None, metadata=None):
        """Detect and track one timestep; call it in chronological order."""
        step = self.detect_time_step(dataset, time_index, time_value, timestamp_s, metadata)
        self._track(step)
        return step

    def run(
        self,
        datasets: Sequence[Any],
        time_values: Optional[Sequence[Any]] = None,
        timestamps_s: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        n_workers: int = 1,
        executor: str = "thread",
        reset: bool = True,
    ):
        """Detect all timesteps (optionally in parallel), then track them in order.

        ``executor="process"`` requires picklable datasets, detectors and rules.
        """
        if reset:
            self.reset()
        values = list(time_values) if time_values is not None else []
        stamps = list(timestamps_s) if timestamps_s is not None else []
        jobs = [
            (
                idx,
                str(values[idx]) if idx < len(values) else str(idx),
                stamps[idx] if idx < len(stamps) else None,
            )
            for idx in range(len(datasets))
        ]

        n_workers = min(max(1, int(n_workers)), len(jobs) or 1, os.cpu_count() or 1)
        if n_workers == 1:
            steps = [_detect_step(self.levels, datasets[i], i, label, ts, metadata) for i, label, ts in jobs]
        else:
            if executor == "process":
                pool_cls, levels = ProcessPoolExecutor, [replace(level, tracker=None) for level in self.levels]
            elif executor == "thread":
                pool_cls, levels = ThreadPoolExecutor, self.levels
            else:
                raise ValueError("executor must be 'thread' or 'process'")
            with pool_cls(max_workers=n_workers) as pool:
                futures = [pool.submit(_detect_step, levels, datasets[i], i, label, ts, metadata) for i, label, ts in jobs]
                steps = [f.result() for f in futures]

        for step in steps:
            self._track(step)
        return self.results_by_time

    def finalize_tracks(self):
        return {level.name: level.finalize_tracks() for level in self.levels}

    def run_and_track(
        self,
        datasets: Sequence[Any],
        time_values: Optional[Sequence[Any]] = None,
        timestamps_s: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        n_workers: int = 1,
        executor: str = "thread",
    ):
        self.run(datasets, time_values, timestamps_s, metadata, n_workers=n_workers, executor=executor)
        return {"results_by_time": self.results_by_time, "tracks_by_level": self.finalize_tracks()}
