from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .level import DetectionContext, TrackingLevel


class HierarchicalTrackingEngine:
    """Runs an arbitrary number of hierarchical detection levels."""

    def __init__(self, levels: List[TrackingLevel]):
        self.levels = levels
        self._level_index = {level.name: level for level in levels}
        self._validate_levels()
        self.results_by_time: List[Dict[str, Any]] = []

    def _validate_levels(self) -> None:
        known = set()
        for level in self.levels:
            if level.name in known:
                raise ValueError(f"Duplicate level name: {level.name}")
            known.add(level.name)

        for level in self.levels:
            if level.parent_name is not None and level.parent_name not in self._level_index:
                raise ValueError(
                    f"Level '{level.name}' references unknown parent '{level.parent_name}'"
                )

        order = [lvl.name for lvl in self.levels]
        for level in self.levels:
            if level.parent_name is None:
                continue
            if order.index(level.parent_name) >= order.index(level.name):
                raise ValueError(
                    f"Parent '{level.parent_name}' must appear before child '{level.name}'"
                )

    def run_time_step(
        self,
        dataset: Any,
        time_index: int,
        time_value: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        time_label = str(time_value if time_value is not None else time_index)
        context = DetectionContext(
            dataset=dataset,
            time_index=time_index,
            time_value=time_label,
            metadata=metadata or {},
        )

        level_results: Dict[str, List[Dict[str, Any]]] = {}

        for level in self.levels:
            if level.parent_name is None:
                parents = [None]
            else:
                parents = level_results.get(level.parent_name, [])

            objects: List[Dict[str, Any]] = []
            for parent in parents:
                detected = level.detect(context=context, parent_object=parent)
                for idx, item in enumerate(detected):
                    normalized = dict(item)
                    normalized.setdefault("level", level.name)
                    normalized.setdefault(
                        "object_id",
                        f"t{time_label}_{level.name}_{len(objects) + idx + 1:04d}",
                    )
                    if parent is not None:
                        normalized.setdefault("parent_object_id", parent.get("object_id"))
                    objects.append(normalized)

            level_results[level.name] = objects
            level.update_tracks(time_index=time_index, time_value=time_label, objects=objects)

        step = {
            "time_index": time_index,
            "time": time_label,
            "levels": level_results,
        }
        self.results_by_time.append(step)
        return step

    def run(
        self,
        datasets: Iterable[Any],
        time_values: Optional[Iterable[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if time_values is None:
            values = []
        else:
            values = list(time_values)

        for idx, dataset in enumerate(datasets):
            time_value = values[idx] if idx < len(values) else idx
            self.run_time_step(
                dataset=dataset,
                time_index=idx,
                time_value=str(time_value),
                metadata=metadata,
            )
        return self.results_by_time

    def finalize_tracks(self) -> Dict[str, Dict[str, Any]]:
        output: Dict[str, Dict[str, Any]] = {}
        for level in self.levels:
            output[level.name] = level.finalize_tracks()
        return output

    def run_and_track(
        self,
        datasets: Iterable[Any],
        time_values: Optional[Iterable[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        results = self.run(datasets=datasets, time_values=time_values, metadata=metadata)
        tracks = self.finalize_tracks()
        return {
            "results_by_time": results,
            "tracks_by_level": tracks,
        }
