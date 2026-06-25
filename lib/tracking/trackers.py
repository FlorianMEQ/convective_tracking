from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .functions import iou_from_sets


class IoUTracker:
    """Simple generic tracker based on IoU and Hungarian assignment."""

    def __init__(
        self,
        track_prefix: str,
        index_key: str = "indices_3d",
        min_iou: float = 0.1,
        patience: int = 0,
        grid_size: int = 10000,
    ):
        self.track_prefix = track_prefix
        self.index_key = index_key
        self.min_iou = float(min_iou)
        self.patience = int(patience)
        self.grid_size = int(grid_size)

        self._all_tracks: Dict[str, Dict[str, Any]] = {}
        self._active: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    def _pack_2d(self, idx: np.ndarray) -> np.ndarray:
        return idx[:, 0].astype(np.int64) * self.grid_size + idx[:, 1].astype(np.int64)

    def _pack_3d(self, idx: np.ndarray) -> np.ndarray:
        g = np.int64(self.grid_size)
        return (
            idx[:, 0].astype(np.int64) * g * g
            + idx[:, 1].astype(np.int64) * g
            + idx[:, 2].astype(np.int64)
        )

    def _extract_keys(self, obj: Dict[str, Any]) -> np.ndarray:
        indices = obj.get(self.index_key, [])
        if len(indices) == 0:
            return np.array([], dtype=np.int64)
        arr = np.array(indices, dtype=np.int64)
        if arr.ndim != 2:
            return np.array([], dtype=np.int64)
        if arr.shape[1] == 2:
            return self._pack_2d(arr)
        if arr.shape[1] == 3:
            return self._pack_3d(arr)
        return np.array([], dtype=np.int64)

    def update(self, time_index: int, time_value: str, objects: List[Dict[str, Any]]) -> None:
        items: List[Tuple[Dict[str, Any], np.ndarray]] = []
        for obj in objects:
            keys = self._extract_keys(obj)
            if keys.size == 0:
                continue
            items.append((obj, keys))

        matched_items = set()
        matched_tracks = set()

        if self._active and items:
            track_ids = list(self._active.keys())
            score = np.zeros((len(track_ids), len(items)), dtype=np.float32)

            for r, tid in enumerate(track_ids):
                track_keys = self._active[tid]["keys"]
                for c, (_, item_keys) in enumerate(items):
                    score[r, c] = iou_from_sets(track_keys, item_keys)

            row_ind, col_ind = linear_sum_assignment(1.0 - score)
            for r, c in zip(row_ind, col_ind):
                if score[r, c] < self.min_iou:
                    continue
                tid = track_ids[r]
                obj, keys = items[c]
                matched_tracks.add(tid)
                matched_items.add(c)
                self._active[tid]["keys"] = keys
                self._active[tid]["last_time_index"] = time_index
                self._all_tracks[tid]["history"].append(obj)
                self._all_tracks[tid]["end_time"] = time_value

        for idx, (obj, keys) in enumerate(items):
            if idx in matched_items:
                continue
            self._counter += 1
            tid = f"{self.track_prefix}_{self._counter:04d}"
            self._all_tracks[tid] = {
                "track_id": tid,
                "start_time": time_value,
                "end_time": time_value,
                "history": [obj],
            }
            self._active[tid] = {
                "keys": keys,
                "last_time_index": time_index,
            }

        dead = [
            tid
            for tid, state in self._active.items()
            if time_index - state["last_time_index"] > self.patience
        ]
        for tid in dead:
            del self._active[tid]

    def finalize(self) -> Dict[str, Any]:
        return self._all_tracks
