from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .functions import advect_indices, iou_from_sets, pack_indices

FeatureFn = Callable[[Dict[str, Any]], np.ndarray]

DEFAULT_STEP_SECONDS = 30.0


class AdvectedIoUTracker:
    """Track objects across timesteps with advected IoU and Hungarian assignment.

    Each active track is advected by its steering wind, then scored against the
    new objects by IoU (optionally blended with a morphological similarity when
    ``feature_fn`` is given). Unmatched tracks that overlap a taken object are
    recorded as merges, unmatched objects overlapping a matched track as splits.
    """

    def __init__(
        self,
        track_prefix,
        index_key = "indices_3d",
        min_score = 0.1,
        patience = 0,
        dx_meters = 1000.0,
        grid_size = 10000,
        pre_filter_pixels = 30.0,
        advection_error_ms = 5.0,
        shape_buffer_pixels = 3.0,
        gap_bridge_seconds = 120.0,
        gap_bridge_max_dist_pixels = 10.0,
        feature_fn = None,
        feature_weights = None,
        feat_ema_alpha = 0.7,
        w_overlap = 0.6,
        w_morph = 0.4,
        steering_fields: Sequence[str] = ("u_steering", "v_steering"),
        max_speed_ms = None,
    ):
        if feature_fn is not None and feature_weights is None:
            raise ValueError("feature_weights is required when feature_fn is provided")

        self.track_prefix = track_prefix
        self.index_key = index_key
        self.min_score = float(min_score)
        self.patience = int(patience)
        self.dx_meters = float(dx_meters)
        self.grid_size = int(grid_size)
        self.pre_filter_pixels = float(pre_filter_pixels)
        self.advection_error_ms = float(advection_error_ms)
        self.shape_buffer_pixels = float(shape_buffer_pixels)
        self.gap_bridge_seconds = float(gap_bridge_seconds)
        self.gap_bridge_max_dist_pixels = float(gap_bridge_max_dist_pixels)
        self.feature_fn = feature_fn
        self.feature_weights = (
            None if feature_weights is None else (feature_weights / feature_weights.sum()).astype(np.float32)
        )
        self.feat_ema_alpha = float(feat_ema_alpha)
        self.w_overlap = float(w_overlap)
        self.w_morph = float(w_morph)
        self.steering_fields = steering_fields
        self.max_speed_ms = max_speed_ms

        self._all_tracks: Dict[str, Dict[str, Any]] = {}
        self._active: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    def _morph_similarity(self, feat_a: np.ndarray, feat_b: np.ndarray):
        """Weighted mean of per-feature min/max ratios (0.5 when undefined)."""
        mx = np.maximum(feat_a, feat_b)
        valid = mx > 1e-9
        ratios = np.where(valid, np.minimum(feat_a, feat_b) / np.where(valid, mx, 1.0), 1.0)
        weights = np.where(valid, self.feature_weights, 0.0)
        w_sum = weights.sum()
        return float((ratios * weights).sum() / w_sum) if w_sum > 0 else 0.5

    def _build_items(self, objects):
        items = []
        u_key, v_key = self.steering_fields
        for obj in objects:
            if not obj.get(self.index_key):
                continue
            idx = np.array(obj[self.index_key], dtype=np.int64)
            items.append(
                {
                    "obj": obj,
                    "idx": idx,
                    "keys": pack_indices(idx, self.grid_size),
                    "cy": float(idx[:, -2].mean()),
                    "cx": float(idx[:, -1].mean()),
                    "feat": self.feature_fn(obj) if self.feature_fn else None,
                    "u": float(obj.get(u_key, 0.0)),
                    "v": float(obj.get(v_key, 0.0)),
                }
            )
        return items

    def _search_radius_sq(self, dt: float):
        """Squared max distance (pixels) between a track's advected position and a candidate."""
        if self.max_speed_ms is not None:
            radius = self.max_speed_ms * dt / self.dx_meters if dt > 0 else self.pre_filter_pixels
        else:
            radius = min(
                self.pre_filter_pixels,
                self.advection_error_ms * dt / self.dx_meters + self.shape_buffer_pixels,
            )
        return radius**2

    def _score_matrix(self, track_ids, items, current_ts):
        scores = np.zeros((len(track_ids), len(items)), dtype=np.float32)
        for r, tid in enumerate(track_ids):
            st = self._active[tid]
            dt = current_ts - st["last_timestamp_s"]
            u_pix = st["u"] * dt / self.dx_meters
            v_pix = st["v"] * dt / self.dx_meters
            pred_cy, pred_cx = st["cy"] + v_pix, st["cx"] + u_pix
            shifted_keys = pack_indices(advect_indices(st["idx"], v_pix, u_pix), self.grid_size)
            limit_sq = self._search_radius_sq(dt)

            for c, it in enumerate(items):
                residual_sq = (it["cy"] - pred_cy) ** 2 + (it["cx"] - pred_cx) ** 2
                if residual_sq > limit_sq:
                    continue

                iou = iou_from_sets(shifted_keys, it["keys"])
                if self.feature_fn is None:
                    scores[r, c] = iou
                    continue

                morph = self._morph_similarity(st["feat_ema"], it["feat"])
                if iou > 0.0:
                    scores[r, c] = self.w_overlap * iou + self.w_morph * morph
                elif dt > self.gap_bridge_seconds and residual_sq <= self.gap_bridge_max_dist_pixels**2:
                    # Bridge a temporal gap without overlap, at a discount.
                    scores[r, c] = self.w_morph * morph * 0.8
        return scores

    def _extend_track(
        self, tid, it, time_index, time_value, current_ts
    ):
        st = self._active[tid]
        st.update(
            last_time_index=time_index,
            last_timestamp_s=current_ts,
            cy=it["cy"],
            cx=it["cx"],
            idx=it["idx"],
            u=it["u"],
            v=it["v"],
        )
        if self.feature_fn is not None:
            st["feat_ema"] = self.feat_ema_alpha * st["feat_ema"] + (1.0 - self.feat_ema_alpha) * it["feat"]
        self._all_tracks[tid]["history"].append(it["obj"])
        self._all_tracks[tid]["end_time"] = time_value

    def _start_track(
        self,
        it,
        parent_tid,
        time_index,
        time_value,
        current_ts,
    ):
        self._counter += 1
        tid = f"{self.track_prefix}_{self._counter:04d}"

        # A split child inherits its parent's steering until it has its own history.
        wind_source = self._active.get(parent_tid, it)

        track = {"track_id": tid, "start_time": time_value, "end_time": time_value, "history": [it["obj"]]}
        if parent_tid:
            track["parent_id"] = parent_tid
        self._all_tracks[tid] = track

        self._active[tid] = {
            "last_time_index": time_index,
            "last_timestamp_s": current_ts,
            "cy": it["cy"],
            "cx": it["cx"],
            "idx": it["idx"],
            "u": wind_source["u"],
            "v": wind_source["v"],
            "feat_ema": None if it["feat"] is None else it["feat"].copy(),
        }

    def update(
        self,
        time_index,
        time_value,
        timestamp_s,
        objects,
    ):
        items = self._build_items(objects)
        current_ts = timestamp_s if timestamp_s is not None else time_index * DEFAULT_STEP_SECONDS

        winner_by_item: Dict[int, str] = {}
        split_parent_by_item: Dict[int, str] = {}

        if self._active and items:
            track_ids = list(self._active)
            scores = self._score_matrix(track_ids, items, current_ts)

            for r, c in zip(*linear_sum_assignment(1.0 - scores)):
                if scores[r, c] >= self.min_score:
                    self._extend_track(track_ids[r], items[c], time_index, time_value, current_ts)
                    winner_by_item[c] = track_ids[r]
            matched_tracks = set(winner_by_item.values())
            matched_rows = [r for r, tid in enumerate(track_ids) if tid in matched_tracks]

            # Merges: unmatched active track overlapping an already-taken item.
            for r, tid in enumerate(track_ids):
                if tid in matched_tracks:
                    continue
                for c in sorted(winner_by_item):
                    if scores[r, c] >= self.min_score:
                        self._all_tracks[tid].setdefault("events", []).append(
                            {"type": "merge", "time": time_value, "into": winner_by_item[c]}
                        )
                        del self._active[tid]
                        break

            # Splits: unmatched item overlapping an already-matched track.
            for c, it in enumerate(items):
                if c in winner_by_item or not matched_rows:
                    continue
                best = matched_rows[int(np.argmax(scores[matched_rows, c]))]
                if scores[best, c] >= self.min_score:
                    parent_tid = track_ids[best]
                    self._all_tracks[parent_tid].setdefault("events", []).append(
                        {"type": "split", "time": time_value, "child_centroid": (it["cy"], it["cx"])}
                    )
                    split_parent_by_item[c] = parent_tid

        for c, it in enumerate(items):
            if c not in winner_by_item:
                self._start_track(it, split_parent_by_item.get(c), time_index, time_value, current_ts)

        for tid in [t for t, st in self._active.items() if time_index - st["last_time_index"] > self.patience]:
            del self._active[tid]

    def finalize(self) -> Dict[str, Any]:
        return self._all_tracks
