from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .functions import advect_indices, iou_from_sets, pack_indices

DEFAULT_STEP_SECONDS = 30.0


class AdvectedIoUTracker:
    """Track objects across timesteps with advected IoU and Hungarian assignment.

    Each active track is advected by its steering wind, then scored against the
    new objects by IoU (optionally blended with a morphological similarity when
    ``feature_fn`` is given). Candidates are pre-selected by overlap of the
    advected bounding boxes, widened by the search radius, so that merges and
    splits (whose centroids jump) stay candidates. Unmatched tracks that
    overlap a taken object are recorded as merges, unmatched objects
    overlapping a matched track as splits.
    """

    def __init__(
        self,
        track_prefix,
        index_key="indices_3d",
        min_score=0.1,
        patience=0,
        dx_meters=1000.0,
        grid_size=10000,
        pre_filter_pixels=30.0,
        advection_error_ms=5.0,
        shape_buffer_pixels=3.0,
        gap_bridge_seconds=120.0,
        gap_bridge_max_dist_pixels=10.0,
        feature_fn=None,
        feature_weights=None,
        feat_ema_alpha=0.7,
        w_overlap=0.6,
        w_morph=0.4,
        min_iou=0.0,
        steering_fields=("u_steering", "v_steering"),
        max_speed_ms=None,
        detect_splits_merges=True,
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
            None if feature_weights is None
            else (np.asarray(feature_weights, dtype=np.float32) / np.sum(feature_weights)).astype(np.float32)
        )
        self.feat_ema_alpha = float(feat_ema_alpha)
        self.w_overlap = float(w_overlap)
        self.w_morph = float(w_morph)
        self.min_iou = float(min_iou)
        self.steering_fields = steering_fields
        self.max_speed_ms = max_speed_ms
        self.detect_splits_merges = bool(detect_splits_merges)
        self.reset()

    def reset(self):
        self._all_tracks = {}
        self._active = {}
        self._counter = 0

    def _morph_similarity(self, feat_a, feat_b):
        """Weighted mean of per-feature min/max ratios (0.5 when undefined)."""
        mx = np.maximum(feat_a, feat_b)
        valid = mx > 1e-9
        ratios = np.where(valid, np.minimum(feat_a, feat_b) / np.where(valid, mx, 1.0), 1.0)
        weights = np.where(valid, self.feature_weights, 0.0)
        w_sum = weights.sum()
        return float((ratios * weights).sum() / w_sum) if w_sum > 0 else 0.5

    def _indices(self, obj):
        idx = obj.get(self.index_key)
        if (idx is None or len(idx) == 0) and self.index_key == "indices_2d" and obj.get("indices_3d") is not None:
            idx = np.unique(np.asarray(obj["indices_3d"])[:, 1:], axis=0)
        if idx is None or len(idx) == 0:
            return None
        return np.asarray(idx, dtype=np.int64)

    def _build_items(self, objects):
        items = []
        u_key, v_key = self.steering_fields
        for obj in objects:
            idx = self._indices(obj)
            if idx is None:
                continue
            yx = idx[:, -2:]
            items.append(
                {
                    "obj": obj,
                    "idx": idx,
                    "keys": pack_indices(idx, self.grid_size),
                    "cy": float(yx[:, 0].mean()),
                    "cx": float(yx[:, 1].mean()),
                    "bbox": np.concatenate([yx.min(axis=0), yx.max(axis=0)]).astype(float),
                    "feat": self.feature_fn(obj) if self.feature_fn else None,
                    "u": float(obj.get(u_key) or 0.0),
                    "v": float(obj.get(v_key) or 0.0),
                }
            )
        return items

    def _search_radius(self, dt):
        """Max distance (pixels) an object may drift away from its advected position."""
        if self.max_speed_ms is not None:
            return self.max_speed_ms * dt / self.dx_meters if dt > 0 else self.pre_filter_pixels
        return min(self.pre_filter_pixels, self.advection_error_ms * dt / self.dx_meters + self.shape_buffer_pixels)

    def _score_matrix(self, track_ids, items, current_ts):
        scores = np.zeros((len(track_ids), len(items)), dtype=np.float32)
        boxes = np.array([it["bbox"] for it in items])
        centers = np.array([[it["cy"], it["cx"]] for it in items])

        for r, tid in enumerate(track_ids):
            st = self._active[tid]
            dt = current_ts - st["last_timestamp_s"]
            dy, dx = round(st["v"] * dt / self.dx_meters), round(st["u"] * dt / self.dx_meters)
            m = self._search_radius(dt)
            y0, x0, y1, x1 = st["bbox"] + np.array([dy, dx, dy, dx])
            candidates = np.flatnonzero(
                (boxes[:, 0] <= y1 + m) & (boxes[:, 2] >= y0 - m) & (boxes[:, 1] <= x1 + m) & (boxes[:, 3] >= x0 - m)
            )
            if candidates.size == 0:
                continue

            shifted_keys = pack_indices(advect_indices(st["idx"], dy, dx), self.grid_size)
            residual_sq = ((centers[candidates] - (st["cy"] + dy, st["cx"] + dx)) ** 2).sum(axis=1)
            for c, res_sq in zip(candidates, residual_sq):
                it = items[c]
                iou = iou_from_sets(shifted_keys, it["keys"])
                if self.feature_fn is None or iou < self.min_iou:
                    scores[r, c] = iou
                    continue
                morph = self._morph_similarity(st["feat_ema"], it["feat"])
                if iou > 0.0:
                    scores[r, c] = self.w_overlap * iou + self.w_morph * morph
                elif dt > self.gap_bridge_seconds and res_sq <= self.gap_bridge_max_dist_pixels**2:
                    scores[r, c] = self.w_morph * morph * 0.8
        return scores

    def _set_state(self, tid, it, time_index, current_ts, u, v):
        state = self._active.setdefault(tid, {})
        state.update(
            last_time_index=time_index,
            last_timestamp_s=current_ts,
            cy=it["cy"],
            cx=it["cx"],
            bbox=it["bbox"],
            idx=it["idx"],
            u=u,
            v=v,
        )
        return state

    def _extend_track(self, tid, it, time_index, time_value, current_ts):
        st = self._set_state(tid, it, time_index, current_ts, it["u"], it["v"])
        if self.feature_fn is not None:
            st["feat_ema"] = self.feat_ema_alpha * st["feat_ema"] + (1.0 - self.feat_ema_alpha) * it["feat"]
        track = self._all_tracks[tid]
        track["history"].append(it["obj"])
        track["end_time"] = time_value

    def _start_track(self, it, parent_tid, time_index, time_value, current_ts):
        self._counter += 1
        tid = f"{self.track_prefix}_{self._counter:04d}"
        wind = self._active.get(parent_tid, it)

        track = {"track_id": tid, "start_time": time_value, "end_time": time_value, "history": [it["obj"]]}
        if parent_tid:
            track["parent_id"] = parent_tid
        self._all_tracks[tid] = track

        st = self._set_state(tid, it, time_index, current_ts, wind["u"], wind["v"])
        st["feat_ema"] = None if it["feat"] is None else np.array(it["feat"], dtype=np.float32)
        return tid

    def _event(self, tid, event):
        self._all_tracks[tid].setdefault("events", []).append(event)

    def update(self, time_index, time_value, timestamp_s, objects):
        items = self._build_items(objects)
        current_ts = timestamp_s if timestamp_s is not None else time_index * DEFAULT_STEP_SECONDS

        winner_by_item = {}
        split_parent_by_item = {}

        if self._active and items:
            track_ids = list(self._active)
            scores = self._score_matrix(track_ids, items, current_ts)

            for r, c in zip(*linear_sum_assignment(1.0 - scores)):
                if scores[r, c] >= self.min_score:
                    self._extend_track(track_ids[r], items[c], time_index, time_value, current_ts)
                    winner_by_item[c] = track_ids[r]

            if self.detect_splits_merges and winner_by_item:
                taken = np.array(sorted(winner_by_item))
                matched = set(winner_by_item.values())
                matched_rows = np.array([r for r, tid in enumerate(track_ids) if tid in matched])

                for r, tid in enumerate(track_ids):
                    if tid in matched:
                        continue
                    hits = taken[scores[r, taken] >= self.min_score]
                    if hits.size:
                        into = winner_by_item[int(hits[np.argmax(scores[r, hits])])]
                        self._event(tid, {"type": "merge", "time": time_value, "into": into})
                        self._event(into, {"type": "absorbed", "time": time_value, "from": tid})
                        self._all_tracks[tid]["end_reason"] = "merge"
                        del self._active[tid]

                for c in range(len(items)):
                    if c in winner_by_item:
                        continue
                    best = matched_rows[int(np.argmax(scores[matched_rows, c]))]
                    if scores[best, c] >= self.min_score:
                        split_parent_by_item[c] = track_ids[best]

        for c, it in enumerate(items):
            if c in winner_by_item:
                continue
            parent_tid = split_parent_by_item.get(c)
            tid = self._start_track(it, parent_tid, time_index, time_value, current_ts)
            if parent_tid:
                self._event(
                    parent_tid,
                    {"type": "split", "time": time_value, "child_id": tid, "child_centroid": (it["cy"], it["cx"])},
                )

        for tid in [t for t, st in self._active.items() if time_index - st["last_time_index"] > self.patience]:
            self._all_tracks[tid].setdefault("end_reason", "lost")
            del self._active[tid]

    def finalize(self):
        return self._all_tracks
