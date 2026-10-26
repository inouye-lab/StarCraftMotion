"""Threshold-free candidate generation for intent matching.

For each (scenario, observer, agent) bucket we produce all (gt_event,
pred_event) pairs whose native-frame onsets fall within `w_max` frames of
each other. Thresholds (`w`, `d`, `require_action_match`) are applied at
analysis time in `metrics.py`, so this stage is one-and-done per rollout
dump and downstream sweeps are cheap.

Default `w_max=48` native frames = 3 seconds at 16 Hz, generous enough to
contain any reasonable temporal threshold sweep.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

import numpy as np


def _bucket_key(ev: dict) -> tuple:
    return (ev["scenario_id"], int(ev["observer"]), int(ev["unit_id"]))


def generate_candidates(
    *,
    gt_events: Iterable[dict],
    pred_events: Iterable[dict],
    w_max: int = 48,
) -> List[dict]:
    """Cross-product GT × Pred per (scenario, observer, agent), keep pairs
    within |delta_t_native| <= w_max."""
    gt_events = list(gt_events)
    pred_events = list(pred_events)

    by_bucket_gt: dict = defaultdict(list)
    for ev in gt_events:
        by_bucket_gt[_bucket_key(ev)].append(ev)
    by_bucket_pred: dict = defaultdict(list)
    for ev in pred_events:
        by_bucket_pred[_bucket_key(ev)].append(ev)

    out: List[dict] = []
    for key in set(by_bucket_gt.keys()) | set(by_bucket_pred.keys()):
        gts = by_bucket_gt.get(key, [])
        preds = by_bucket_pred.get(key, [])
        if not gts or not preds:
            # FN-only or FP-only buckets are accounted for at metric time
            # by reading the full event tables alongside the candidates;
            # the candidates table itself only carries pair rows.
            continue
        scenario_id, observer, unit_id = key
        for g in gts:
            gt_t = int(g["t_native_frame"])
            gt_action = int(g["coarse_action"])
            gt_has_t = bool(g["has_target"])
            gt_x = float(g["target_x"]) if gt_has_t else float("nan")
            gt_y = float(g["target_y"]) if gt_has_t else float("nan")
            for p in preds:
                pred_t = int(p["t_native_frame"])
                dt = pred_t - gt_t
                if abs(dt) > w_max:
                    continue
                pred_action = int(p["coarse_action"])
                pred_has_t = bool(p["has_target"])
                if gt_has_t and pred_has_t:
                    px = float(p["target_x"])
                    py = float(p["target_y"])
                    dxy = float(np.hypot(px - gt_x, py - gt_y))
                else:
                    dxy = float("nan")
                out.append({
                    "scenario_id": scenario_id,
                    "observer": int(observer),
                    "mode": "own",
                    "unit_id": int(unit_id),
                    "rollout_id": int(p["rollout_id"]),
                    "gt_event_id": int(g["gt_event_id"]),
                    "pred_event_id": int(p["pred_event_id"]),
                    "delta_t_frames": int(dt),
                    "delta_xy": dxy,
                    "pred_action": pred_action,
                    "gt_action": gt_action,
                    "action_match": bool(pred_action == gt_action),
                    "gt_has_target": gt_has_t,
                    "pred_has_target": pred_has_t,
                    "t_gt_native": gt_t,
                    "t_pred_native": pred_t,
                })
    return out
