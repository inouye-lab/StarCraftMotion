"""Tests for the intent candidate matcher and pooled F1 computation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.starcraft.eval.intent.candidates import generate_candidates
from src.starcraft.eval.intent.metrics import (
    compute_metric_records,
    pool_f1_records,
)


def _gt_event(scenario, observer, unit_id, eid, t_native, action,
              has_target=True, tx=0.0, ty=0.0, target_unit=-1, length=1):
    return {
        "scenario_id": scenario,
        "observer": int(observer),
        "mode": "own",
        "unit_id": int(unit_id),
        "gt_event_id": int(eid),
        "t_2hz_step": int((t_native - 24) // 8),
        "t_native_frame": int(t_native),
        "coarse_action": int(action),
        "has_target": bool(has_target),
        "target_x": float(tx) if has_target else float("nan"),
        "target_y": float(ty) if has_target else float("nan"),
        "target_unit_id": int(target_unit),
        "segment_length_steps": int(length),
    }


def _pred_event(scenario, observer, unit_id, eid, rollout, t_native, action,
                has_target=True, tx=0.0, ty=0.0, length=1):
    return {
        "scenario_id": scenario,
        "observer": int(observer),
        "mode": "own",
        "unit_id": int(unit_id),
        "pred_event_id": int(eid),
        "rollout_id": int(rollout),
        "t_2hz_step": int((t_native - 24) // 8),
        "t_native_frame": int(t_native),
        "coarse_action": int(action),
        "has_target": bool(has_target),
        "target_x": float(tx) if has_target else float("nan"),
        "target_y": float(ty) if has_target else float("nan"),
        "segment_length_steps": int(length),
    }


def _meta(scenario_id="s0"):
    return {scenario_id: {"map_name": "Test_LE", "stage": "val"}}


def test_identity_predictions_yield_perfect_f1():
    gt = [
        _gt_event("s0", 1, 100, 0, 32, action=0, has_target=True, tx=10, ty=10),
        _gt_event("s0", 1, 100, 1, 56, action=1, has_target=True, tx=30, ty=30),
        _gt_event("s0", 1, 200, 0, 24, action=0, has_target=False),
    ]
    pred = [
        _pred_event("s0", 1, 100, 0, 0, 32, action=0, tx=10, ty=10),
        _pred_event("s0", 1, 100, 1, 0, 56, action=1, tx=30, ty=30),
        _pred_event("s0", 1, 200, 0, 0, 24, action=0, has_target=False),
    ]
    cands = generate_candidates(gt_events=gt, pred_events=pred, w_max=48)
    recs = compute_metric_records(
        gt_events=gt,
        pred_events=pred,
        candidates=cands,
        scenario_meta=_meta(),
        w=16,
        d_sweep=[3.0],
        require_action_match=True,
    )
    pooled = pool_f1_records(recs)
    f1_records = {
        (int(r["coarse_action"]), float(r["d"])): r
        for r in pooled if r["metric"].startswith("agent_oracle_intent_f1")
    }
    assert math.isclose(f1_records[(0, 3.0)]["value"], 1.0)
    assert math.isclose(f1_records[(1, 3.0)]["value"], 1.0)


def test_temporal_shift_pred_within_window_still_matches():
    gt = [_gt_event("s0", 1, 100, 0, 32, action=0, tx=10, ty=10)]
    # +8 native frames shift.
    pred = [_pred_event("s0", 1, 100, 0, 0, 40, action=0, tx=10, ty=10)]
    cands = generate_candidates(gt_events=gt, pred_events=pred, w_max=48)
    recs = compute_metric_records(
        gt_events=gt, pred_events=pred, candidates=cands,
        scenario_meta=_meta(), w=16, d_sweep=[3.0],
        require_action_match=True,
    )
    pooled = pool_f1_records(recs)
    f1 = next(r for r in pooled
              if r["metric"].startswith("agent_oracle_intent_f1") and int(r["coarse_action"]) == 0)
    assert math.isclose(f1["value"], 1.0)


def test_temporal_shift_above_w_becomes_fp_plus_fn():
    gt = [_gt_event("s0", 1, 100, 0, 32, action=0, tx=10, ty=10)]
    pred = [_pred_event("s0", 1, 100, 0, 0, 40, action=0, tx=10, ty=10)]
    cands = generate_candidates(gt_events=gt, pred_events=pred, w_max=48)
    # w=4 < dt=8 → no match.
    recs = compute_metric_records(
        gt_events=gt, pred_events=pred, candidates=cands,
        scenario_meta=_meta(), w=4, d_sweep=[3.0],
        require_action_match=True,
    )
    counts = {r["metric"].split("_")[3].split("[")[0]: int(r["value"])
              for r in recs if int(r["coarse_action"]) == 0}
    assert counts.get("tp", 0) == 0
    assert counts.get("fp", 0) == 1
    assert counts.get("fn", 0) == 1


def test_missing_target_rule_action_only_matches_action_only():
    gt = [_gt_event("s0", 1, 100, 0, 32, action=0, has_target=False)]
    pred = [_pred_event("s0", 1, 100, 0, 0, 32, action=0, has_target=False)]
    cands = generate_candidates(gt_events=gt, pred_events=pred, w_max=48)
    recs = compute_metric_records(
        gt_events=gt, pred_events=pred, candidates=cands,
        scenario_meta=_meta(), w=16, d_sweep=[3.0],
        require_action_match=True,
    )
    counts = {r["kind"]: int(r["value"]) for r in recs}
    assert counts["tp"] == 1
    assert counts.get("fp", 0) == 0
    assert counts.get("fn", 0) == 0


def test_missing_target_mismatch_does_not_match():
    # GT has target, pred does not — should fail spatial check.
    gt = [_gt_event("s0", 1, 100, 0, 32, action=0, has_target=True, tx=10, ty=10)]
    pred = [_pred_event("s0", 1, 100, 0, 0, 32, action=0, has_target=False)]
    cands = generate_candidates(gt_events=gt, pred_events=pred, w_max=48)
    recs = compute_metric_records(
        gt_events=gt, pred_events=pred, candidates=cands,
        scenario_meta=_meta(), w=16, d_sweep=[3.0],
        require_action_match=True,
    )
    counts = {r["kind"]: int(r["value"]) for r in recs}
    assert counts.get("tp", 0) == 0
    assert counts["fp"] == 1
    assert counts["fn"] == 1


def test_spatial_threshold_filters_far_targets():
    gt = [_gt_event("s0", 1, 100, 0, 32, action=0, tx=10, ty=10)]
    pred = [_pred_event("s0", 1, 100, 0, 0, 32, action=0, tx=20, ty=10)]   # dxy=10
    cands = generate_candidates(gt_events=gt, pred_events=pred, w_max=48)
    # d=3 → no match; d=12 → match.
    recs_strict = compute_metric_records(
        gt_events=gt, pred_events=pred, candidates=cands,
        scenario_meta=_meta(), w=16, d_sweep=[3.0],
        require_action_match=True,
    )
    recs_loose = compute_metric_records(
        gt_events=gt, pred_events=pred, candidates=cands,
        scenario_meta=_meta(), w=16, d_sweep=[12.0],
        require_action_match=True,
    )
    strict = {r["kind"]: int(r["value"]) for r in recs_strict}
    loose = {r["kind"]: int(r["value"]) for r in recs_loose}
    assert strict.get("tp", 0) == 0 and strict["fp"] == 1 and strict["fn"] == 1
    assert loose["tp"] == 1


def test_greedy_match_one_to_one():
    # Two GTs and two preds, all pairwise within w/d. Greedy should match
    # closest in time first (dt ordering by |delta_t|).
    gt = [
        _gt_event("s0", 1, 100, 0, 32, action=0, tx=10, ty=10),
        _gt_event("s0", 1, 100, 1, 40, action=0, tx=10, ty=10),
    ]
    pred = [
        _pred_event("s0", 1, 100, 0, 0, 32, action=0, tx=10, ty=10),
        _pred_event("s0", 1, 100, 1, 0, 40, action=0, tx=10, ty=10),
    ]
    cands = generate_candidates(gt_events=gt, pred_events=pred, w_max=48)
    recs = compute_metric_records(
        gt_events=gt, pred_events=pred, candidates=cands,
        scenario_meta=_meta(), w=16, d_sweep=[3.0],
        require_action_match=True,
    )
    counts = {r["kind"]: int(r["value"]) for r in recs}
    assert counts["tp"] == 2
    assert counts.get("fp", 0) == 0
    assert counts.get("fn", 0) == 0


def test_pooling_aggregates_counts_across_scenarios():
    gt = [
        _gt_event("s0", 1, 100, 0, 32, action=0, tx=10, ty=10),
        _gt_event("s1", 1, 100, 0, 32, action=0, tx=10, ty=10),
    ]
    pred = [
        _pred_event("s0", 1, 100, 0, 0, 32, action=0, tx=10, ty=10),
        # s1 has no matching pred → FN
    ]
    cands = generate_candidates(gt_events=gt, pred_events=pred, w_max=48)
    meta = {
        "s0": {"map_name": "Test_LE", "stage": "val"},
        "s1": {"map_name": "Test_LE", "stage": "val"},
    }
    recs = compute_metric_records(
        gt_events=gt, pred_events=pred, candidates=cands,
        scenario_meta=meta, w=16, d_sweep=[3.0],
        require_action_match=True,
    )
    pooled = pool_f1_records(recs)
    pooled_by = {r["metric"]: r for r in pooled}
    # tp=1, fp=0, fn=1 → precision=1.0, recall=0.5, f1≈0.6667
    f1 = next(r for r in pooled
              if r["metric"].startswith("agent_oracle_intent_f1") and int(r["coarse_action"]) == 0)
    assert math.isclose(f1["value"], 2 * 1.0 * 0.5 / (1.0 + 0.5))
