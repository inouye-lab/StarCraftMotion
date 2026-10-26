"""Tests for GT and predicted intent event segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from src.starcraft.eval.intent.extract_gt import (
    _segment_one_agent as _segment_gt,
)
from src.starcraft.eval.intent.extract_pred import (
    _segment_one_agent as _segment_pred,
    extract_pred_events,
    inverse_rel_target_to_global,
)


N_STEPS = 16


# ---------------------------------------------------------------------------
# GT segmentation
# ---------------------------------------------------------------------------

def _gt_inputs(n=N_STEPS):
    return {
        "steps_native": np.arange(24, 24 + n * 8, 8, dtype=np.int64),
        "coarse_action": np.zeros(n, dtype=np.int64),
        "has_target_unit": np.zeros(n, dtype=bool),
        "target_unit_tag": np.full(n, -1, dtype=np.int64),
        "has_target_xy": np.zeros(n, dtype=bool),
        "target_xy": np.zeros((n, 2), dtype=np.float32),
        "is_active": np.zeros(n, dtype=bool),
    }


def test_gt_persistent_command_with_target_id_collapses_to_one_event():
    g = _gt_inputs()
    # MOVE (action=0) on target unit tag 100 for steps 0..15.
    g["coarse_action"][:] = 0
    g["has_target_unit"][:] = True
    g["target_unit_tag"][:] = 100
    g["target_xy"][:] = [42.0, 42.0]
    g["is_active"][:] = True
    events = _segment_gt(**g, target_merge_dist=10.0)
    assert len(events) == 1
    assert events[0]["t_2hz_step"] == 0
    assert events[0]["coarse_action"] == 0
    assert events[0]["target_unit_id"] == 100
    assert events[0]["segment_length_steps"] == N_STEPS


def test_gt_no_op_gap_breaks_a_run_into_two_events():
    g = _gt_inputs()
    g["coarse_action"][:] = 1   # ATTACK
    g["has_target_unit"][:] = True
    g["target_unit_tag"][:] = 7
    g["is_active"][:4] = True   # steps 0-3 active
    g["is_active"][4] = False    # NO_OP gap
    g["is_active"][5:9] = True   # steps 5-8 active
    events = _segment_gt(**g, target_merge_dist=10.0)
    assert len(events) == 2
    assert events[0]["t_2hz_step"] == 0
    assert events[0]["segment_length_steps"] == 4
    assert events[1]["t_2hz_step"] == 5
    assert events[1]["segment_length_steps"] == 4


def test_gt_target_unit_moving_within_run_still_one_event():
    g = _gt_inputs()
    g["coarse_action"][:6] = 1
    g["has_target_unit"][:6] = True
    g["target_unit_tag"][:6] = 99
    # Target XY moves 2 cells/step — would split under XY-chain rule, but
    # target_id-based segmentation keeps them as one event.
    g["target_xy"][:6] = np.stack([
        np.arange(6, dtype=np.float32) * 2.0,
        np.zeros(6, dtype=np.float32),
    ], axis=1)
    g["is_active"][:6] = True
    events = _segment_gt(**g, target_merge_dist=10.0)
    assert len(events) == 1
    assert events[0]["segment_length_steps"] == 6


def test_gt_target_id_change_breaks_segment():
    g = _gt_inputs()
    g["coarse_action"][:6] = 1
    g["has_target_unit"][:6] = True
    g["target_unit_tag"][:3] = 7
    g["target_unit_tag"][3:6] = 8
    g["is_active"][:6] = True
    events = _segment_gt(**g, target_merge_dist=10.0)
    assert len(events) == 2
    assert events[0]["target_unit_id"] == 7
    assert events[1]["target_unit_id"] == 8


def test_gt_xy_only_chain_within_threshold_one_event():
    g = _gt_inputs()
    g["coarse_action"][:5] = 0
    g["has_target_xy"][:5] = True
    g["target_xy"][:5] = np.stack([
        np.linspace(0.0, 8.0, 5, dtype=np.float32),
        np.zeros(5, dtype=np.float32),
    ], axis=1)
    g["is_active"][:5] = True
    events = _segment_gt(**g, target_merge_dist=10.0)
    assert len(events) == 1
    assert events[0]["target_unit_id"] == -1   # XY-only path


def test_gt_xy_jump_above_threshold_splits_segment():
    g = _gt_inputs()
    g["coarse_action"][:4] = 0
    g["has_target_xy"][:4] = True
    g["target_xy"][:2] = [5.0, 5.0]
    g["target_xy"][2:4] = [50.0, 50.0]   # >>10 cells away
    g["is_active"][:4] = True
    events = _segment_gt(**g, target_merge_dist=10.0)
    assert len(events) == 2


# ---------------------------------------------------------------------------
# Pred segmentation (segment_one_agent inputs)
# ---------------------------------------------------------------------------

def test_pred_persistent_with_drifting_target_xy_one_event():
    has_action = np.ones(N_STEPS, dtype=bool)
    action_class = np.zeros(N_STEPS, dtype=np.int64)
    has_target_pos = np.ones(N_STEPS, dtype=bool)
    # Target drifts 2 cells/step in x (well within 10).
    target_global = np.stack([
        np.arange(N_STEPS, dtype=np.float32) * 2.0,
        np.zeros(N_STEPS, dtype=np.float32),
    ], axis=1)
    events = _segment_pred(
        has_action=has_action,
        action_class=action_class,
        has_target_pos=has_target_pos,
        target_global=target_global,
        target_merge_dist=10.0,
    )
    assert len(events) == 1
    assert events[0]["segment_length_steps"] == N_STEPS


def test_pred_target_jump_above_threshold_splits():
    has_action = np.zeros(N_STEPS, dtype=bool)
    has_action[:4] = True
    action_class = np.zeros(N_STEPS, dtype=np.int64)
    has_target_pos = np.zeros(N_STEPS, dtype=bool)
    has_target_pos[:4] = True
    target_global = np.zeros((N_STEPS, 2), dtype=np.float32)
    target_global[:2] = [5.0, 5.0]
    target_global[2:4] = [50.0, 50.0]
    events = _segment_pred(
        has_action=has_action,
        action_class=action_class,
        has_target_pos=has_target_pos,
        target_global=target_global,
        target_merge_dist=10.0,
    )
    assert len(events) == 2


def test_pred_action_class_change_splits():
    has_action = np.ones(N_STEPS, dtype=bool)
    action_class = np.zeros(N_STEPS, dtype=np.int64)
    action_class[8:] = 3
    has_target_pos = np.zeros(N_STEPS, dtype=bool)
    target_global = np.zeros((N_STEPS, 2), dtype=np.float32)
    events = _segment_pred(
        has_action=has_action,
        action_class=action_class,
        has_target_pos=has_target_pos,
        target_global=target_global,
        target_merge_dist=10.0,
    )
    assert len(events) == 2


def test_pred_has_target_flip_splits():
    has_action = np.ones(N_STEPS, dtype=bool)
    action_class = np.zeros(N_STEPS, dtype=np.int64)
    has_target_pos = np.zeros(N_STEPS, dtype=bool)
    has_target_pos[:8] = True
    target_global = np.zeros((N_STEPS, 2), dtype=np.float32)
    target_global[:8] = [10.0, 10.0]
    events = _segment_pred(
        has_action=has_action,
        action_class=action_class,
        has_target_pos=has_target_pos,
        target_global=target_global,
        target_merge_dist=10.0,
    )
    assert len(events) == 2


def test_pred_has_action_false_breaks_run():
    has_action = np.zeros(N_STEPS, dtype=bool)
    has_action[:4] = True
    has_action[5:9] = True
    action_class = np.zeros(N_STEPS, dtype=np.int64)
    has_target_pos = np.zeros(N_STEPS, dtype=bool)
    target_global = np.zeros((N_STEPS, 2), dtype=np.float32)
    events = _segment_pred(
        has_action=has_action,
        action_class=action_class,
        has_target_pos=has_target_pos,
        target_global=target_global,
        target_merge_dist=10.0,
    )
    assert len(events) == 2
    assert events[0]["t_2hz_step"] == 0
    assert events[1]["t_2hz_step"] == 5


# ---------------------------------------------------------------------------
# Inverse target_pos transform round-trip
# ---------------------------------------------------------------------------

def test_inverse_target_transform_round_trip():
    rng = np.random.default_rng(0)
    n = 32
    agent_pos = rng.uniform(0, 200, size=(n, 2)).astype(np.float32)
    headings = rng.uniform(-np.pi, np.pi, size=(n,)).astype(np.float32)
    targets_global = rng.uniform(0, 200, size=(n, 2)).astype(np.float32)

    # Forward (training transform)
    delta = targets_global - agent_pos
    cos_h = np.cos(headings)
    sin_h = np.sin(headings)
    rel_x = cos_h * delta[..., 0] + sin_h * delta[..., 1]
    rel_y = -sin_h * delta[..., 0] + cos_h * delta[..., 1]
    rel_norm = np.stack([rel_x, rel_y], axis=-1) / 200.0

    # Inverse
    recovered = inverse_rel_target_to_global(rel_norm, agent_pos, headings)
    np.testing.assert_allclose(recovered, targets_global, atol=1e-3)


# ---------------------------------------------------------------------------
# extract_pred_events end-to-end on synthetic aux arrays
# ---------------------------------------------------------------------------

def test_extract_pred_events_uses_selected_rollout_per_agent():
    n_agents, R, n_2hz = 2, 3, N_STEPS
    pred_len = 128

    has_action = np.zeros((n_agents, R, n_2hz), dtype=bool)
    action_class = np.zeros((n_agents, R, n_2hz), dtype=np.uint8)
    has_target_pos = np.zeros((n_agents, R, n_2hz), dtype=bool)
    target_pos_pred = np.zeros((n_agents, R, n_2hz, 2), dtype=np.float16)

    # Agent 0, rollout 1 ONLY: 8 steps of MOVE (action=0) without target.
    has_action[0, 1, :8] = True
    action_class[0, 1, :8] = 0
    # Agent 1, rollout 2 ONLY: 4 steps of ATTACK (action=1) with same target XY.
    has_action[1, 2, :4] = True
    action_class[1, 2, :4] = 1
    has_target_pos[1, 2, :4] = True
    # Set target_pos to a constant rel value; with zero heading and zero pos,
    # global target = 200 * rel.
    target_pos_pred[1, 2, :4, 0] = np.float16(0.25)   # global x = 50
    target_pos_pred[1, 2, :4, 1] = np.float16(0.10)   # global y = 20

    aux = {
        "has_action_pred": has_action,
        "action_class_pred": action_class,
        "has_target_pos_pred": has_target_pos,
        "target_pos_pred": target_pos_pred,
    }
    pred_traj = np.zeros((n_agents, R, pred_len, 2), dtype=np.float32)
    pred_head = np.zeros((n_agents, R, pred_len), dtype=np.float32)
    selected = np.array([1, 2], dtype=np.int64)
    has_any_valid = np.array([True, True], dtype=bool)
    agent_id = np.array([1001, 1002], dtype=np.int64)

    events = extract_pred_events(
        aux=aux,
        pred_traj=pred_traj,
        pred_head=pred_head,
        agent_id=agent_id,
        selected_rollout=selected,
        has_any_valid=has_any_valid,
        scenario_id="scn0",
        observer=1,
        target_merge_dist=10.0,
    )

    by_unit = {ev["unit_id"]: ev for ev in events}
    assert set(by_unit.keys()) == {1001, 1002}
    assert by_unit[1001]["coarse_action"] == 0
    assert by_unit[1001]["rollout_id"] == 1
    assert by_unit[1001]["has_target"] is False
    assert by_unit[1002]["coarse_action"] == 1
    assert by_unit[1002]["rollout_id"] == 2
    assert by_unit[1002]["has_target"] is True
    np.testing.assert_allclose(
        [by_unit[1002]["target_x"], by_unit[1002]["target_y"]],
        [50.0, 20.0],
        atol=0.5,
    )


def test_extract_pred_events_skips_no_valid_agents():
    n_agents, R, n_2hz = 2, 1, N_STEPS
    pred_len = 128
    has_action = np.zeros((n_agents, R, n_2hz), dtype=bool)
    has_action[0, 0, :2] = True
    has_action[1, 0, :2] = True
    aux = {
        "has_action_pred": has_action,
        "action_class_pred": np.zeros((n_agents, R, n_2hz), dtype=np.uint8),
        "has_target_pos_pred": np.zeros((n_agents, R, n_2hz), dtype=bool),
        "target_pos_pred": np.zeros((n_agents, R, n_2hz, 2), dtype=np.float16),
    }
    pred_traj = np.zeros((n_agents, R, pred_len, 2), dtype=np.float32)
    pred_head = np.zeros((n_agents, R, pred_len), dtype=np.float32)
    events = extract_pred_events(
        aux=aux,
        pred_traj=pred_traj,
        pred_head=pred_head,
        agent_id=np.array([10, 20], dtype=np.int64),
        selected_rollout=np.zeros(2, dtype=np.int64),
        has_any_valid=np.array([True, False], dtype=bool),
        scenario_id="scn",
        observer=1,
    )
    assert all(ev["unit_id"] == 10 for ev in events)
