"""Tests for the per-agent best-ADE rollout selector."""

from __future__ import annotations

import numpy as np
import pytest

from src.starcraft.eval.intent.selector import select_best_ade_per_agent


def test_selects_per_agent_argmin():
    # N=2, R=3, T=4. Agent 0 should pick rollout 1; agent 1 should pick rollout 2.
    pred = np.zeros((2, 3, 4, 2), dtype=np.float32)
    gt = np.zeros((2, 4, 2), dtype=np.float32)
    valid = np.ones((2, 4), dtype=bool)

    # Agent 0: x-only offsets 5 / 1 / 3 across rollouts. y=0 → ADE=offset.
    pred[0, 0, :, 0] = 5.0
    pred[0, 1, :, 0] = 1.0
    pred[0, 2, :, 0] = 3.0
    # Agent 1: x-only offsets 2 / 4 / 0.5
    pred[1, 0, :, 0] = 2.0
    pred[1, 1, :, 0] = 4.0
    pred[1, 2, :, 0] = 0.5

    sel = select_best_ade_per_agent(pred_traj=pred, gt_traj=gt, gt_valid=valid)
    np.testing.assert_array_equal(sel.selected_rollout, [1, 2])
    np.testing.assert_allclose(sel.ade, [1.0, 0.5], atol=1e-5)
    np.testing.assert_array_equal(sel.valid_steps, [4, 4])
    np.testing.assert_array_equal(sel.has_any_valid, [True, True])


def test_tiebreak_prefers_lowest_rollout_id():
    # All three rollouts are equidistant; argmin returns 0.
    pred = np.full((1, 3, 2, 2), 1.0, dtype=np.float32)
    gt = np.zeros((1, 2, 2), dtype=np.float32)
    valid = np.ones((1, 2), dtype=bool)
    sel = select_best_ade_per_agent(pred_traj=pred, gt_traj=gt, gt_valid=valid)
    assert sel.selected_rollout[0] == 0


def test_skips_agent_with_no_valid_steps():
    pred = np.zeros((2, 2, 3, 2), dtype=np.float32)
    pred[0] = 1.0  # all rollouts equally bad for agent 0
    gt = np.zeros((2, 3, 2), dtype=np.float32)
    valid = np.array([
        [True, True, True],
        [False, False, False],
    ], dtype=bool)
    sel = select_best_ade_per_agent(pred_traj=pred, gt_traj=gt, gt_valid=valid)
    np.testing.assert_array_equal(sel.has_any_valid, [True, False])
    assert np.isnan(sel.ade[1])
    assert sel.valid_steps[1] == 0


def test_argmin_equivalence_masked_sum_vs_ade():
    # Confirm the design comment: argmin(masked_sum) == argmin(ADE) because
    # valid.sum() is constant across rollouts for a fixed agent.
    rng = np.random.default_rng(0)
    N, R, T = 4, 5, 6
    pred = rng.normal(size=(N, R, T, 2)).astype(np.float32)
    gt = rng.normal(size=(N, T, 2)).astype(np.float32)
    valid = (rng.random((N, T)) > 0.3).astype(bool)
    valid[0] = True  # ensure at least one agent has all-True

    sel = select_best_ade_per_agent(pred_traj=pred, gt_traj=gt, gt_valid=valid)
    # Reference: compute ADE explicitly and argmin.
    dist = np.linalg.norm(pred - gt[:, None, :, :], axis=-1)
    masked_sum = (dist * valid[:, None, :]).sum(-1)
    valid_count = valid.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ade_ref = np.where(
            valid_count[:, None] > 0,
            masked_sum / np.maximum(valid_count[:, None], 1),
            np.inf,
        )
    expected = ade_ref.argmin(axis=-1)
    np.testing.assert_array_equal(sel.selected_rollout, expected)


def test_rejects_shape_mismatch():
    pred = np.zeros((2, 3, 4, 2))
    bad_gt = np.zeros((2, 5, 2))
    valid = np.ones((2, 4), dtype=bool)
    with pytest.raises(ValueError):
        select_best_ade_per_agent(pred_traj=pred, gt_traj=bad_gt, gt_valid=valid)
