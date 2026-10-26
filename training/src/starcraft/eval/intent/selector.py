"""Per-agent best-ADE rollout selector for agent-oracle intent evaluation.

Mirrors the per-agent argmin convention already used by the headline
trajectory metric in `src/starcraft/eval/metrics/min_ade.py`. Different
agents may pick different rollout indices — this is the explicit design
choice of "agent oracle over K".

Why argmin on `masked_sum` (not on ADE):
    ADE_per_rollout = masked_sum / valid_count
    valid_count is constant across rollouts for a fixed agent (it depends
    only on `gt_valid[i]`, not on which rollout we picked). So
        argmin_r masked_sum[i, r]  ==  argmin_r ADE[i, r]
    We compute the unnormalized argmin (one fewer division per row) and
    divide afterwards only for the diagnostic ADE column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AgentOracleSelection:
    """Per-agent rollout selection result.

    Arrays are indexed by the same `N_target` axis as `ScenarioRollout.agent_id`.
    """

    selected_rollout: np.ndarray   # [N] int64, in [0, R)
    ade: np.ndarray                # [N] float32, mean displacement of selected rollout
    valid_steps: np.ndarray        # [N] int32, number of valid future steps
    has_any_valid: np.ndarray      # [N] bool, True iff agent has >=1 valid step


def select_best_ade_per_agent(
    pred_traj: np.ndarray,   # [N, R, T, 2] float
    gt_traj: np.ndarray,     # [N, T, 2] float
    gt_valid: np.ndarray,    # [N, T] bool, already AND'd with alive_at_current
) -> AgentOracleSelection:
    """Pick, for each agent, the rollout index minimizing ADE to GT.

    Tie-break: lowest rollout index (numpy `argmin` default).

    Agents with zero valid steps get `selected_rollout = 0` and `ade = nan`;
    callers should mask them out via `has_any_valid` before downstream use.
    """
    if pred_traj.ndim != 4 or pred_traj.shape[-1] != 2:
        raise ValueError(
            f"pred_traj must have shape [N, R, T, 2], got {pred_traj.shape}"
        )
    if gt_traj.shape != (pred_traj.shape[0], pred_traj.shape[2], 2):
        raise ValueError(
            f"gt_traj shape {gt_traj.shape} inconsistent with pred_traj "
            f"{pred_traj.shape}"
        )
    if gt_valid.shape != (pred_traj.shape[0], pred_traj.shape[2]):
        raise ValueError(
            f"gt_valid shape {gt_valid.shape} inconsistent with pred_traj "
            f"{pred_traj.shape}"
        )

    pred = pred_traj.astype(np.float32, copy=False)
    gt = gt_traj.astype(np.float32, copy=False)
    valid = gt_valid.astype(np.float32, copy=False)

    # [N, R, T] L2 displacements; per-agent valid mask is broadcast across rollouts.
    dist = np.linalg.norm(pred - gt[:, None, :, :], axis=-1)
    masked_sum = (dist * valid[:, None, :]).sum(axis=-1)        # [N, R]
    selected = masked_sum.argmin(axis=-1).astype(np.int64)      # [N]

    valid_steps = valid.sum(axis=-1).astype(np.int32)            # [N]
    has_any_valid = valid_steps > 0
    # Diagnostic ADE: divide picked rollout's masked_sum by valid_steps.
    picked_sum = masked_sum[np.arange(masked_sum.shape[0]), selected]  # [N]
    with np.errstate(invalid="ignore", divide="ignore"):
        ade = np.where(
            has_any_valid,
            picked_sum / np.maximum(valid_steps, 1).astype(np.float32),
            np.nan,
        ).astype(np.float32)

    return AgentOracleSelection(
        selected_rollout=selected,
        ade=ade,
        valid_steps=valid_steps,
        has_any_valid=has_any_valid,
    )
