"""Extract predicted intent events from saved aux head outputs.

For each agent, we read the **selected rollout** slice (different agents
may use different rollout indices — agent-oracle convention). Aux outputs
live under `/obs_p{O}/own/aux/` in the rollout HDF5 (see
`src/starcraft/utils/sc_rollout_io.py`):

    has_action_pred       [N, R, 16] bool
    action_class_pred     [N, R, 16] uint8 (values 0..10)
    has_target_pos_pred   [N, R, 16] bool
    target_pos_pred       [N, R, 16, 2] fp16 (agent-LOCAL, /200 normalized)

The 2Hz step axis (length 16) corresponds to native frames {24, 32, ..., 144}
and to indices `8*s + 7` into `pred_traj[N, R, 128, 2]` /
`pred_head[N, R, 128]` (which cover native frames 17..144).

Inverse target_pos transform (forward in sc_token_processor.py:417-421):
    forward:  rel = R(-h) @ (target_global - agent_pos) / 200
    inverse:  target_global = agent_pos + 200 * R(+h) @ rel

We anchor each predicted event at the **first** 2Hz step of the segment.
Segmentation is symmetric with the GT extractor: a run of consecutive
"True has_action" steps with the same action class and either
(a) both action-only or (b) both targeted with chain-comparison target
distance <= `target_merge_dist` collapses to one event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# 2Hz step index s ∈ {0..15}  ↔  native frame (s + 3) * 8  ∈ {24, 32, ..., 144}
# pred_traj/pred_head index for step s: 8*s + 7  (covers native frames 17..144).
_TARGET_POS_NORM = 200.0
_NUM_2HZ_STEPS = 16


def step_2hz_to_native_frame(s: int) -> int:
    return (s + 3) * 8


def step_2hz_to_pred_idx(s: int) -> int:
    return 8 * s + 7


def inverse_rel_target_to_global(
    rel_norm: np.ndarray,        # [..., 2] (agent-local, /200 normalized)
    agent_pos: np.ndarray,       # [..., 2] (global)
    agent_heading: np.ndarray,   # [...]
) -> np.ndarray:
    """Undo the forward rel-target transform from sc_token_processor.py.

        rel_x =  cos(h) * (tx - px) + sin(h) * (ty - py),  rel_x_norm = rel_x / 200
        rel_y = -sin(h) * (tx - px) + cos(h) * (ty - py),  rel_y_norm = rel_y / 200

    Inverse:
        dx =  cos(h) * (200 * rel_x_norm) - sin(h) * (200 * rel_y_norm)
        dy =  sin(h) * (200 * rel_x_norm) + cos(h) * (200 * rel_y_norm)
        target_global = agent_pos + (dx, dy)
    """
    rx = rel_norm[..., 0].astype(np.float32) * _TARGET_POS_NORM
    ry = rel_norm[..., 1].astype(np.float32) * _TARGET_POS_NORM
    h = agent_heading.astype(np.float32)
    cos_h = np.cos(h)
    sin_h = np.sin(h)
    dx = cos_h * rx - sin_h * ry
    dy = sin_h * rx + cos_h * ry
    out = np.empty(rel_norm.shape[:-1] + (2,), dtype=np.float32)
    out[..., 0] = agent_pos[..., 0].astype(np.float32) + dx
    out[..., 1] = agent_pos[..., 1].astype(np.float32) + dy
    return out


@dataclass
class _ActiveSegment:
    start_step: int
    action: int
    has_target: bool
    first_target_global: Optional[np.ndarray]   # [2] or None
    prev_target_global: Optional[np.ndarray]    # [2] or None


def _segment_one_agent(
    has_action: np.ndarray,         # [16] bool
    action_class: np.ndarray,       # [16] int (0..10)
    has_target_pos: np.ndarray,     # [16] bool
    target_global: np.ndarray,      # [16, 2] float (only valid where has_target_pos)
    target_merge_dist: float,
) -> List[dict]:
    """Run the segmentation rule over one agent's per-step pred state."""
    events: List[dict] = []
    active: Optional[_ActiveSegment] = None

    def _emit(seg: _ActiveSegment, end_step_inclusive: int) -> None:
        if seg.has_target and seg.first_target_global is not None:
            tx = float(seg.first_target_global[0])
            ty = float(seg.first_target_global[1])
        else:
            tx = float("nan")
            ty = float("nan")
        events.append({
            "t_2hz_step": int(seg.start_step),
            "t_native_frame": step_2hz_to_native_frame(int(seg.start_step)),
            "coarse_action": int(seg.action),
            "has_target": bool(seg.has_target),
            "target_x": tx,
            "target_y": ty,
            "segment_length_steps": int(end_step_inclusive - seg.start_step + 1),
        })

    for s in range(_NUM_2HZ_STEPS):
        if not bool(has_action[s]):
            if active is not None:
                _emit(active, s - 1)
                active = None
            continue

        cur_action = int(action_class[s])
        cur_has_target = bool(has_target_pos[s])
        cur_target = target_global[s] if cur_has_target else None

        new_segment = False
        if active is None:
            new_segment = True
        elif cur_action != active.action:
            new_segment = True
        elif cur_has_target != active.has_target:
            new_segment = True
        elif cur_has_target and active.prev_target_global is not None:
            # Both have targets — chain comparison vs previous step.
            d = float(np.linalg.norm(cur_target - active.prev_target_global))
            if d > target_merge_dist:
                new_segment = True

        if new_segment:
            if active is not None:
                _emit(active, s - 1)
            active = _ActiveSegment(
                start_step=s,
                action=cur_action,
                has_target=cur_has_target,
                first_target_global=cur_target.copy() if cur_target is not None else None,
                prev_target_global=cur_target.copy() if cur_target is not None else None,
            )
        else:
            # Continue current segment; advance prev_target reference only when
            # this step has a target (otherwise leave it None).
            if cur_has_target:
                active.prev_target_global = cur_target.copy()

    if active is not None:
        _emit(active, _NUM_2HZ_STEPS - 1)

    return events


def extract_pred_events(
    *,
    aux: dict,                   # has_action_pred, action_class_pred, has_target_pos_pred, target_pos_pred
    pred_traj: np.ndarray,       # [N, R, 128, 2] (global)
    pred_head: np.ndarray,       # [N, R, 128]
    agent_id: np.ndarray,        # [N] int64 (unit_tag)
    selected_rollout: np.ndarray,    # [N] int (per-agent picked rollout)
    has_any_valid: np.ndarray,       # [N] bool (skip agents with no valid GT step)
    scenario_id: str,
    observer: int,
    target_merge_dist: float = 10.0,
) -> List[dict]:
    """Return one record per predicted event, segmented per the rule above.

    Output rows include scenario/observer/unit identifiers and a per-(unit)
    `pred_event_id` so candidate rows can reference them.
    """
    has_action = aux["has_action_pred"]              # [N, R, 16] bool
    action_class = aux["action_class_pred"]          # [N, R, 16] uint8
    has_target_pos = aux["has_target_pos_pred"]      # [N, R, 16] bool
    rel_target = aux["target_pos_pred"]              # [N, R, 16, 2] fp16

    n_agents = pred_traj.shape[0]
    if has_action.shape[0] != n_agents or rel_target.shape[0] != n_agents:
        raise ValueError(
            "aux row count does not match pred_traj agent count"
        )

    # Indices into the 128-step axis at 2Hz boundaries.
    pred_indices = np.array(
        [step_2hz_to_pred_idx(s) for s in range(_NUM_2HZ_STEPS)], dtype=np.int64,
    )

    out: List[dict] = []
    for i in range(n_agents):
        if not bool(has_any_valid[i]):
            continue
        r = int(selected_rollout[i])
        # Per-step predicted agent pos / heading at 2Hz boundaries.
        agent_pos_2hz = pred_traj[i, r, pred_indices, :].astype(np.float32)   # [16, 2]
        agent_head_2hz = pred_head[i, r, pred_indices].astype(np.float32)      # [16]
        # Inverse-transform predicted relative target -> global XY (always
        # compute; we only emit it when has_target_pos is True).
        rel_i = rel_target[i, r].astype(np.float32)                            # [16, 2]
        target_global = inverse_rel_target_to_global(
            rel_i, agent_pos_2hz, agent_head_2hz,
        )                                                                       # [16, 2]
        events_i = _segment_one_agent(
            has_action=has_action[i, r].astype(bool),
            action_class=action_class[i, r].astype(np.int64),
            has_target_pos=has_target_pos[i, r].astype(bool),
            target_global=target_global,
            target_merge_dist=target_merge_dist,
        )
        for j, ev in enumerate(events_i):
            row = {
                "scenario_id": scenario_id,
                "observer": int(observer),
                "mode": "own",
                "unit_id": int(agent_id[i]),
                "pred_event_id": j,
                "rollout_id": r,
            }
            row.update(ev)
            out.append(row)
    return out
