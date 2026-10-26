"""Extract ground-truth intent events from a raw StarCraft replay.

Reads raw `ability_id`, `target_id`, `target_pos`, `coordinate` and resolves
target unit indices to global XY (mirroring `sc_dataset.py:101-109`). Then
samples the per-step state at the 2Hz future boundaries `{24, 32, ..., 144}`
native frames, applies the same coarse-action LUT used at training time
(`sc_token_processor.py:_vectorized_coarse_action`), and segments runs of
identical command state into single events.

Segmentation rule (per agent):
  * Skip steps with `ability_id == 0` (NO_OP) — these break a run.
  * Group consecutive 2Hz steps by `(coarse_action, target_unit_tag)` when the
    target_id is valid (resolves to a stable unit_tag).
  * Else group by `(coarse_action, target_xy_chain)`: consecutive steps stay
    in the same segment when their resolved target XY moves <= `target_merge_dist`.
  * Else (action only) group by `coarse_action`.

Anchor each event at the **first** 2Hz step of its run. The output schema
matches `extract_pred.py.extract_pred_events` plus `target_unit_id` for the
target_id-resolved case (set to -1 otherwise).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from src.starcraft.utils.coarse_action_mapping import (
    ABILITY_ID_TO_COARSE_ACTION,
    COARSE_ACTION_NAMES,
)
from src.starcraft.utils.sc_replay_io import apply_ever_alive_filter

# Build a numpy LUT identical to `sc_token_processor._COARSE_ACTION_LUT`
# (label 0 unused at supervision time, 1..10 shifted to 0..9, UNKNOWN -> 10).
_MAX_ABILITY_ID = max(ABILITY_ID_TO_COARSE_ACTION.keys())
_NUM_ACTION_CLASSES = 11
_COARSE_ACTION_LUT_NP = np.full(_MAX_ABILITY_ID + 2, 10, dtype=np.int64)
for _aid, _label in ABILITY_ID_TO_COARSE_ACTION.items():
    if _label == 0:
        _COARSE_ACTION_LUT_NP[_aid] = 0
    elif _label == 255:
        _COARSE_ACTION_LUT_NP[_aid] = 10
    else:
        _COARSE_ACTION_LUT_NP[_aid] = _label - 1

# Shifted-label -> human-readable name. The model emits class indices 0..10
# (1..10 shifted down by one + UNKNOWN at 10), so this mapping keeps the
# metric outputs interpretable without re-deriving the shift downstream.
SHIFTED_ACTION_NAMES = {
    i: COARSE_ACTION_NAMES[i + 1] for i in range(10)
}
SHIFTED_ACTION_NAMES[10] = "UNKNOWN"


def vectorized_coarse_action(ability_ids: np.ndarray) -> np.ndarray:
    """numpy port of `sc_token_processor._vectorized_coarse_action`."""
    clamped = np.clip(ability_ids, 0, _MAX_ABILITY_ID + 1).astype(np.int64)
    return _COARSE_ACTION_LUT_NP[clamped]


_NUM_2HZ_STEPS = 16
_TARGET_ID_SENTINEL = np.uint32(0xFFFFFFFF)


def _step_2hz_to_native_frame(s: int) -> int:
    return (s + 3) * 8


def _segment_one_agent(
    steps_native: np.ndarray,        # [16] int (native frame index)
    coarse_action: np.ndarray,       # [16] int (0..10)
    has_target_unit: np.ndarray,     # [16] bool (target_id resolved + alive)
    target_unit_tag: np.ndarray,     # [16] int64 (-1 when has_target_unit=False)
    has_target_xy: np.ndarray,       # [16] bool (resolved target_pos non-zero)
    target_xy: np.ndarray,           # [16, 2] float
    is_active: np.ndarray,           # [16] bool (ability_id != 0)
    target_merge_dist: float,
) -> List[dict]:
    """Segment runs of identical command state into events.

    Within an active run, consecutive steps stay in the same segment when:
        same coarse_action AND
        ((same target_unit_tag, both target_unit-resolved) OR
         (both target_xy-only, chain distance <= target_merge_dist) OR
         (both action-only))
    Anything else closes the current segment and starts a new one. NO_OP
    (is_active=False) breaks any active run.
    """
    events: List[dict] = []

    seg_start: Optional[int] = None
    seg_action: int = -1
    seg_kind: str = ""              # "unit" | "xy" | "none"
    seg_first_target_xy: Optional[np.ndarray] = None
    seg_first_target_tag: int = -1
    seg_prev_target_xy: Optional[np.ndarray] = None

    def _emit(end_step_inclusive: int) -> None:
        if seg_kind == "unit":
            tx, ty = float(seg_first_target_xy[0]), float(seg_first_target_xy[1])
            target_unit = int(seg_first_target_tag)
            has_target = True
        elif seg_kind == "xy":
            tx, ty = float(seg_first_target_xy[0]), float(seg_first_target_xy[1])
            target_unit = -1
            has_target = True
        else:
            tx, ty = float("nan"), float("nan")
            target_unit = -1
            has_target = False
        events.append({
            "t_2hz_step": int(seg_start),
            "t_native_frame": _step_2hz_to_native_frame(int(seg_start)),
            "coarse_action": int(seg_action),
            "has_target": has_target,
            "target_x": tx,
            "target_y": ty,
            "target_unit_id": target_unit,
            "segment_length_steps": int(end_step_inclusive - seg_start + 1),
        })

    for s in range(_NUM_2HZ_STEPS):
        if not bool(is_active[s]):
            if seg_start is not None:
                _emit(s - 1)
                seg_start = None
                seg_first_target_xy = None
                seg_prev_target_xy = None
            continue

        cur_action = int(coarse_action[s])
        if has_target_unit[s]:
            cur_kind = "unit"
            cur_tag = int(target_unit_tag[s])
            cur_xy = target_xy[s]
        elif has_target_xy[s]:
            cur_kind = "xy"
            cur_tag = -1
            cur_xy = target_xy[s]
        else:
            cur_kind = "none"
            cur_tag = -1
            cur_xy = None

        new_segment = False
        if seg_start is None:
            new_segment = True
        elif cur_action != seg_action:
            new_segment = True
        elif cur_kind != seg_kind:
            new_segment = True
        elif cur_kind == "unit" and cur_tag != seg_first_target_tag:
            new_segment = True
        elif cur_kind == "xy" and seg_prev_target_xy is not None:
            d = float(np.linalg.norm(cur_xy - seg_prev_target_xy))
            if d > target_merge_dist:
                new_segment = True

        if new_segment:
            if seg_start is not None:
                _emit(s - 1)
            seg_start = s
            seg_action = cur_action
            seg_kind = cur_kind
            seg_first_target_tag = cur_tag
            seg_first_target_xy = cur_xy.copy() if cur_xy is not None else None
            seg_prev_target_xy = cur_xy.copy() if cur_xy is not None else None
        else:
            if cur_kind == "xy" and cur_xy is not None:
                seg_prev_target_xy = cur_xy.copy()

    if seg_start is not None:
        _emit(_NUM_2HZ_STEPS - 1)

    return events


def extract_gt_events(
    *,
    replay_path: Path,
    agent_id: np.ndarray,            # [N] int64 (rollout-side unit_tags to evaluate)
    scenario_id: str,
    observer: int,
    target_merge_dist: float = 10.0,
) -> List[dict]:
    """Read a raw replay HDF5, resolve target XY, and emit GT intent events.

    Output rows include `scenario_id`, `observer`, `mode="own"`, `unit_id`
    (SC2 unit_tag), and a per-(unit) `gt_event_id` so candidate rows can
    reference them.
    """
    replay_path = Path(replay_path)
    with h5py.File(replay_path, "r") as f:
        rep = f["unit_data"]["repeated"]
        unit_tag_raw = f["unit_data"]["global"]["unit_tag"][:].astype(np.int64)  # (N_raw,)
        is_alive = rep["is_alive"][:].astype(bool)                                # (T, N_raw)
        coordinate = rep["coordinate"][:].astype(np.float32)                      # (T, N_raw, 3)
        ability_id = rep["ability_id"][:].astype(np.int64)                        # (T, N_raw)
        target_pos = rep["target_pos"][:].astype(np.float32)                      # (T, N_raw, 2)
        target_id = rep["target_id"][:]                                           # (T, N_raw) uint32

    T, N_raw = is_alive.shape

    # --- Resolve target_id -> XY and a stable target unit tag (mirrors
    # sc_dataset.py:101-109; we additionally remember the target's unit_tag).
    has_tid = (target_id != _TARGET_ID_SENTINEL) & (target_id < N_raw)
    tid_safe = np.where(has_tid, target_id, 0).astype(np.intp)
    t_idx = np.arange(T)[:, None]
    has_tid &= is_alive[t_idx, tid_safe]                  # target unit must be alive
    resolved_xy = coordinate[t_idx, tid_safe, :2]         # (T, N_raw, 2)
    fill_mask = has_tid & (np.abs(target_pos).sum(axis=-1) == 0)
    target_pos = target_pos.copy()
    target_pos[fill_mask] = resolved_xy[fill_mask]
    # Stable identifier for the target unit (independent of ever_alive filter).
    target_unit_tag_full = np.where(
        has_tid, unit_tag_raw[tid_safe], np.int64(-1),
    )                                                      # (T, N_raw)

    # --- Apply ever_alive filter and tag-join the rollout's agent_ids.
    keep_idx = apply_ever_alive_filter(is_alive)
    unit_tag_kept = unit_tag_raw[keep_idx]
    is_alive_kept = is_alive[:, keep_idx]
    ability_kept = ability_id[:, keep_idx]
    target_pos_kept = target_pos[:, keep_idx]
    has_tid_kept = has_tid[:, keep_idx]
    target_tag_kept = target_unit_tag_full[:, keep_idx]

    tag_to_idx = {int(t): i for i, t in enumerate(unit_tag_kept)}
    try:
        join_idx = np.array(
            [tag_to_idx[int(t)] for t in agent_id], dtype=np.intp,
        )
    except KeyError as e:
        raise ValueError(
            f"agent_id {e} from rollout not found in replay {replay_path.name} "
            "after ever_alive filter — preprocessing or replay version mismatch."
        ) from e

    # 2Hz future native-frame boundaries.
    native_frames = np.array(
        [_step_2hz_to_native_frame(s) for s in range(_NUM_2HZ_STEPS)], dtype=np.int64,
    )
    if T <= native_frames[-1]:
        raise ValueError(
            f"replay {replay_path.name} only has T={T} frames; expected at least "
            f"{native_frames[-1] + 1} for the 2Hz future window."
        )

    # Slice all per-step arrays at the 2Hz frames AND the joined agent rows.
    # Result shape: [N_join, 16].
    abi = ability_kept[native_frames[:, None], join_idx[None, :]].T               # [N, 16]
    has_tid_2hz = has_tid_kept[native_frames[:, None], join_idx[None, :]].T       # [N, 16]
    target_tag_2hz = target_tag_kept[native_frames[:, None], join_idx[None, :]].T  # [N, 16]
    target_pos_2hz = target_pos_kept[native_frames, :, :][:, join_idx, :]          # [16, N, 2]
    target_pos_2hz = target_pos_2hz.transpose(1, 0, 2)                              # [N, 16, 2]

    is_active = abi != 0
    coarse = vectorized_coarse_action(abi)
    # has_target_xy: target_pos non-zero AND not already covered by has_target_unit.
    target_pos_nonzero = np.abs(target_pos_2hz).sum(axis=-1) > 0
    has_target_xy_only = target_pos_nonzero & (~has_tid_2hz)

    out: List[dict] = []
    for i in range(len(agent_id)):
        events_i = _segment_one_agent(
            steps_native=native_frames,
            coarse_action=coarse[i],
            has_target_unit=has_tid_2hz[i],
            target_unit_tag=target_tag_2hz[i],
            has_target_xy=has_target_xy_only[i],
            target_xy=target_pos_2hz[i],
            is_active=is_active[i],
            target_merge_dist=target_merge_dist,
        )
        for j, ev in enumerate(events_i):
            row = {
                "scenario_id": scenario_id,
                "observer": int(observer),
                "mode": "own",
                "unit_id": int(agent_id[i]),
                "gt_event_id": j,
            }
            row.update(ev)
            out.append(row)
    return out
