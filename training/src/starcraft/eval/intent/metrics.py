"""Apply temporal/spatial thresholds to threshold-free candidates and
compute per-coarse-action TP / FP / FN counts.

The candidate CSV emitted by `candidates.py` contains every (gt_event,
pred_event) pair within `w_max` native frames. Here we filter to the
matching window `(w, d, require_action_match)` and run greedy 1:1 matching
within each `(scenario, observer, unit_id)` bucket.

Missing-target rule (three-branch, per design):
  * neither side has a target → spatial condition passes
    (action-only events can match action-only events)
  * exactly one side has a target → spatial condition fails
    (action-only and target-bearing events are different command types)
  * both have targets → require `delta_xy <= d`
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from src.starcraft.eval.intent.extract_gt import SHIFTED_ACTION_NAMES


def _bucket_key(row: Mapping) -> Tuple:
    return (row["scenario_id"], int(row["observer"]), int(row["unit_id"]))


def _passes_thresholds(c: Mapping, w: int, d: float, require_action_match: bool) -> bool:
    if abs(int(c["delta_t_frames"])) > w:
        return False
    if require_action_match and not bool(c["action_match"]):
        return False
    g_has = bool(c["gt_has_target"])
    p_has = bool(c["pred_has_target"])
    if g_has and p_has:
        dxy = float(c["delta_xy"])
        if not math.isfinite(dxy) or dxy > d:
            return False
    elif g_has != p_has:
        return False
    # else: neither has target — spatial passes
    return True


def _candidate_sort_key(c: Mapping) -> Tuple:
    """Sort key for greedy matching: prefer smaller |delta_t|, then smaller
    delta_xy (NaN sorts last), then deterministic ids."""
    dt_abs = abs(int(c["delta_t_frames"]))
    dxy = c["delta_xy"]
    dxy_v = float(dxy) if dxy is not None and math.isfinite(float(dxy)) else math.inf
    return (dt_abs, dxy_v, int(c["gt_event_id"]), int(c["pred_event_id"]))


def _greedy_match(
    cands: Sequence[Mapping],
) -> List[Tuple[int, int, int, int]]:
    """Greedy 1:1 match. Returns list of (gt_eid, pred_eid, gt_action, pred_action)."""
    cands = sorted(cands, key=_candidate_sort_key)
    matched_gt: set = set()
    matched_pred: set = set()
    matches: List[Tuple[int, int, int, int]] = []
    for c in cands:
        g_id = int(c["gt_event_id"])
        p_id = int(c["pred_event_id"])
        if g_id in matched_gt or p_id in matched_pred:
            continue
        matched_gt.add(g_id)
        matched_pred.add(p_id)
        matches.append((g_id, p_id, int(c["gt_action"]), int(c["pred_action"])))
    return matches


def _scenario_records_for_d(
    *,
    scenario_id: str,
    map_name: str,
    observer: int,
    stage: str,
    gt_events_bucket: Sequence[Mapping],
    pred_events_bucket: Sequence[Mapping],
    candidates_bucket: Sequence[Mapping],
    w: int,
    d: float,
    require_action_match: bool,
) -> List[dict]:
    """Run matching for a single (scenario, observer) at one threshold d.

    Returns one record per (coarse_action × {tp,fp,fn}) for this scenario.
    """
    # Group by unit_id within this (scenario, observer).
    by_unit_gt: dict = defaultdict(list)
    for ev in gt_events_bucket:
        by_unit_gt[int(ev["unit_id"])].append(ev)
    by_unit_pred: dict = defaultdict(list)
    for ev in pred_events_bucket:
        by_unit_pred[int(ev["unit_id"])].append(ev)
    by_unit_cand: dict = defaultdict(list)
    for c in candidates_bucket:
        by_unit_cand[int(c["unit_id"])].append(c)

    tp_by_action: dict = defaultdict(int)
    fp_by_action: dict = defaultdict(int)
    fn_by_action: dict = defaultdict(int)
    units_by_action: dict = defaultdict(set)   # for n_agents

    all_unit_ids = set(by_unit_gt.keys()) | set(by_unit_pred.keys())
    for unit_id in all_unit_ids:
        gt_evs = by_unit_gt.get(unit_id, [])
        pred_evs = by_unit_pred.get(unit_id, [])
        cands = [
            c for c in by_unit_cand.get(unit_id, [])
            if _passes_thresholds(c, w, d, require_action_match)
        ]
        matches = _greedy_match(cands)
        matched_gt_ids = {m[0] for m in matches}
        matched_pred_ids = {m[1] for m in matches}

        # TPs attributed to GT action (the truth).
        for g_id, p_id, g_act, p_act in matches:
            tp_by_action[g_act] += 1
            units_by_action[g_act].add(unit_id)
        # FNs: unmatched GT events, attributed to GT action.
        for ev in gt_evs:
            if int(ev["gt_event_id"]) not in matched_gt_ids:
                a = int(ev["coarse_action"])
                fn_by_action[a] += 1
                units_by_action[a].add(unit_id)
        # FPs: unmatched pred events, attributed to predicted action.
        for ev in pred_evs:
            if int(ev["pred_event_id"]) not in matched_pred_ids:
                a = int(ev["coarse_action"])
                fp_by_action[a] += 1
                units_by_action[a].add(unit_id)

    actions = set(tp_by_action) | set(fp_by_action) | set(fn_by_action)
    out: List[dict] = []
    base = {
        "scenario_id": scenario_id,
        "map_name": map_name,
        "observer": int(observer),
        "mode": "own",
        "stage": stage,
    }
    d_str = _format_d(d)
    for a in sorted(actions):
        n_agents = len(units_by_action[a])
        for kind, count in (
            ("tp", tp_by_action[a]),
            ("fp", fp_by_action[a]),
            ("fn", fn_by_action[a]),
        ):
            row = dict(base)
            row.update({
                "metric": f"agent_oracle_intent_{kind}[d={d_str},act={a}]",
                "value": int(count),
                "n_agents": int(n_agents),
                "weight": int(count),
                "coarse_action": int(a),
                "d": float(d),
                "kind": kind,
            })
            out.append(row)
    return out


def _format_d(d: float) -> str:
    if float(d).is_integer():
        return str(int(d))
    return f"{d:g}"


def compute_metric_records(
    *,
    gt_events: Iterable[Mapping],
    pred_events: Iterable[Mapping],
    candidates: Iterable[Mapping],
    scenario_meta: Mapping[str, Mapping],   # {scenario_id: {map_name, stage}}
    w: int,
    d_sweep: Sequence[float],
    require_action_match: bool,
) -> List[dict]:
    """Compute per-(scenario, observer, coarse_action, d) TP/FP/FN records.

    `scenario_meta` provides `map_name` and `stage` per scenario_id (read
    once from the rollout files in stage 1 and persisted via `selector` CSV).
    """
    gt_events = list(gt_events)
    pred_events = list(pred_events)
    candidates = list(candidates)

    # Bucket events / candidates by (scenario, observer).
    by_so_gt: dict = defaultdict(list)
    for ev in gt_events:
        by_so_gt[(ev["scenario_id"], int(ev["observer"]))].append(ev)
    by_so_pred: dict = defaultdict(list)
    for ev in pred_events:
        by_so_pred[(ev["scenario_id"], int(ev["observer"]))].append(ev)
    by_so_cand: dict = defaultdict(list)
    for c in candidates:
        by_so_cand[(c["scenario_id"], int(c["observer"]))].append(c)

    out: List[dict] = []
    all_so_keys = set(by_so_gt.keys()) | set(by_so_pred.keys())
    for (scenario_id, observer) in sorted(all_so_keys):
        meta = scenario_meta.get(scenario_id, {})
        map_name = str(meta.get("map_name", ""))
        stage = str(meta.get("stage", ""))
        for d in d_sweep:
            recs = _scenario_records_for_d(
                scenario_id=scenario_id,
                map_name=map_name,
                observer=observer,
                stage=stage,
                gt_events_bucket=by_so_gt.get((scenario_id, observer), []),
                pred_events_bucket=by_so_pred.get((scenario_id, observer), []),
                candidates_bucket=by_so_cand.get((scenario_id, observer), []),
                w=w,
                d=float(d),
                require_action_match=require_action_match,
            )
            out.extend(recs)
    return out


def build_wide_table(records: Iterable[Mapping]) -> List[dict]:
    """Pivot per-scenario TP/FP/FN counts to a wide table.

    One row per (coarse_action × d) with columns:
        coarse_action, coarse_action_name, d, precision, recall, f1, tp, fp, fn

    Useful for spreadsheet / pandas-free inspection — mirrors the table
    `eval_intent_metrics.py` prints to stdout.
    """
    sums: dict = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in records:
        sums[(float(r["d"]), int(r["coarse_action"]))][r["kind"]] += int(r["value"])

    out: List[dict] = []
    for (d, a), counts in sorted(sums.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if (tp + fp) > 0 and (tp + fn) > 0 and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")
        out.append({
            "coarse_action": a,
            "coarse_action_name": SHIFTED_ACTION_NAMES.get(a, str(a)),
            "d": d,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
    return out


def _scenario_distance_records(
    *,
    scenario_id: str,
    map_name: str,
    observer: int,
    stage: str,
    candidates_bucket: Sequence[Mapping],
    w: int,
    require_action_match: bool,
) -> List[dict]:
    """Run greedy matching at d=inf and accumulate per-action target-XY error.

    Distance is only counted on TPs where both sides have a target (the
    "spatial gate passes" branch of `_passes_thresholds`). Each per-action
    record stores `n`, `sum_dxy`, `sum_sq_dxy` so pooling across scenarios
    is additive — mean and stddev fall out of the totals at pool time.
    """
    by_unit_cand: dict = defaultdict(list)
    for c in candidates_bucket:
        by_unit_cand[int(c["unit_id"])].append(c)

    n_by_action: dict = defaultdict(int)
    sum_by_action: dict = defaultdict(float)
    sum_sq_by_action: dict = defaultdict(float)
    units_by_action: dict = defaultdict(set)

    for unit_id, raw_cands in by_unit_cand.items():
        cands = [
            c for c in raw_cands
            if _passes_thresholds(c, w, math.inf, require_action_match)
        ]
        cands.sort(key=_candidate_sort_key)
        matched_gt: set = set()
        matched_pred: set = set()
        for c in cands:
            g_id = int(c["gt_event_id"])
            p_id = int(c["pred_event_id"])
            if g_id in matched_gt or p_id in matched_pred:
                continue
            matched_gt.add(g_id)
            matched_pred.add(p_id)
            if not (bool(c["gt_has_target"]) and bool(c["pred_has_target"])):
                continue
            try:
                dxy = float(c["delta_xy"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(dxy):
                continue
            a = int(c["gt_action"])
            n_by_action[a] += 1
            sum_by_action[a] += dxy
            sum_sq_by_action[a] += dxy * dxy
            units_by_action[a].add(unit_id)

    out: List[dict] = []
    base = {
        "scenario_id": scenario_id,
        "map_name": map_name,
        "observer": int(observer),
        "mode": "own",
        "stage": stage,
    }
    for a in sorted(n_by_action.keys()):
        n = int(n_by_action[a])
        s = float(sum_by_action[a])
        ss = float(sum_sq_by_action[a])
        for kind, val in (("n", n), ("sum_dxy", s), ("sum_sq_dxy", ss)):
            row = dict(base)
            row.update({
                "metric": f"agent_oracle_intent_target_dist_{kind}[act={a}]",
                "value": int(val) if kind == "n" else float(val),
                "n_agents": len(units_by_action[a]),
                "weight": n,
                "coarse_action": int(a),
                "kind": kind,
            })
            out.append(row)
    return out


def compute_distance_records(
    *,
    candidates: Iterable[Mapping],
    scenario_meta: Mapping[str, Mapping],
    w: int,
    require_action_match: bool,
) -> List[dict]:
    """Compute per-(scenario, observer, coarse_action) target-XY distance
    accumulators (`n`, `sum_dxy`, `sum_sq_dxy`) on greedy d=inf TPs.
    """
    candidates = list(candidates)
    by_so_cand: dict = defaultdict(list)
    for c in candidates:
        by_so_cand[(c["scenario_id"], int(c["observer"]))].append(c)

    out: List[dict] = []
    for (scenario_id, observer) in sorted(by_so_cand.keys()):
        meta = scenario_meta.get(scenario_id, {})
        recs = _scenario_distance_records(
            scenario_id=scenario_id,
            map_name=str(meta.get("map_name", "")),
            observer=observer,
            stage=str(meta.get("stage", "")),
            candidates_bucket=by_so_cand[(scenario_id, observer)],
            w=w,
            require_action_match=require_action_match,
        )
        out.extend(recs)
    return out


def pool_distance_records(records: Iterable[Mapping]) -> List[dict]:
    """Sum n / sum_dxy / sum_sq_dxy per coarse_action across scenarios and
    emit pooled `mean` and `std` rows alongside the raw totals.

    Mean = sum_dxy / n; std = sqrt(max(sum_sq_dxy/n - mean^2, 0)).
    """
    sums: dict = defaultdict(lambda: {"n": 0, "sum_dxy": 0.0, "sum_sq_dxy": 0.0})
    n_agents_max: dict = defaultdict(int)
    for r in records:
        a = int(r["coarse_action"])
        kind = r["kind"]
        if kind == "n":
            sums[a]["n"] += int(r["value"])
        else:
            sums[a][kind] += float(r["value"])
        n_agents_max[a] = max(n_agents_max[a], int(r.get("n_agents", 0)))

    out: List[dict] = []
    for a in sorted(sums.keys()):
        s = sums[a]
        n = int(s["n"])
        if n == 0:
            mean = float("nan")
            std = float("nan")
        else:
            mean = s["sum_dxy"] / n
            var = max(s["sum_sq_dxy"] / n - mean * mean, 0.0)
            std = math.sqrt(var)
        base = {
            "scenario_id": "__pooled__",
            "map_name": "",
            "observer": 0,
            "mode": "own",
            "stage": "",
            "coarse_action": int(a),
        }
        for name, val in (
            ("n", n),
            ("sum_dxy", s["sum_dxy"]),
            ("sum_sq_dxy", s["sum_sq_dxy"]),
            ("mean", mean),
            ("std", std),
        ):
            row = dict(base)
            row.update({
                "metric": f"agent_oracle_intent_target_dist_{name}[act={a}]",
                "value": int(val) if name == "n" else float(val),
                "n_agents": int(n_agents_max[a]),
                "weight": n,
            })
            out.append(row)
    return out


def pool_f1_records(records: Iterable[Mapping]) -> List[dict]:
    """Sum TP/FP/FN across scenarios per (coarse_action × d) and compute
    pooled precision / recall / F1.

    Returns one summary record per (metric, d, action) for f1 / precision /
    recall plus the raw pooled tp / fp / fn counts. These records carry
    `scenario_id="__pooled__"` and `mode="own"` so they can be concatenated
    into the headline eval CSV without colliding with per-scenario rows.
    """
    sums: dict = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    n_agents_max: dict = defaultdict(int)
    for r in records:
        a = int(r["coarse_action"])
        d = float(r["d"])
        kind = r["kind"]
        sums[(d, a)][kind] += int(r["value"])
        n_agents_max[(d, a)] = max(n_agents_max[(d, a)], int(r.get("n_agents", 0)))

    out: List[dict] = []
    for (d, a), counts in sorted(sums.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if (precision + recall) > 0 and not (math.isnan(precision) or math.isnan(recall)):
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")
        d_str = _format_d(d)
        base = {
            "scenario_id": "__pooled__",
            "map_name": "",
            "observer": 0,
            "mode": "own",
            "stage": "",
            "coarse_action": a,
            "d": d,
        }
        for name, val in (("tp", tp), ("fp", fp), ("fn", fn)):
            row = dict(base)
            row.update({
                "metric": f"agent_oracle_intent_{name}_pooled[d={d_str},act={a}]",
                "value": int(val),
                "n_agents": int(n_agents_max[(d, a)]),
                "weight": 1,
            })
            out.append(row)
        for name, val in (
            ("precision", precision),
            ("recall", recall),
            ("f1", f1),
        ):
            row = dict(base)
            row.update({
                "metric": f"agent_oracle_intent_{name}[d={d_str},act={a}]",
                "value": float(val),
                "n_agents": int(n_agents_max[(d, a)]),
                "weight": int(tp + fn) if name in ("recall", "f1") else int(tp + fp),
            })
            out.append(row)
    return out
