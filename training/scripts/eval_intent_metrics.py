"""Stage 2 of agent-oracle intent F1 evaluation.

Reads the four CSVs produced by `compute_intent_candidates.py`, applies a
threshold sweep over `--d`, and emits per-(scenario, observer, coarse_action)
TP/FP/FN counts plus a pooled summary.

Example:
    python scripts/eval_intent_metrics.py \\
      --candidates_dir logs/<run>/intent \\
      --w 16 --d 5,10,20 --require_action_match true
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.starcraft.eval.aggregate import write_csv  # noqa: E402
from src.starcraft.eval.intent.extract_gt import SHIFTED_ACTION_NAMES  # noqa: E402
from src.starcraft.eval.intent.metrics import (  # noqa: E402
    build_wide_table,
    compute_distance_records,
    compute_metric_records,
    pool_distance_records,
    pool_f1_records,
)


_INT_FIELDS = {
    "observer", "unit_id", "rollout_id", "valid_steps",
    "gt_event_id", "pred_event_id",
    "t_2hz_step", "t_native_frame",
    "coarse_action", "target_unit_id",
    "segment_length_steps",
    "delta_t_frames",
    "pred_action", "gt_action",
    "t_gt_native", "t_pred_native",
}
_FLOAT_FIELDS = {"ade", "target_x", "target_y", "delta_xy"}
_BOOL_FIELDS = {
    "has_target", "action_match", "gt_has_target", "pred_has_target",
}


def _coerce_row(row: Mapping[str, str]) -> dict:
    """CSV reader returns strings; cast known numeric/bool fields."""
    out: dict = {}
    for k, v in row.items():
        if v is None or v == "":
            out[k] = v
            continue
        if k in _INT_FIELDS:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = float(v)  # tolerate scientific notation
        elif k in _FLOAT_FIELDS:
            out[k] = float(v)
        elif k in _BOOL_FIELDS:
            out[k] = v.lower() in ("true", "1", "yes")
        else:
            out[k] = v
    return out


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [_coerce_row(r) for r in reader]


def _parse_d_sweep(spec: str) -> List[float]:
    out: List[float] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(float(chunk))
    if not out:
        raise SystemExit("--d must list at least one threshold")
    return out


def _parse_bool(s: str) -> bool:
    return str(s).lower() in ("true", "1", "yes")


def _print_distance_summary(pooled_distance: Sequence[Mapping]) -> None:
    """Pretty-print pooled target-XY error per action (one block, mean+std+n)."""
    by_action: Dict[int, Dict[str, float]] = {}
    for r in pooled_distance:
        a = int(r["coarse_action"])
        if a not in by_action:
            by_action[a] = {}
        m = str(r["metric"])
        # Extract the kind suffix between "_dist_" and "[".
        kind = m.split("_dist_", 1)[1].split("[", 1)[0]
        by_action[a][kind] = float(r["value"])
    if not by_action:
        return
    print("\n=== Pooled target-XY error (cells) on TPs at d=inf, both have target ===")
    print(f"  {'act':<3s}  {'name':<10s}  {'n':>8s}  {'mean':>8s}  {'std':>8s}")
    for a in sorted(by_action):
        s = by_action[a]
        name = SHIFTED_ACTION_NAMES.get(a, str(a))
        n = int(s.get("n", 0))
        mean = s.get("mean", float("nan"))
        std = s.get("std", float("nan"))
        print(f"  {a:<3d}  {name:<10s}  {n:>8d}  {mean:>8.3f}  {std:>8.3f}")


def _print_summary(wide: Sequence[Mapping]) -> None:
    """Pretty-print the wide table: one block per d, action names included."""
    by_d: Dict[float, List[Mapping]] = {}
    for r in wide:
        by_d.setdefault(float(r["d"]), []).append(r)
    if not by_d:
        return
    print("\n=== Pooled agent-oracle intent F1 ===")
    for d in sorted(by_d):
        rows = sorted(by_d[d], key=lambda x: int(x["coarse_action"]))
        print(f"\n  d={d:g}")
        print(f"  {'act':<3s}  {'name':<10s}  {'precision':>10s}  {'recall':>10s}  {'f1':>10s}")
        for r in rows:
            a = int(r["coarse_action"])
            name = str(r["coarse_action_name"])
            p = float(r["precision"])
            rec = float(r["recall"])
            f = float(r["f1"])
            print(f"  {a:<3d}  {name:<10s}  {p:>10.4f}  {rec:>10.4f}  {f:>10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates_dir", type=Path, required=True,
                        help="Stage 1 output dir containing the four CSVs")
    parser.add_argument("--w", type=int, default=16,
                        help="Temporal threshold in native frames (default 16 = 1s @ 16Hz)")
    parser.add_argument("--d", default="5,10,20",
                        help="Comma list of spatial thresholds in cells "
                             "(default sweep: 5,10,20)")
    parser.add_argument("--require_action_match", default="true",
                        help="If true, only candidates with matching coarse "
                             "action are eligible for greedy matching")
    parser.add_argument("--out_csv", type=Path, default=None,
                        help="Per-scenario record CSV (default "
                             "<candidates_dir>/intent_metrics.csv)")
    parser.add_argument("--pooled_csv", type=Path, default=None,
                        help="Pooled summary CSV (default "
                             "<candidates_dir>/intent_metrics_pooled.csv)")
    parser.add_argument("--table_csv", type=Path, default=None,
                        help="Wide-format table CSV (default "
                             "<candidates_dir>/intent_metrics_table.csv) — "
                             "one row per (action, d) with precision/recall/f1 "
                             "and action name columns")
    parser.add_argument("--distance_csv", type=Path, default=None,
                        help="Per-scenario target-XY distance accumulators "
                             "(default <candidates_dir>/intent_distance.csv). "
                             "Computed at d=inf, action_match=True, both sides "
                             "have target.")
    parser.add_argument("--distance_pooled_csv", type=Path, default=None,
                        help="Pooled target-XY distance summary "
                             "(default <candidates_dir>/intent_distance_pooled.csv)")
    args = parser.parse_args()

    cand_dir = args.candidates_dir.resolve()
    if not cand_dir.is_dir():
        raise SystemExit(f"--candidates_dir not a directory: {cand_dir}")

    sel = _read_csv(cand_dir / "selected_rollouts.csv")
    gt = _read_csv(cand_dir / "gt_events.csv")
    pr = _read_csv(cand_dir / "pred_events.csv")
    cd = _read_csv(cand_dir / "intent_candidates.csv")

    if not sel:
        raise SystemExit(
            f"selected_rollouts.csv missing or empty under {cand_dir}; "
            "run compute_intent_candidates.py first."
        )

    scenario_meta: Dict[str, dict] = {}
    for r in sel:
        sid = str(r["scenario_id"])
        if sid not in scenario_meta:
            scenario_meta[sid] = {
                "map_name": str(r.get("map_name", "")),
                "stage": str(r.get("stage", "")),
            }

    d_sweep = _parse_d_sweep(args.d)
    require_action_match = _parse_bool(args.require_action_match)

    print(f"[intent] gt={len(gt)} pred={len(pr)} candidates={len(cd)} "
          f"scenarios={len(scenario_meta)}")
    print(f"[intent] w={args.w} frames, d_sweep={d_sweep}, "
          f"require_action_match={require_action_match}")

    per_scn = compute_metric_records(
        gt_events=gt,
        pred_events=pr,
        candidates=cd,
        scenario_meta=scenario_meta,
        w=args.w,
        d_sweep=d_sweep,
        require_action_match=require_action_match,
    )
    pooled = pool_f1_records(per_scn)
    wide = build_wide_table(per_scn)

    out_csv = args.out_csv or (cand_dir / "intent_metrics.csv")
    pooled_csv = args.pooled_csv or (cand_dir / "intent_metrics_pooled.csv")
    table_csv = args.table_csv or (cand_dir / "intent_metrics_table.csv")
    write_csv(per_scn, out_csv)
    write_csv(pooled, pooled_csv)
    write_csv(wide, table_csv)
    print(f"[intent] wrote {len(per_scn)} per-scenario rows -> {out_csv}")
    print(f"[intent] wrote {len(pooled)} pooled summary rows -> {pooled_csv}")
    print(f"[intent] wrote {len(wide)} wide-table rows -> {table_csv}")

    per_scn_dist = compute_distance_records(
        candidates=cd,
        scenario_meta=scenario_meta,
        w=args.w,
        require_action_match=require_action_match,
    )
    pooled_dist = pool_distance_records(per_scn_dist)
    distance_csv = args.distance_csv or (cand_dir / "intent_distance.csv")
    distance_pooled_csv = (
        args.distance_pooled_csv or (cand_dir / "intent_distance_pooled.csv")
    )
    write_csv(per_scn_dist, distance_csv)
    write_csv(pooled_dist, distance_pooled_csv)
    print(f"[intent] wrote {len(per_scn_dist)} per-scenario distance rows -> {distance_csv}")
    print(f"[intent] wrote {len(pooled_dist)} pooled distance rows -> {distance_pooled_csv}")

    _print_summary(wide)
    _print_distance_summary(pooled_dist)


if __name__ == "__main__":
    main()
