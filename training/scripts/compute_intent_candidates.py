"""Stage 1 of agent-oracle intent F1 evaluation.

Walks `*.h5` rollout files under `--rollouts_dir`, selects the best-ADE
rollout per agent (matching `min_ade.py`'s argmin convention), extracts
GT and predicted intent events at the 16 future 2Hz boundaries, segments
runs of identical command state into discrete events, and emits
threshold-free candidate (gt, pred) pairs within `--w_max` native frames.

Outputs four CSVs under `--out_dir`:
    selected_rollouts.csv  (scenario, observer, agent, rollout, ADE)
    gt_events.csv          (segmented GT intent events)
    pred_events.csv        (segmented predicted intent events on selected rollout)
    intent_candidates.csv  (gt × pred pairs within w_max frames)

Stage 2 (`eval_intent_metrics.py`) consumes these and applies (w, d,
require_action_match) thresholds to compute precision / recall / F1 per
coarse action class.

Mode is fixed to `own` because aux predictions are saved only under
`/obs_p{1,2}/own/aux/` in the rollout HDF5 schema.

Example:
    python scripts/compute_intent_candidates.py \\
      --rollouts_dir logs/<run>/rollouts \\
      --replays_dir  /scratch/.../StarCraftMotion_split_v2_adversarial \\
      --observers 1,2 \\
      --out_dir logs/<run>/intent
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.starcraft.eval.aggregate import write_csv  # noqa: E402
from src.starcraft.eval.intent.candidates import generate_candidates  # noqa: E402
from src.starcraft.eval.intent.extract_gt import extract_gt_events  # noqa: E402
from src.starcraft.eval.intent.extract_pred import extract_pred_events  # noqa: E402
from src.starcraft.eval.intent.selector import select_best_ade_per_agent  # noqa: E402
from src.starcraft.eval.load_rollout import load_rollout  # noqa: E402


def _load_manifest_index(replays_dir: Path) -> Dict[str, str]:
    """Build {stem: map_name} from <replays_dir>/manifests/*.txt.

    Lines look like `test/Abyssal_Reef_LE/<stem>.h5`. Mirrors the helper in
    `scripts/eval_sc_rollouts.py` so the same `--map_names` filter works for
    intent eval.
    """
    manifests_dir = replays_dir / "manifests"
    files = sorted(manifests_dir.glob("*_manifest.txt")) if manifests_dir.is_dir() else []
    if not files:
        raise SystemExit(
            f"No manifest files under {manifests_dir}. Expected "
            f"{{train,val,test}}_manifest.txt with `<split>/<map>/<stem>.h5` lines."
        )
    stem_to_map: Dict[str, str] = {}
    for mf in files:
        for line in mf.read_text().splitlines():
            parts = line.strip().split("/")
            if len(parts) < 3 or not parts[2].endswith(".h5"):
                continue
            stem_to_map[parts[2][:-3]] = parts[1]
    return stem_to_map


def _find_replay_path(
    replays_dir: Path, map_name: str, scenario_id: str, stage: str,
) -> Optional[Path]:
    """Mirror the search order in `load_rollout.py` to locate the raw replay."""
    candidates: List[Path] = []
    if stage in ("val", "test"):
        candidates.append(replays_dir / stage / map_name / f"{scenario_id}.h5")
    candidates.extend(
        replays_dir / s / map_name / f"{scenario_id}.h5"
        for s in ("test", "val", "train")
        if not candidates or candidates[0].parent.parent.name != s
    )
    return next((p for p in candidates if p.exists()), None)


def _process_scenario(
    rollout_path: Path,
    replays_dir: Path,
    observers: List[int],
    w_max: int,
    target_merge_dist: float,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    sel_rows: List[dict] = []
    gt_rows: List[dict] = []
    pred_rows: List[dict] = []
    cand_rows: List[dict] = []

    for obs in observers:
        try:
            scenario = load_rollout(rollout_path, replays_dir, obs, "own")
        except KeyError:
            continue
        if scenario is None:
            continue
        if scenario.aux is None:
            # No aux predictions saved for this rollout — model wasn't trained
            # with use_aux_loss. Skip silently.
            continue

        sel = select_best_ade_per_agent(
            pred_traj=scenario.pred_traj,
            gt_traj=scenario.gt_traj,
            gt_valid=scenario.gt_valid,
        )

        replay_path = _find_replay_path(
            replays_dir,
            scenario.map_name,
            scenario.scenario_id,
            scenario.stage,
        )
        if replay_path is None:
            raise FileNotFoundError(
                f"Replay file not found for {scenario.scenario_id} "
                f"(map {scenario.map_name})."
            )

        # Selected-rollout CSV: one row per agent (skip agents with no valid GT).
        for i in range(len(scenario.agent_id)):
            if not bool(sel.has_any_valid[i]):
                continue
            sel_rows.append({
                "scenario_id": scenario.scenario_id,
                "map_name": scenario.map_name,
                "observer": int(obs),
                "mode": "own",
                "stage": scenario.stage,
                "unit_id": int(scenario.agent_id[i]),
                "rollout_id": int(sel.selected_rollout[i]),
                "ade": float(sel.ade[i]),
                "valid_steps": int(sel.valid_steps[i]),
            })

        gt_evs = extract_gt_events(
            replay_path=replay_path,
            agent_id=scenario.agent_id[sel.has_any_valid],
            scenario_id=scenario.scenario_id,
            observer=obs,
            target_merge_dist=target_merge_dist,
        )
        pred_evs = extract_pred_events(
            aux=scenario.aux,
            pred_traj=scenario.pred_traj,
            pred_head=scenario.pred_head,
            agent_id=scenario.agent_id,
            selected_rollout=sel.selected_rollout,
            has_any_valid=sel.has_any_valid,
            scenario_id=scenario.scenario_id,
            observer=obs,
            target_merge_dist=target_merge_dist,
        )
        gt_rows.extend(gt_evs)
        pred_rows.extend(pred_evs)
        cand_rows.extend(generate_candidates(
            gt_events=gt_evs,
            pred_events=pred_evs,
            w_max=w_max,
        ))
    return sel_rows, gt_rows, pred_rows, cand_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts_dir", type=Path, required=True)
    parser.add_argument("--replays_dir", type=Path, required=True)
    parser.add_argument("--observers", default="1,2",
                        help="Comma list of observers (1,2)")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Output dir for the four stage-1 CSVs")
    parser.add_argument("--w_max", type=int, default=48,
                        help="Max |delta_t_native_frames| for candidate pairs "
                             "(default 48 = 3s @ 16Hz)")
    parser.add_argument("--target_merge_dist", type=float, default=10.0,
                        help="Cells: max chain-XY drift to keep two adjacent "
                             "2Hz steps in the same segment (default 10)")
    parser.add_argument("--map_names", default="",
                        help="Comma list of map_names to keep (e.g., "
                             "Abyssal_Reef_LE,Acolyte_LE). Empty = all maps. "
                             "Filter requires manifests under <replays_dir>/manifests/.")
    parser.add_argument("--max_scenarios", type=int, default=0,
                        help="Cap scenario count for a quick first pass "
                             "(0 = no cap)")
    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(32, (os.cpu_count() or 2) // 2)))
    args = parser.parse_args()

    rollouts_dir = args.rollouts_dir.resolve()
    replays_dir = args.replays_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not rollouts_dir.is_dir():
        raise SystemExit(f"--rollouts_dir not a directory: {rollouts_dir}")
    if not replays_dir.is_dir():
        raise SystemExit(f"--replays_dir not a directory: {replays_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    observers = [int(x) for x in args.observers.split(",") if x.strip()]
    rollout_files = sorted(rollouts_dir.glob("*.h5"))
    if not rollout_files:
        raise SystemExit(f"No .h5 rollout files found under {rollouts_dir}")

    if args.map_names:
        allow = {m.strip() for m in args.map_names.split(",") if m.strip()}
        stem_to_map = _load_manifest_index(replays_dir)
        missing = [p.stem for p in rollout_files if p.stem not in stem_to_map]
        if missing:
            raise SystemExit(
                f"{len(missing)} rollout stems not found in "
                f"{replays_dir}/manifests/*.txt (e.g., {missing[0]})."
            )
        kept = [p for p in rollout_files if stem_to_map[p.stem] in allow]
        print(f"[intent] map_names filter {sorted(allow)}: kept "
              f"{len(kept)} of {len(rollout_files)} files")
        rollout_files = kept
        if not rollout_files:
            raise SystemExit("No rollout files matched --map_names filter")

    if args.max_scenarios and len(rollout_files) > args.max_scenarios:
        rollout_files = rollout_files[: args.max_scenarios]

    print(f"[intent] {len(rollout_files)} scenarios, observers={observers}, "
          f"workers={args.n_workers}, w_max={args.w_max}, "
          f"target_merge_dist={args.target_merge_dist}")

    process_fn = partial(
        _process_scenario,
        replays_dir=replays_dir,
        observers=observers,
        w_max=args.w_max,
        target_merge_dist=args.target_merge_dist,
    )

    sel_all: List[dict] = []
    gt_all: List[dict] = []
    pred_all: List[dict] = []
    cand_all: List[dict] = []

    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            for sel, g, p, c in tqdm(
                pool.imap_unordered(process_fn, rollout_files, chunksize=4),
                total=len(rollout_files),
                desc="intent stage1",
                unit="scn",
            ):
                sel_all.extend(sel)
                gt_all.extend(g)
                pred_all.extend(p)
                cand_all.extend(c)
    else:
        for path in tqdm(rollout_files, desc="intent stage1", unit="scn"):
            sel, g, p, c = process_fn(path)
            sel_all.extend(sel)
            gt_all.extend(g)
            pred_all.extend(p)
            cand_all.extend(c)

    write_csv(sel_all, out_dir / "selected_rollouts.csv")
    write_csv(gt_all, out_dir / "gt_events.csv")
    write_csv(pred_all, out_dir / "pred_events.csv")
    write_csv(cand_all, out_dir / "intent_candidates.csv")

    print(f"[intent] selected_rollouts: {len(sel_all)} rows")
    print(f"[intent] gt_events:        {len(gt_all)} rows")
    print(f"[intent] pred_events:      {len(pred_all)} rows")
    print(f"[intent] candidates:       {len(cand_all)} rows")
    print(f"[intent] wrote stage-1 CSVs to {out_dir}")


if __name__ == "__main__":
    main()
