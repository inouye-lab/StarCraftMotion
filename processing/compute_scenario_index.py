#!/usr/bin/env python
"""Compute per-scenario interaction index for quality-aware sampling.

Scans all HDF5 scenario files in the source dataset directory and computes:
  - mutual_frame_cnt: # frames with any mutually visible player unit
  - mutual_unit_sum:  total (unit, frame) pairs with visible_status == 8
  - transition_cnt:   # (unit, frame) visibility transitions into/out of 8,
                      debounced with a minimum hold of 3 frames

Output is saved as a pickle dict: {replay_hash/scenario.h5 -> metrics_dict}.

Usage:
    python compute_scenario_index.py --input_dir /path/to/StarCraftMotion_scenarios_v2 --workers 32
"""

import argparse
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def compute_scenario_metrics(h5_path):
    """Compute interaction metrics for a single scenario HDF5 file.

    Returns (relative_key, metrics_dict) or (relative_key, None) on failure.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            owners = f["unit_data/global/unit_owner"][:]          # (N,)
            is_alive = f["unit_data/repeated/is_alive"][:]        # (T, N)
            visible_status = f["unit_data/repeated/visible_status"][:]  # (T, N)

        T, N = is_alive.shape
        player_mask = owners != 16                                # (N,)

        # Mutually visible: alive player units with visible_status == 8
        mutual = (visible_status == 8) & is_alive & player_mask[None, :]  # (T, N)

        # mutual_frame_cnt: frames with any mutual visibility
        per_frame = mutual.any(axis=1)                            # (T,)
        mutual_frame_cnt = int(per_frame.sum())

        # mutual_unit_sum: total mutually-visible player-unit-frames
        mutual_unit_sum = int(mutual.sum())

        # transition_cnt: debounced transitions into/out of visible_status==8
        # Only for player units
        player_indices = np.where(player_mask)[0]
        transition_cnt = 0

        if len(player_indices) > 0 and T > 1:
            # mutual status per player unit across time: (T, N_player)
            m_player = mutual[:, player_indices]                  # (T, N_player)

            # Raw transitions: frame-to-frame state changes
            raw_changes = m_player[1:] != m_player[:-1]           # (T-1, N_player)

            # Debounce: a transition only counts if the new state persists
            # for at least 3 consecutive frames
            min_hold = 3
            for u in range(len(player_indices)):
                if not raw_changes[:, u].any():
                    continue
                change_frames = np.where(raw_changes[:, u])[0]
                for cf in change_frames:
                    new_state = m_player[cf + 1, u]
                    # Check if new state holds for min_hold frames
                    end = min(cf + 1 + min_hold, T)
                    if end - (cf + 1) < min_hold:
                        # Not enough frames left to verify hold
                        continue
                    if np.all(m_player[cf + 1:end, u] == new_state):
                        transition_cnt += 1

        return {
            "mutual_frame_cnt": mutual_frame_cnt,
            "mutual_unit_sum": mutual_unit_sum,
            "transition_cnt": transition_cnt,
        }
    except Exception as e:
        return None


def process_replay_dir(args):
    """Process all scenarios in a single replay directory.

    Returns list of (relative_key, metrics_dict) tuples.
    """
    replay_dir, input_dir = args
    replay_hash = replay_dir.name
    results = []
    h5_files = sorted(replay_dir.glob("*.h5"))
    for h5_path in h5_files:
        key = f"{replay_hash}/{h5_path.name}"
        metrics = compute_scenario_metrics(str(h5_path))
        if metrics is not None:
            results.append((key, metrics))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-scenario interaction index.")
    parser.add_argument(
        "--input_dir", required=True,
        help="Source directory containing {replay_hash}/*.h5 files.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output pickle path (default: {input_dir}/scenario_index.pkl).",
    )
    parser.add_argument(
        "--workers", type=int, default=32,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--max_replays", type=int, default=0,
        help="Max replay dirs to process (0=all, for debugging).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = args.output or str(input_dir / "scenario_index.pkl")

    # Discover replay directories
    replay_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if args.max_replays > 0:
        replay_dirs = replay_dirs[:args.max_replays]
    print(f"Found {len(replay_dirs)} replay directories in {input_dir}")

    # Process in parallel (one task per replay dir to reduce pool overhead)
    tasks = [(d, str(input_dir)) for d in replay_dirs]

    t0 = time.time()
    index = {}
    n_scenarios = 0
    n_errors = 0

    with Pool(args.workers) as pool:
        for batch in tqdm(
            pool.imap_unordered(process_replay_dir, tasks, chunksize=1),
            total=len(tasks),
            desc="Indexing",
            unit="replay",
        ):
            for key, metrics in batch:
                index[key] = metrics
                n_scenarios += 1

    elapsed = time.time() - t0

    # --- Summary stats ---
    mutual_counts = np.array([v["mutual_frame_cnt"] for v in index.values()])
    unit_sums = np.array([v["mutual_unit_sum"] for v in index.values()])
    trans_counts = np.array([v["transition_cnt"] for v in index.values()])
    n_with_mutual = int((mutual_counts > 0).sum())

    print(f"\n{'='*60}")
    print(f"Indexed {n_scenarios} scenarios from {len(replay_dirs)} replays "
          f"in {elapsed:.1f}s ({n_scenarios/elapsed:.0f} files/s)")
    print(f"{'='*60}")

    print(f"\n--- Basic stats ---")
    print(f"  Total scenarios:               {n_scenarios:>10,}")
    print(f"  With mutual visibility (>0):   {n_with_mutual:>10,}  "
          f"({100*n_with_mutual/n_scenarios:.1f}%)")
    print(f"  Without mutual visibility (=0):{n_scenarios - n_with_mutual:>10,}  "
          f"({100*(n_scenarios - n_with_mutual)/n_scenarios:.1f}%)")

    print(f"\n--- Distribution across ALL scenarios ---")
    for name, arr in [("mutual_frame_cnt", mutual_counts),
                      ("mutual_unit_sum", unit_sums),
                      ("transition_cnt", trans_counts)]:
        print(f"  {name}:")
        print(f"    mean={arr.mean():.1f}  std={arr.std():.1f}  "
              f"median={np.median(arr):.0f}")
        print(f"    min={arr.min()}  p25={np.percentile(arr, 25):.0f}  "
              f"p75={np.percentile(arr, 75):.0f}  "
              f"p95={np.percentile(arr, 95):.0f}  max={arr.max()}")

    print(f"\n--- Distribution across scenarios WITH mutual visibility ---")
    for name, arr in [("mutual_frame_cnt", mutual_counts[mutual_counts > 0]),
                      ("mutual_unit_sum", unit_sums[mutual_counts > 0]),
                      ("transition_cnt", trans_counts[mutual_counts > 0])]:
        print(f"  {name}:")
        print(f"    mean={arr.mean():.1f}  std={arr.std():.1f}  "
              f"median={np.median(arr):.0f}")
        print(f"    min={arr.min()}  p25={np.percentile(arr, 25):.0f}  "
              f"p75={np.percentile(arr, 75):.0f}  "
              f"p95={np.percentile(arr, 95):.0f}  max={arr.max()}")

    # --- Per-replay analysis (for sample_rate planning) ---
    # Group scenarios by replay hash
    replay_to_scenarios = {}
    for key, metrics in index.items():
        replay_hash = key.split("/")[0]
        replay_to_scenarios.setdefault(replay_hash, []).append(metrics)

    n_replays = len(replay_to_scenarios)
    per_replay_total = []
    per_replay_with_mutual = []
    for replay_hash, scenarios in replay_to_scenarios.items():
        total = len(scenarios)
        with_mutual = sum(1 for s in scenarios if s["mutual_frame_cnt"] > 0)
        per_replay_total.append(total)
        per_replay_with_mutual.append(with_mutual)

    per_replay_total = np.array(per_replay_total)
    per_replay_with_mutual = np.array(per_replay_with_mutual)
    per_replay_ratio = per_replay_with_mutual / np.maximum(per_replay_total, 1)

    print(f"\n--- Per-replay breakdown ({n_replays} replays) ---")
    print(f"  Scenarios per replay:  mean={per_replay_total.mean():.1f}  "
          f"median={np.median(per_replay_total):.0f}  "
          f"min={per_replay_total.min()}  max={per_replay_total.max()}")
    print(f"  With-mutual per replay: mean={per_replay_with_mutual.mean():.1f}  "
          f"median={np.median(per_replay_with_mutual):.0f}")
    print(f"  Ratio with-mutual/total per replay: "
          f"mean={per_replay_ratio.mean():.2f}  "
          f"median={np.median(per_replay_ratio):.2f}")

    # --- Sample rate planning ---
    target_size = 443000  # current dataset size
    print(f"\n--- Sample rate planning (target ~{target_size:,} scenarios) ---")
    total_with_mutual = int(per_replay_with_mutual.sum())
    print(f"  Total scenarios after hard filter: {total_with_mutual:,}")

    if total_with_mutual > 0:
        rate_needed = target_size / total_with_mutual
        print(f"  sample_rate to get ~{target_size:,} from filtered pool: "
              f"{rate_needed:.3f} ({rate_needed*100:.1f}%)")

    # Show what you get at various sample rates from the filtered pool
    print(f"\n  Expected scenario count at various sample_rate "
          f"(from {total_with_mutual:,} filtered scenarios):")
    for rate in [0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 1.00]:
        # Estimate: per replay, sample max(1, filtered_count * rate)
        expected = sum(
            max(1, int(wm * rate)) for wm in per_replay_with_mutual if wm > 0
        )
        print(f"    rate={rate:.2f} -> ~{expected:,} scenarios")

    # Save
    with open(output_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nIndex saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
