#!/usr/bin/env python
"""Create train/val/test split of scenario HDF5 files, organized by map.

Reads the flat layout at INPUT_DIR/{replay_hash}/{scenario}.h5, discovers the
map_name from HDF5 attributes, splits replays per map into train/val/test,
samples a fraction of scenarios per replay, and creates symlinks in the output
directory.
"""

from __future__ import absolute_import, division

import argparse
import logging
import os
import pickle
import random
from pathlib import Path

import h5py
import joblib
import numpy as np
from tqdm import tqdm

logger = logging.getLogger('create_dataset_split')
logger.setLevel(logging.DEBUG)
logger.propagate = False


# ---------------------------------------------------------------------------
# Phase 1: discover replay -> map mapping
# ---------------------------------------------------------------------------

def _discover_replay(replay_dir):
    """Open the first .h5 in a replay directory and return metadata.

    Returns (replay_hash, map_name, num_scenarios) or None on failure.
    """
    replay_hash = replay_dir.name
    h5_files = sorted(replay_dir.glob('*.h5'))
    if not h5_files:
        return None
    try:
        with h5py.File(str(h5_files[0]), 'r') as f:
            map_name = f.attrs.get('map_name', None)
            if map_name is None:
                return None
            if isinstance(map_name, bytes):
                map_name = map_name.decode('utf-8')
    except Exception as e:
        logger.warning(f'Could not read {h5_files[0]}: {e}')
        return None
    return (replay_hash, map_name, len(h5_files))


# ---------------------------------------------------------------------------
# Phase 3: create a single symlink
# ---------------------------------------------------------------------------

def _create_symlink(split, map_name, replay_hash, scenario_name, src_path,
                    output_dir):
    """Create one symlink and return the relative path string."""
    link_name = f'{replay_hash}_{scenario_name}'
    link_dir = output_dir / split / map_name
    link_path = link_dir / link_name
    link_dir.mkdir(parents=True, exist_ok=True)

    src_abs = Path(src_path).resolve()
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(src_abs)

    # Return path relative to output_dir
    return str(Path(split) / map_name / link_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Create train/val/test split with symlinks, organized by map.')
    parser.add_argument(
        '--input_dir', required=True,
        help='Directory containing {replay_hash}/ subdirectories with .h5 files.')
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for symlinked split.')
    parser.add_argument(
        '--maps', nargs='+', default=None,
        help='Map names to include (default: all discovered maps).')
    parser.add_argument(
        '--id_maps', nargs='+', default=None,
        help='In-distribution map names. These get train/val/test; '
             'all other maps get test only. If not set, all maps are ID.')
    parser.add_argument('--sample_rate', type=float, default=0.1,
                        help='Fraction of scenarios to sample per replay (min 1).')
    parser.add_argument('--n_jobs', type=int, default=64,
                        help='Number of parallel joblib workers.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--max_replays', type=int, default=0,
                        help='Max replay dirs to process (0=all, for debugging).')
    parser.add_argument('--adversarial_index', type=str, default=None,
                        help='Path to scenario_index.pkl for adversarial-'
                             'weighted sampling. When provided, scenarios are '
                             'sampled with probability proportional to '
                             'log(1+mutual_unit_sum)+log(1+transition_cnt), '
                             'and zero-interaction scenarios are excluded.')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    args = parser.parse_args()

    # Logging
    console_h = logging.StreamHandler()
    console_h.setLevel(logging.WARNING)
    console_h.setFormatter(logging.Formatter(
        '%(levelname)s - %(asctime)s - %(message)s', datefmt='%m-%d %H:%M'))
    logger.addHandler(console_h)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate sample_rate
    if not 0 < args.sample_rate <= 1:
        print(f'Error: --sample_rate must be in (0, 1], got {args.sample_rate}')
        return

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logger.error(f'Ratios must sum to 1.0, got {total_ratio}')
        return

    # Load adversarial index if provided
    adv_index = None
    if args.adversarial_index:
        print(f'Loading adversarial index from {args.adversarial_index}...')
        with open(args.adversarial_index, 'rb') as f:
            adv_index = pickle.load(f)
        print(f'  Loaded {len(adv_index):,} scenario entries.')

    # -----------------------------------------------------------------------
    # Phase 1: Discover replay-to-map mapping
    # -----------------------------------------------------------------------
    replay_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if args.max_replays > 0:
        replay_dirs = replay_dirs[:args.max_replays]
    print(f'Processing {len(replay_dirs)} replay directories.')

    results = joblib.Parallel(n_jobs=args.n_jobs, backend='threading')(
        joblib.delayed(_discover_replay)(d)
        for d in tqdm(replay_dirs, desc='Discovering', unit='replay')
    )

    # Build map -> list of (replay_hash, num_scenarios)
    map_to_replays = {}
    skipped_none = 0
    skipped_k = 0
    for r in results:
        if r is None:
            skipped_none += 1
            continue
        replay_hash, map_name, num_scenarios = r
        if num_scenarios == 0:
            skipped_k += 1
            continue
        map_to_replays.setdefault(map_name, []).append(
            (replay_hash, num_scenarios))

    print(f'Skipped {skipped_none} unreadable replays, '
          f'{skipped_k} replays with 0 scenarios.')
    print(f'Discovered {len(map_to_replays)} maps: '
          f'{sorted(map_to_replays.keys())}')

    # Filter maps if --maps specified
    if args.maps is not None:
        map_to_replays = {m: v for m, v in map_to_replays.items()
                          if m in args.maps}
        print(f'After --maps filter: {sorted(map_to_replays.keys())}')

    # -----------------------------------------------------------------------
    # Phase 2: Split & sample
    # -----------------------------------------------------------------------
    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)
    symlink_tasks = []  # (split, map_name, replay_hash, scenario_name, src)
    n_skipped_replays = 0

    for map_name, replays in sorted(map_to_replays.items()):
        rng_map = random.Random(rng.randint(0, 2**32 - 1))
        rng_map.shuffle(replays)

        n = len(replays)

        if args.id_maps is not None and map_name not in args.id_maps:
            # OOD map: test only, using test_ratio portion of replays
            n_test = max(1, round(n * args.test_ratio))
            splits = {'test': replays[:n_test]}
            print(f'Map {map_name} (OOD): {n} replays -> test={n_test}')
        else:
            # ID map (or no --id_maps specified): full train/val/test split
            n_val = round(n * args.val_ratio)
            n_test = round(n * args.test_ratio)
            n_train = n - n_val - n_test
            splits = {
                'train': replays[:n_train],
                'val': replays[n_train:n_train + n_val],
                'test': replays[n_train + n_val:],
            }
            tag = ' (ID)' if args.id_maps else ''
            print(f'Map {map_name}{tag}: {n} replays -> '
                  f'train={n_train}, val={n_val}, test={n_test}')

        for split_name, split_replays in splits.items():
            for replay_hash, num_scenarios in split_replays:
                all_scenarios = sorted(
                    f.name for f in
                    (input_dir / replay_hash).glob('*.h5'))
                k = max(1, int(len(all_scenarios) * args.sample_rate))

                if adv_index is not None:
                    # Adversarial-weighted sampling
                    candidates = []
                    scores = []
                    for s in all_scenarios:
                        entry = adv_index.get(f'{replay_hash}/{s}')
                        if entry is None:
                            continue
                        sc = (np.log1p(entry['mutual_unit_sum'])
                              + np.log1p(entry['transition_cnt']))
                        if sc > 0:
                            candidates.append(s)
                            scores.append(sc)
                    if not candidates:
                        n_skipped_replays += 1
                        continue
                    k = min(k, len(candidates))
                    probs = np.array(scores)
                    probs /= probs.sum()
                    np_rng_map = np.random.RandomState(
                        rng_map.randint(0, 2**32 - 1))
                    chosen = np_rng_map.choice(
                        len(candidates), size=k, replace=False, p=probs)
                    sampled = [candidates[i] for i in chosen]
                else:
                    # Uniform random sampling (original behavior)
                    sampled = rng_map.sample(all_scenarios, k=k)

                for scenario_name in sampled:
                    src = input_dir / replay_hash / scenario_name
                    symlink_tasks.append((
                        split_name, map_name, replay_hash,
                        scenario_name, str(src)))

    print(f'Total symlinks to create: {len(symlink_tasks)}')
    if adv_index is not None:
        print(f'Replays skipped (no adversarial candidates): {n_skipped_replays}')

    # -----------------------------------------------------------------------
    # Phase 3: Create symlinks & manifests
    # -----------------------------------------------------------------------
    rel_paths = joblib.Parallel(n_jobs=args.n_jobs, backend='threading')(
        joblib.delayed(_create_symlink)(
            split, map_name, replay_hash, scenario_name, src, output_dir)
        for split, map_name, replay_hash, scenario_name, src
        in tqdm(symlink_tasks, desc='Creating symlinks', unit='link')
    )

    # Write manifests
    manifest_dir = output_dir / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {'train': [], 'val': [], 'test': []}
    for task, rel_path in zip(symlink_tasks, rel_paths):
        split_paths[task[0]].append(rel_path)

    for split_name, paths in split_paths.items():
        paths.sort()
        manifest_file = manifest_dir / f'{split_name}_manifest.txt'
        with open(manifest_file, 'w') as f:
            f.write('\n'.join(paths) + '\n')
        print(f'{split_name}: {len(paths)} entries -> {manifest_file}')

    print('Done.')


if __name__ == '__main__':
    main()
