#!/usr/bin/python
"""Split HDF5 replay files into fixed-length temporal chunks.

Follows the Waymo Motion format: 1s past + 1 current frame + 8s future.
Default: chunk_size=145 (at 16fps: 16 past + 1 current + 128 future),
         current_frame_idx=fps=16 (0-indexed position of "now" within chunk).
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import json
import logging
import math
import os
import tempfile
from pathlib import Path

import h5py
import joblib
import numpy as np
from tqdm import tqdm

logger = logging.getLogger('split_scenarios')
logger.setLevel(logging.DEBUG)
logger.propagate = False

_TARGET_NONE = np.uint32(0xFFFFFFFF)

# All repeated datasets: (name, ndim)  — ndim=2 means (T,N), ndim=3 means (T,N,d)
_REPEATED_2D = [
    'is_alive', 'is_flying', 'is_burrowed', 'is_carried',
    'unit_type', 'ability_id', 'target_id',
    'radius', 'heading',
    'health', 'health_max', 'shield',
    'energy', 'build_progress',
    'mineral_contents', 'vespene_contents',
    'visible_status',
]
_REPEATED_3D = ['coordinate', 'target_pos']


def _split_replay(input_path, output_dir, chunk_size, stride, fps,
                  map_frame_skip, skip_incomplete, min_duration, min_apm):
    """Split a single HDF5 replay file into chunk files.

    Returns (replay_hash, drop_reason) where drop_reason is None if
    the replay was processed, or a string describing why it was dropped.
    """
    replay_hash = Path(input_path).stem
    out_dir = Path(output_dir) / replay_hash

    with h5py.File(input_path, 'r') as src:
        total_frames = int(src.attrs['total_frames'])
        N = int(src['unit_data/global/unit_tag'].shape[0])

        # --- Quality filters ---
        ri = json.loads(src.attrs.get('replay_info', '{}'))
        dur = ri.get('game_duration_seconds', 0)
        if dur < min_duration:
            return (replay_hash, f'duration={dur:.0f}s < {min_duration}s')

        pi = json.loads(src.attrs.get('player_info', '[]'))
        for p in pi:
            if p.get('player_apm', 0) < min_apm:
                return (replay_hash, f'player {p.get("player_id","?")} APM={p.get("player_apm",0)} < {min_apm}')

        # Compute number of chunks
        if total_frames < chunk_size:
            logger.warning(f'{replay_hash}: total_frames={total_frames} < chunk_size={chunk_size}, skipping')
            return (replay_hash, f'total_frames={total_frames} < chunk_size={chunk_size}')

        out_dir.mkdir(parents=True, exist_ok=True)

        if skip_incomplete:
            n_chunks = (total_frames - chunk_size) // stride + 1
        else:
            n_chunks = math.ceil((total_frames - chunk_size) / stride) + 1

        logger.info(f'{replay_hash}: total_frames={total_frames}, N={N}, n_chunks={n_chunks}')

        # Pre-load global unit arrays (small, N-dim)
        unit_tag_all = src['unit_data/global/unit_tag'][:]
        unit_owner_all = src['unit_data/global/unit_owner'][:]

        # Pre-load is_alive for alive_mask computation (T, N) — needed for all chunks
        is_alive_all = src['unit_data/repeated/is_alive'][:]

        output_paths = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * stride
            end = start + chunk_size

            if end > total_frames:
                if skip_incomplete:
                    break
                end = total_frames

            # Unit filter: keep units alive at any point in this chunk
            alive_chunk = is_alive_all[start:end, :]  # (chunk_T, N)
            alive_mask = alive_chunk.any(axis=0)       # (N,)
            n_alive = int(alive_mask.sum())

            # Map frame slice
            map_start = start // map_frame_skip
            map_end = (end - 1) // map_frame_skip + 1

            # Build old_to_new remap for target_id
            old_to_new = np.full(N, _TARGET_NONE, dtype=np.uint32)
            new_indices = np.where(alive_mask)[0]
            old_to_new[new_indices] = np.arange(n_alive, dtype=np.uint32)

            out_name = f'{chunk_idx:04d}.h5'
            out_path = out_dir / out_name

            if out_path.exists():
                logger.info(f'Skipping {out_name} (already exists)')
                output_paths.append(str(out_path))
                continue

            fd, tmp_path = tempfile.mkstemp(dir=str(out_dir), suffix='.h5.tmp')
            os.close(fd)
            try:
                _write_chunk(
                    src=src,
                    tmp_path=tmp_path,
                    chunk_idx=chunk_idx,
                    start=start,
                    end=end,
                    map_start=map_start,
                    map_end=map_end,
                    alive_mask=alive_mask,
                    n_alive=n_alive,
                    old_to_new=old_to_new,
                    unit_tag_all=unit_tag_all,
                    unit_owner_all=unit_owner_all,
                    fps=fps,
                    map_frame_skip=map_frame_skip,
                    replay_hash=replay_hash,
                )
                os.rename(tmp_path, str(out_path))
                output_paths.append(str(out_path))
                logger.debug(f'Wrote {out_name}: frames [{start},{end}), n_alive={n_alive}')
            except Exception as e:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                logger.error(f'Error writing {out_name}: {e}', exc_info=True)

        return (replay_hash, None)


def _write_chunk(src, tmp_path, chunk_idx, start, end,
                 map_start, map_end,
                 alive_mask, n_alive, old_to_new,
                 unit_tag_all, unit_owner_all,
                 fps, map_frame_skip, replay_hash):
    """Write a single chunk HDF5 file."""
    chunk_T = end - start
    assert fps < chunk_T, f'current_frame_idx={fps} out of bounds for chunk_T={chunk_T}'

    with h5py.File(tmp_path, 'w') as dst:
        # --- Root attributes: copy all source attrs then override ---
        for key, val in src.attrs.items():
            dst.attrs[key] = val

        dst.attrs['total_frames'] = chunk_T
        dst.attrs['max_unique_units'] = n_alive
        dst.attrs['current_frame_idx'] = fps
        dst.attrs['chunk_start_frame'] = start
        dst.attrs['chunk_end_frame'] = end
        dst.attrs['chunk_idx'] = chunk_idx
        dst.attrs['map_frame_start'] = map_start
        dst.attrs['map_frame_skip'] = map_frame_skip
        dst.attrs['parent_replay'] = replay_hash

        comp = dict(compression='gzip', compression_opts=4)

        # --- Player data (T, 2) ---
        pg = dst.create_group('player_data')
        for name in ('minerals', 'vespene', 'food_used', 'food_cap'):
            data = src[f'player_data/{name}'][start:end]
            p_chunks = (min(chunk_T, 256), 2)
            pg.create_dataset(name, data=data, chunks=p_chunks, **comp)

        # --- Dynamic map data (T_map, H, W) ---
        mg = dst.create_group('map_data/dynamic')
        for name in ('creep', 'fow_player_1', 'fow_player_2'):
            data = src[f'map_data/dynamic/{name}'][map_start:map_end]
            T_map, H, W = data.shape
            m_chunks = (1, H, W)
            mg.create_dataset(name, data=data, chunks=m_chunks, **comp)

        # --- Unit data: global (n_alive,) ---
        ug = dst.create_group('unit_data/global')
        ug.create_dataset('unit_tag', data=unit_tag_all[alive_mask], **comp)
        ug.create_dataset('unit_owner', data=unit_owner_all[alive_mask], **comp)

        # --- Unit data: repeated ---
        ur = dst.create_group('unit_data/repeated')

        def c2d():
            return (min(chunk_T, 64), min(n_alive, 128))

        def c3d(d):
            return (min(chunk_T, 64), min(n_alive, 128), d)

        for name in _REPEATED_2D:
            raw = src[f'unit_data/repeated/{name}'][start:end, :]  # (chunk_T, N)
            data = raw[:, alive_mask]  # (chunk_T, n_alive)

            if name == 'target_id':
                # Remap row indices; values >= N or == sentinel stay sentinel
                valid = (data != _TARGET_NONE) & (data < len(old_to_new))
                remapped = np.full_like(data, _TARGET_NONE)
                remapped[valid] = old_to_new[data[valid]]
                data = remapped

            ur.create_dataset(name, data=data, chunks=c2d(), **comp)

        for name in _REPEATED_3D:
            raw = src[f'unit_data/repeated/{name}'][start:end, :]  # (chunk_T, N, d)
            data = raw[:, alive_mask, :]  # (chunk_T, n_alive, d)
            d = data.shape[2]
            ur.create_dataset(name, data=data, chunks=c3d(d), **comp)


def _safe_split(input_path, output_dir, chunk_size, stride, fps,
                map_frame_skip, skip_incomplete, min_duration, min_apm):
    """Top-level wrapper for joblib (must be picklable)."""
    try:
        return _split_replay(
            input_path=input_path,
            output_dir=output_dir,
            chunk_size=chunk_size,
            stride=stride,
            fps=fps,
            map_frame_skip=map_frame_skip,
            skip_incomplete=skip_incomplete,
            min_duration=min_duration,
            min_apm=min_apm,
        )
    except Exception as e:
        replay_hash = Path(input_path).stem
        logger.error(f'Failed on {input_path}: {e}', exc_info=True)
        return (replay_hash, f'exception: {e}')


def main():
    parser = argparse.ArgumentParser(description='Split HDF5 replays into fixed-length chunks.')
    parser.add_argument('--input_dir', required=True,
                        help='Directory of source .h5 replay files.')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for chunk .h5 files.')
    parser.add_argument('--fps', type=int, default=16,
                        help='Frames per second (sets current_frame_idx = fps).')
    parser.add_argument('--chunk_size', type=int, default=145,
                        help='Number of frames per chunk (default: 145 = (8+1)*fps+1).')
    parser.add_argument('--stride', type=int, default=144,
                        help='Stride between chunk starts (default: 144, a multiple of map_frame_skip for map/unit frame alignment; 1-frame overlap with chunk_size=145).')
    parser.add_argument('--map_frame_skip', type=int, default=16,
                        help='Map frames per unit frame (used to slice map_data).')
    parser.add_argument('--num_parallel_jobs', type=int, default=64,
                        help='Number of parallel joblib workers.')
    parser.add_argument('--min_duration', type=float, default=120,
                        help='Minimum game duration in seconds (default: 120).')
    parser.add_argument('--min_apm', type=int, default=1,
                        help='Minimum APM for both players (default: 1).')
    args = parser.parse_args()

    # Logging setup
    console_h = logging.StreamHandler()
    console_h.setLevel(logging.WARNING)
    console_h.setFormatter(logging.Formatter(
        '%(levelname)s - %(asctime)s - %(message)s', datefmt='%m-%d %H:%M'))
    logger.addHandler(console_h)

    input_dir = Path(args.input_dir)
    h5_files = sorted(str(p) for p in input_dir.glob('*.h5'))
    if not h5_files:
        logger.error(f'No .h5 files found in {input_dir}')
        return

    print(f'Found {len(h5_files)} replay files to split.')

    kwargs = dict(
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        stride=args.stride,
        fps=args.fps,
        map_frame_skip=args.map_frame_skip,
        skip_incomplete=True, # Skip the last partial chunk if total_frames % stride != 0
        min_duration=args.min_duration,
        min_apm=args.min_apm,
    )

    if args.num_parallel_jobs == 1:
        results = [_safe_split(f, **kwargs) for f in tqdm(h5_files, desc='Splitting', unit='replay')]
    else:
        results = joblib.Parallel(n_jobs=args.num_parallel_jobs, backend='multiprocessing')(
            joblib.delayed(_safe_split)(f, **kwargs)
            for f in tqdm(h5_files, desc='Splitting', unit='replay')
        )

    # Summary
    n_processed = sum(1 for _, reason in results if reason is None)
    n_dropped = len(results) - n_processed

    print(f'Split complete. {n_processed} processed, {n_dropped} dropped.')


if __name__ == '__main__':
    main()
