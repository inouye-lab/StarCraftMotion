#!/usr/bin/python
"""Build a discrete motion token vocabulary for SC2 units.

Inspired by NVIDIA's SMART/CATK method (arXiv 2405.15677, 2412.05334):
1. Extract short trajectory segments from HDF5 scenarios (1-frame boundary overlap, stride=segment_length-1=8, segment_length=9=0.5s@16fps)
2. Normalize to agent-local coordinates (translate to origin, rotate heading → +x)
3. Compute circular bounding contours using unit radius
4. Cluster with K-disk algorithm using contour distance (matching CATK)
5. Visualize as arrow fan and save dictionary as pickle

Usage:
  python build_motion_dictionary.py --input_dir split_scenarios_out/
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_scenario_motion(path: str) -> dict:
    """Load arrays needed for motion extraction from a scenario HDF5 file."""
    with h5py.File(path, "r") as f:
        data = {
            "coord": f["unit_data/repeated/coordinate"][:, :, :2].astype(np.float32),
            "heading": f["unit_data/repeated/heading"][:].astype(np.float32),
            "radius": f["unit_data/repeated/radius"][:].astype(np.float32),
            "alive": f["unit_data/repeated/is_alive"][:],
            "owners": f["unit_data/global/unit_owner"][:],
            "unit_types": f["unit_data/repeated/unit_type"][:],
        }
    return data


def extract_segments(data, segment_length, segment_stride, min_alive_frames=48, with_metadata=False):
    """Extract trajectory segments from a single scenario.

    Args:
        with_metadata: if True, also return unit_indices, start_frames, unit_type_ids

    Returns:
        coords: (M, seg_len, 2) raw XY positions
        headings: (M, seg_len) raw heading angles
        [if with_metadata:]
        unit_indices: (M,) which unit index each segment belongs to
        start_frames: (M,) the starting frame of each segment
        unit_type_ids: (M,) SC2 unit type ID at segment start frame
    """
    coord, heading, alive, owners = (
        data["coord"], data["heading"], data["alive"], data["owners"]
    )
    T, N = alive.shape

    # Non-neutral units
    player_units = np.where(owners != 16)[0]

    seg_coords, seg_headings = [], []
    if with_metadata:
        unit_types = data["unit_types"]
        seg_unit_indices, seg_start_frames, seg_unit_type_ids = [], [], []

    for u in player_units:
        # Skip short-lived units (fewer than min_alive_frames total alive frames).
        # Matches CATK: agents with < 30 valid frames skipped (30 frames @ 10Hz = 3s).
        # Here: 48 frames @ 16fps = 3s.
        if alive[:, u].sum() < min_alive_frames:
            continue
        for t_start in range(0, T - segment_length + 1, segment_stride):
            t_end = t_start + segment_length
            # Both endpoints must be alive (matching CATK valid_mask check)
            if not (alive[t_start, u] and alive[t_end - 1, u]):
                continue
            seg_coords.append(coord[t_start:t_end, u, :])
            seg_headings.append(heading[t_start:t_end, u])
            if with_metadata:
                seg_unit_indices.append(u)
                seg_start_frames.append(t_start)
                seg_unit_type_ids.append(int(unit_types[t_start, u]))

    if not seg_coords:
        return (None,) * (5 if with_metadata else 2)

    out = (np.array(seg_coords), np.array(seg_headings))
    if with_metadata:
        return out + (np.array(seg_unit_indices), np.array(seg_start_frames), np.array(seg_unit_type_ids))
    return out


def normalize_segments(coords, headings):
    """Normalize segments to agent-local coordinates.

    Translates to origin, rotates so initial heading aligns with +x.

    Returns:
        rel_xy: (M, seg_len, 2)
        rel_heading: (M, seg_len)
    """
    # Translate: relative to first position
    rel_xy = coords - coords[:, 0:1, :]  # (M, seg_len, 2)

    # Rotate by -heading[0] so initial heading = +x
    theta = -headings[:, 0]  # (M,)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x = rel_xy[:, :, 0]
    y = rel_xy[:, :, 1]
    rot_x = cos_t[:, None] * x - sin_t[:, None] * y
    rot_y = sin_t[:, None] * x + cos_t[:, None] * y
    rel_xy = np.stack([rot_x, rot_y], axis=-1)

    # Relative heading
    diff = headings - headings[:, 0:1]
    rel_heading = np.arctan2(np.sin(diff), np.cos(diff))

    return rel_xy, rel_heading


def compute_contours(pos_heading, radius=0.5):
    """Compute 4-point circular bounding contour.

    Mimics CATK cal_polygon_contour: places 4 points around a circle at each
    position.  Accepts any leading dimensions so the same function handles
    both the last-timestep call (M, 3) → (M, 4, 2) and the full-sequence
    call (K, seg_len, 3) → (K, seg_len, 4, 2).

    Args:
        pos_heading: (..., 3) — x, y, heading
        radius: bounding circle radius

    Returns:
        contours: (..., 4, 2)
    """
    offsets = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    heading = pos_heading[..., 2:3]                      # (..., 1)
    angles = heading + offsets                            # (..., 4)
    x = pos_heading[..., 0:1]                            # (..., 1)
    y = pos_heading[..., 1:2]                            # (..., 1)
    cx = x + radius * np.cos(angles)                     # (..., 4)
    cy = y + radius * np.sin(angles)                     # (..., 4)
    return np.stack([cx, cy], axis=-1)                   # (..., 4, 2)


def compute_contour_dist(contours, center_contour):
    """Compute distance from each contour to a single center contour.

    Distance = mean L2 over 4 contour points at the final timestep.
    Matches CATK: torch.norm(X - x0, dim=-1).mean(-1)

    Args:
        contours: (M, 4, 2)  — final timestep contour for all segments
        center_contour: (4, 2)

    Returns:
        distances: (M,)
    """
    diff = contours - center_contour[None, :, :]  # (M, 4, 2)
    per_point = np.linalg.norm(diff, axis=-1)     # (M, 4)
    return per_point.mean(axis=-1)                # (M,)


def kdisk_cluster(segments, contours_last, K, tol, seed=42):
    """K-disk clustering matching CATK's Kdisk_cluster.

    Iterates K times. Each iteration picks a random remaining segment as
    center, removes all segments within tol distance, and stores the mean
    of removed segments as the token representative.

    Iteration 0 always picks index 0 (seeds the static/zero token).

    Args:
        segments: (M, seg_len, 3) — rel_x, rel_y, rel_heading
        contours_last: (M, 4, 2) — final timestep contour for each segment
        K: number of clusters
        tol: distance tolerance (game units)
        seed: random seed

    Returns:
        center_segments: (K, seg_len, 3)
    """
    rng = np.random.RandomState(seed)
    M = segments.shape[0]
    K = min(K, M)

    seg_remaining = segments.copy()
    cont_remaining = contours_last.copy()
    ret_list = []

    for i in range(K):
        if seg_remaining.shape[0] == 0:
            break

        if i == 0:
            choice_idx = 0  # always seed static token first
        else:
            choice_idx = rng.randint(seg_remaining.shape[0])

        x0 = cont_remaining[choice_idx]  # (4, 2)
        dists = compute_contour_dist(cont_remaining, x0)  # (M_remaining,)
        inside = dists <= tol  # segments within tolerance

        # Token = mean of all consumed segments.
        # x and y are averaged arithmetically (fine for Euclidean coords).
        # Heading is averaged with the circular mean: average sin and cos
        # independently, then recover the angle with arctan2.
        # Arithmetic mean of angles near the ±π boundary is wrong —
        # e.g. mean(π-0.1, -π+0.1) = 0 (right) instead of ±π (left).
        token = seg_remaining[inside].mean(axis=0, keepdims=True)  # (1, seg_len, 3)
        headings = seg_remaining[inside][:, :, 2]                  # (M_inside, seg_len)
        mean_sin = np.sin(headings).mean(axis=0)                   # (seg_len,)
        mean_cos = np.cos(headings).mean(axis=0)                   # (seg_len,)
        token[0, :, 2] = np.arctan2(mean_sin, mean_cos)
        ret_list.append(token)

        # Remove consumed segments
        outside = ~inside
        seg_remaining = seg_remaining[outside]
        cont_remaining = cont_remaining[outside]

        if (i + 1) % 256 == 0:
            remain_pct = seg_remaining.shape[0] * 100.0 / M
            print(f"  K-disk: {i+1}/{K} tokens, {remain_pct:.1f}% remaining")

        if seg_remaining.shape[0] == 0:
            print(f"  K-disk: all segments consumed after {i+1} tokens")
            break

    center_segments = np.concatenate(ret_list, axis=0)  # (actual_K, seg_len, 3)
    return center_segments


def visualize_dictionary(center_segments, output_path):
    """Visualize motion dictionary as trajectory paths with arrowheads (SMART/CATK Fig 1b style).

    Each token draws its full trajectory from origin with an arrowhead at the
    final frame. Color encodes movement direction; line length encodes distance.
    Styled for inclusion in a paper figure.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    seg_len = center_segments.shape[1]
    K = center_segments.shape[0]

    endpoint_xy = center_segments[:, -1, :2]       # (K, 2)
    dists = np.linalg.norm(endpoint_xy, axis=-1)   # (K,)
    max_dist = dists.max() if dists.max() > 0 else 1.0

    # Color by movement direction using a muted colormap.
    # 0° (straight-ahead/right) through 360° mapped to subdued tones.
    angles = np.arctan2(endpoint_xy[:, 1], endpoint_xy[:, 0])  # (K,) in [-π, π]
    angle_norm = (angles / (2 * np.pi)) % 1.0                   # → [0, 1]
    cmap = plt.cm.twilight
    colors = cmap(angle_norm)                                    # (K, 4) RGBA

    # Alpha: faint for near-stationary tokens, opaque for fast ones
    alphas = np.clip(0.25 + 0.65 * (dists / max_dist), 0.25, 0.90)

    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(K):
        xy = center_segments[i, :, :2]
        color = colors[i]
        alpha = float(alphas[i])

        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=alpha * 0.75,
                linewidth=0.8, solid_capstyle="round", zorder=2)

        ax.annotate(
            "", xy=(xy[-1, 0], xy[-1, 1]),
            xytext=(xy[-2, 0], xy[-2, 1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                alpha=alpha,
                lw=0.8,
                mutation_scale=7,
            ),
            zorder=3,
        )

    # Origin marker
    ax.plot(0, 0, "k.", markersize=4, zorder=10)

    ax.set_aspect("equal")
    ax.set_xlabel("Relative X (game units)", fontsize=11)
    ax.set_ylabel("Relative Y (game units)", fontsize=11)
    ax.set_title(f"Motion Token Dictionary  ($K={K}$, {seg_len} frames)",
                 fontsize=12, pad=8)
    ax.grid(True, linewidth=0.4, alpha=0.3, color="gray")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    mpl.rcParams.update(mpl.rcParamsDefault)
    print(f"Visualization saved → {output_path}")


def get_or_create_split(input_dir, output_dir, val_ratio, seed):
    """Get or create a reproducible train/val split at the replay level.

    If a split file already exists in output_dir, loads and returns it.
    Otherwise creates a new split, saves it, and returns it.

    Args:
        input_dir: Path to directory containing replay subdirectories
        output_dir: Path to save split JSON
        val_ratio: Fraction of replays for validation
        seed: Random seed for reproducibility

    Returns:
        dict with keys 'train', 'val', 'seed', 'val_ratio'
    """
    split_path = os.path.join(output_dir, "train_val_split.json")

    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            split = json.load(f)
        print(f"Loaded existing split from {split_path}")
        print(f"  Train: {len(split['train'])} replays, Val: {len(split['val'])} replays")
        return split

    input_dir = Path(input_dir)
    replay_hashes = sorted([d.name for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(replay_hashes)} replay directories for splitting")

    rng = np.random.RandomState(seed)
    rng.shuffle(replay_hashes)

    split_idx = int(len(replay_hashes) * (1 - val_ratio))
    split = {
        "train": replay_hashes[:split_idx],
        "val": replay_hashes[split_idx:],
        "seed": seed,
        "val_ratio": val_ratio,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Created new split → {split_path}")
    print(f"  Train: {len(split['train'])} replays, Val: {len(split['val'])} replays")
    return split


def main():
    parser = argparse.ArgumentParser(
        description="Build discrete motion token vocabulary for SC2 units."
    )
    parser.add_argument(
        "--input_dir", default=None,
        help="Directory of flat HDF5 scenarios (replay_hash/scenario.h5). "
             "Used only when --manifest is not provided.",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Path to a train manifest file (one relative path per line). "
             "When provided, --data_root is used to resolve paths.",
    )
    parser.add_argument(
        "--data_root", default=None,
        help="Root directory for resolving manifest relative paths.",
    )
    parser.add_argument("--output_dir", default="motion_dict_out/",
                        help="Output directory.")
    parser.add_argument("--segment_length", type=int, default=9,
                        help="Frames per segment (default: 9 = 0.5s at 16 FPS). "
                             "Matches CATK: shift+1=6 frames at 10Hz. Formula: fps*duration+1.")
    parser.add_argument("--segment_stride", type=int, default=None,
                        help="Stride between segment starts. Defaults to segment_length-1, "
                             "giving 1-frame overlap at boundaries (matching CATK: stride=shift=seg_len-1).")
    parser.add_argument("--n_clusters", type=int, default=2048,
                        help="Number of motion tokens (default: 2048).")
    parser.add_argument("--tol", type=float, default=0.05,
                        help="K-disk tolerance: segments within this contour distance are consumed (game units).")
    parser.add_argument("--max_scenarios", type=int, default=30000,
                        help="Stop loading after this many scenarios (for quick inspection).")
    parser.add_argument("--max_segments", type=int, default=500000,
                        help="Max segments for clustering (subsample if exceeded).")
    parser.add_argument("--max_step_disp", type=float, default=5.0,
                        help="If set, filter segments where any single-step displacement "
                             "exceeds this value (game units). Removes teleports/abilities. ")
    parser.add_argument("--min_alive_frames", type=int, default=48,
                        help="Skip units alive for fewer than this many total frames. "
                             "Matches CATK's 30-frame filter at 10Hz. Default: 48 = 3s at 16fps.")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of replays for validation (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Default stride = segment_length - 1 (matching CATK: stride=shift=seg_len-1).
    # Consecutive segments share exactly 1 boundary frame.
    # At 16fps with seg_len=9: stride=8, so segments cover frames [0..8], [8..16], etc.
    if args.segment_stride is None:
        args.segment_stride = args.segment_length - 1

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    t0 = time.time()

    # --- 1. Discover all scenario files ---
    if args.manifest:
        # Load from manifest (pre-split dataset)
        if args.data_root is None:
            print("ERROR: --data_root is required when using --manifest.")
            return
        data_root = Path(args.data_root)
        with open(args.manifest, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        h5_files = [data_root / line for line in lines]
        print(f"Loaded {len(h5_files)} scenario files from manifest {args.manifest}")
        split = None
    else:
        if args.input_dir is None:
            print("ERROR: Either --manifest or --input_dir must be provided.")
            return
        input_dir = Path(args.input_dir)
        h5_files = sorted(input_dir.glob("*/*.h5"))
        print(f"Found {len(h5_files)} scenario files in {input_dir}")
        # Train/val split (legacy path)
        split = get_or_create_split(args.input_dir, args.output_dir, args.val_ratio, args.seed)
        train_replays_set = set(split["train"])
        h5_files = [f for f in h5_files if f.parent.name in train_replays_set]
        print(f"After filtering to train split: {len(h5_files)} scenario files")

    if not h5_files:
        print("ERROR: No .h5 scenario files found. Check input arguments.")
        return

    # Shuffle and truncate to max_scenarios
    rng.shuffle(h5_files)
    scenario_limit = min(args.max_scenarios, len(h5_files)) if args.max_scenarios else len(h5_files)
    h5_files = h5_files[:scenario_limit]
    print(f"Using {len(h5_files)} scenarios (max_scenarios={args.max_scenarios})")

    # --- 2. Extract segments ---
    # On-the-fly dedup is skipped for visualization: K-disk clustering absorbs
    # near-duplicates naturally (tol-radius consumes all nearby segments into one token).
    kept_segs_list = []
    n_kept = 0
    n_total = 0
    n_scenarios_processed = 0

    for i, h5_path in enumerate(h5_files):
        data = load_scenario_motion(str(h5_path))
        coords, headings = extract_segments(
            data, args.segment_length, args.segment_stride,
            min_alive_frames=args.min_alive_frames,
        )
        if coords is not None:
            n_total += coords.shape[0]

            # Normalize this scenario's segments
            rel_xy, rel_heading = normalize_segments(coords, headings)

            # Optional teleport filter
            if args.max_step_disp is not None:
                step_disp = np.linalg.norm(np.diff(rel_xy, axis=1), axis=-1)
                keep = step_disp.max(axis=1) <= args.max_step_disp
                rel_xy, rel_heading = rel_xy[keep], rel_heading[keep]

            # Build (M, seg_len, 3) for this scenario and append directly
            scenario_segs = np.concatenate(
                [rel_xy, rel_heading[:, :, None]], axis=-1
            )
            kept_segs_list.append(scenario_segs)
            n_kept += scenario_segs.shape[0]

        n_scenarios_processed += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(h5_files)} scenarios, "
                  f"{n_total} raw segments, {n_kept} collected so far...")

    if not kept_segs_list:
        print("ERROR: No segments extracted. Check input data.")
        return

    segments = np.concatenate(kept_segs_list, axis=0)  # (M, seg_len, 3)
    print(f"\nTotal raw segments: {n_total}")
    print(f"Segments for clustering: {segments.shape[0]}")

    # --- 3. Compute contours (final timestep only, used for clustering distance) ---
    contours_last = compute_contours(segments[:, -1, :])  # (M, 4, 2)

    # --- 4. Prepend static (all-zeros) token — matches CATK which seeds the pool
    #        with torch.zeros([1, 6, 3]) so iteration 0 always produces a "don't move" token.
    static_seg = np.zeros((1, args.segment_length, 3), dtype=np.float32)
    static_cont = np.zeros((1, 4, 2), dtype=np.float32)  # radius=0 at origin
    segments = np.concatenate([static_seg, segments], axis=0)
    contours_last = np.concatenate([static_cont, contours_last], axis=0)

    n_cluster_input = segments.shape[0]

    # --- 5. Cluster (K-disk, matching CATK) ---
    print(f"\nClustering {n_cluster_input} segments into {args.n_clusters} tokens "
          f"(tol={args.tol})...")
    t_cluster = time.time()
    center_segments = kdisk_cluster(
        segments, contours_last, args.n_clusters, tol=args.tol, seed=args.seed
    )
    print(f"Clustering done in {time.time() - t_cluster:.1f}s")
    print(f"Final cluster centers: {center_segments.shape[0]}")

    # --- 6. Visualize ---
    viz_path = os.path.join(args.output_dir, "motion_dictionary_final.png")
    visualize_dictionary(center_segments, viz_path)

    # --- 7. Compute full contours for cluster centers (all timesteps) ---
    # Second call to compute_contours — mimics CATK's second cal_polygon_contour
    # call which uses all positions/headings (not just last).
    center_contours = compute_contours(center_segments)  # (K, seg_len, 4, 2)

    # --- 8. Save ---
    out_pkl = os.path.join(args.output_dir, "motion_dictionary_final.pkl")
    result = {
        "cluster_centers": center_contours,
        "metadata": {
            "K": center_contours.shape[0],
            "segment_length": args.segment_length,
            "segment_stride": args.segment_stride,
            "fps": 16,
            "tol": args.tol,
            "contour_radius": 0.5,
            "n_total_segments": n_total,
            "n_cluster_input": n_cluster_input,
            "n_scenarios_processed": n_scenarios_processed,
            "manifest": args.manifest,
            "train_replays": len(split["train"]) if split else None,
            "val_replays": len(split["val"]) if split else None,
            "min_alive_frames": args.min_alive_frames,
        },
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(result, f)
    print(f"Dictionary saved → {out_pkl}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  scenarios processed: {n_scenarios_processed}")
    print(f"  Total segments:   {n_total}")
    print(f"  Cluster centers:  {center_segments.shape[0]}")


if __name__ == "__main__":
    main()
