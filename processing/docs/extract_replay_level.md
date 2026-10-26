# Replay-Level HDF5 Extraction

`extract_replay_level.py` converts StarCraft II replay files (`.SC2Replay`) into self-contained HDF5 files that preserve raw, high-fidelity data. Each replay produces one `.h5` file containing all unit tracking data, player economy, and dynamic map state, so downstream analysis never needs the SC2 engine again.

## Motivation

The existing pipeline (`script-extract-data.py`) quantizes coordinates to a 256x256 grid, maps unit types to compact 0-255 channel indices, and normalizes health/shield/energy to uint8. This loses precision and ties the data format to custom mappings.

`extract_replay_level.py` instead stores:

- **Original float coordinates** (x, y, z) in the map's native coordinate system
- **Raw SC2 unit type IDs** (e.g., Marine = 48, Hatchery = 341)
- **Raw ability IDs** from the SC2 protobuf API
- **Full-precision health, shield, energy** as float16
- **Original map dimensions** (not forced to 256x256)

## Prerequisites

1. **StarCraft II** installed with replays and maps available.
2. **Conda environment** with required packages:
   ```bash
   conda activate starcraft
   pip install h5py  # if not already installed via environment.yml
   ```
3. **`SC2PATH` environment variable** pointing to the StarCraft II installation:
   ```bash
   export SC2PATH=/path/to/StarCraftII/
   ```

## Usage

### Single replay

```bash
python extract_replay_level.py \
  --replay /path/to/replay.SC2Replay \
  --data_save_dir data/my-extraction
```

### Directory of replays

```bash
python extract_replay_level.py \
  --replay_dir /path/to/SC2Replays/ \
  --data_save_dir data/my-extraction \
  --num_parallel_jobs 8
```

### Quick test run (limited steps)

```bash
python extract_replay_level.py \
  --replay /path/to/replay.SC2Replay \
  --data_save_dir data/test-run \
  --max_steps 300
```

## Command-Line Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--replay` | string | None | Path to a single `.SC2Replay` file |
| `--replay_dir` | string | None | Directory containing `.SC2Replay` files |
| `--data_save_dir` | string | `data/replay-level-{timestamp}` | Output directory for per-replay HDF5 files |
| `--map_data_dir` | string | `map_data/static` | Directory for static map HDF5 files |
| `--num_parallel_jobs` | int | 1 | Number of parallel joblib workers |
| `--max_replays` | int | -1 | Maximum replays to process (-1 = all) |
| `--step_mul` | int | 1 | Game steps per observation |
| `--max_steps` | int | -1 | Max steps per replay (-1 = unlimited) |
| `--map_frame_skip` | int | 16 | Frames between dynamic map snapshots |
| `--map_path` | string | None | Override map path for replay |

Specify either `--replay` or `--replay_dir`, not both.

## How It Works

The script runs **three passes** over each replay using the SC2 engine:

1. **Pass 1 (Omniscient)**: `disable_fog=True`, `observed_player_id=1`
   - Extracts all unit fields (position, health, type, passengers, etc.)
   - Extracts Player 1 economy (minerals, vespene, food)
   - Extracts creep map every `map_frame_skip` frames
   - Note: `unit.orders` are not populated in this pass (SC2 3.16.1 limitation)

2. **Pass 2 (Player 1 FOW)**: `disable_fog=False`, `observed_player_id=1`
   - Records per-unit visibility state from Player 1's perspective
   - Extracts Player 1's fog-of-war map every `map_frame_skip` frames
   - Extracts orders (ability_id, target_pos, target_unit_tag) for Player 1's units
   - Extracts passenger data (is_carried, health, etc.) for Player 1's transports

3. **Pass 3 (Player 2 FOW)**: `disable_fog=False`, `observed_player_id=2`
   - Records per-unit visibility state from Player 2's perspective
   - Extracts Player 2's fog-of-war map and economy data
   - Extracts orders for Player 2's units
   - Extracts passenger data for Player 2's transports

After all passes, visibility states are combined into a single `visible_status` field and everything is written to HDF5.

## Output Files

### Per-Replay HDF5: `{data_save_dir}/{replay_hash}.h5`

#### Metadata (HDF5 attributes)

| Attribute | Type | Description |
|-----------|------|-------------|
| `replay_name` | string | Replay file stem (hash) |
| `map_name` | string | Map name (spaces replaced with `_`) |
| `total_frames` | int | Number of extracted frames (T) |
| `max_unique_units` | int | Total unique unit tags seen (N) |
| `map_size` | [int, int] | Map dimensions [H, W] |
| `playable_area` | [float x4] | Playable bounds [x0, y0, x1, y1] |
| `player_info` | JSON string | Player races, MMR, APM, results |
| `replay_info` | JSON string | Game duration, version, build info |
| `player_1_start_location` | float32[3] | Player 1's starting town hall (x, y, z) |
| `player_2_start_location` | float32[3] | Player 2's starting town hall (x, y, z) |

#### `/player_data` -- Economy (T, 2)

Each dataset has shape `(T, 2)` where column 0 = Player 1, column 1 = Player 2.

| Dataset | dtype |
|---------|-------|
| `minerals` | uint16 |
| `vespene` | uint16 |
| `food_used` | uint8 |
| `food_cap` | uint8 |

#### `/map_data/dynamic` -- Dynamic Maps (T_map, H, W)

Sampled every `map_frame_skip` frames, so `T_map = ceil(T / map_frame_skip)`.

| Dataset | dtype | Description |
|---------|-------|-------------|
| `creep` | bool | Zerg creep coverage |
| `fow_player_1` | bool | Player 1's visible area |
| `fow_player_2` | bool | Player 2's visible area |

#### `/unit_data/global` -- Per-Unit Constants (N,)

| Dataset | dtype | Description |
|---------|-------|-------------|
| `unit_tag` | uint64 | Raw SC2 engine tag (unique per unit instance) |
| `unit_owner` | uint8 | Owner (1 = P1, 2 = P2, 16 = neutral) |

#### `/unit_data/repeated` -- Per-Frame Unit Data (T, N, ...)

| Dataset | dtype | Shape | Description |
|---------|-------|-------|-------------|
| `is_alive` | bool | (T, N) | Whether the unit exists this frame |
| `is_flying` | bool | (T, N) | Air unit or lifted building |
| `is_burrowed` | bool | (T, N) | Zerg burrowed state |
| `is_carried` | bool | (T, N) | Inside a transport |
| `visible_status` | uint8 | (T, N) | Combined P1/P2 visibility (see below) |
| `unit_type` | uint32 | (T, N) | Raw SC2 unit type ID |
| `ability_id` | uint32 | (T, N) | First order's ability ID |
| `target_id` | uint32 | (T, N) | Target unit's row index (`0xFFFFFFFF` = no target) |
| `coordinate` | float16 | (T, N, 3) | Raw (x, y, z) position |
| `target_pos` | float16 | (T, N, 2) | Order target ground position (x, y) |
| `radius` | float16 | (T, N) | Unit collision radius |
| `heading` | float16 | (T, N) | Facing direction in radians (0 to 2pi) |
| `health` | float16 | (T, N) | Current health |
| `health_max` | float16 | (T, N) | Maximum health |
| `shield` | float16 | (T, N) | Current shield (Protoss) |
| `energy` | float16 | (T, N) | Current energy |
| `build_progress` | float16 | (T, N) | 0.0 to 1.0 |
| `mineral_contents` | uint16 | (T, N) | Remaining minerals (mineral fields) |
| `vespene_contents` | uint16 | (T, N) | Remaining vespene (geysers) |

#### `visible_status` Encoding

The `visible_status` field encodes both players' visibility as `p1_state * 3 + p2_state`, where each state is:
- 0 = Unseen (unit not in that player's observation)
- 1 = Snapshot (last known position, `display_type == 2`)
- 2 = Visible (currently visible, `display_type == 1`)

| Value | P1 State | P2 State |
|-------|----------|----------|
| 0 | Unseen | Unseen |
| 1 | Unseen | Snapshot |
| 2 | Unseen | Visible |
| 3 | Snapshot | Unseen |
| 4 | Snapshot | Snapshot |
| 5 | Snapshot | Visible |
| 6 | Visible | Unseen |
| 7 | Visible | Snapshot |
| 8 | Visible | Visible |

### Static Map HDF5: `{map_data_dir}/{Map_Name}.h5`

One file per unique map (deduplicated across replays). Spaces in map names are replaced with `_`.

| Dataset | dtype | Shape | Description |
|---------|-------|-------|-------------|
| `pathing_grid` | bool | (H, W) | Pathable terrain |
| `placement_grid` | bool | (H, W) | Valid building placement |
| `height_map` | uint8 | (H, W) | Terrain height |

## Reading the Output

```python
import h5py
import numpy as np

with h5py.File('data/my-extraction/replay_hash.h5', 'r') as f:
    # Metadata
    print(f.attrs['map_name'], f.attrs['total_frames'], f.attrs['max_unique_units'])

    # Get all alive units at frame 100
    alive = f['unit_data/repeated/is_alive'][100]       # (N,) bool
    coords = f['unit_data/repeated/coordinate'][100]    # (N, 3) float16
    types = f['unit_data/repeated/unit_type'][100]      # (N,) uint32
    owners = f['unit_data/global/unit_owner'][:]        # (N,) uint8

    alive_idx = np.where(alive)[0]
    for i in alive_idx[:5]:
        print(f'  Unit row={i}, type={types[i]}, owner={owners[i]}, '
              f'pos=({coords[i,0]:.1f}, {coords[i,1]:.1f}, {coords[i,2]:.1f})')

    # Player 1 minerals over time
    minerals_p1 = f['player_data/minerals'][:, 0]      # (T,) uint16

    # Creep at map-frame 0
    creep = f['map_data/dynamic/creep'][0]              # (H, W) bool
```

```python
# Reading a static map file
with h5py.File('map_data/static/Abyssal_Reef_LE.h5', 'r') as f:
    pathing = f['pathing_grid'][:]       # (H, W) bool
    height = f['height_map'][:]          # (H, W) uint8
```

## Comparison with Old Pipeline

| Aspect | `script-extract-data.py` | `extract_replay_level.py` |
|--------|--------------------------|---------------------------|
| Output format | NPZ (numpy) + MP4 video | HDF5 |
| Coordinates | Quantized to 256x256 uint8 grid | Raw float16 (x, y, z) |
| Unit types | Custom 0-255 channel mapping | Raw SC2 IDs (uint32) |
| Actions | Custom coarse/fine mapping | Raw ability IDs (uint32) |
| Health/shield/energy | Normalized 0-255 uint8 | Raw values (float16) |
| Map dimensions | Cropped and resized to 256x256 | Original map size (H, W) |
| Morph tracking | Links morphed Zerg units to source row | No morph tracking (new tag = new row) |
| Video output | MP4 (god's-eye + 2 FOW) | None |
| Dependencies | ffmpeg, torch, scikit-image | h5py only |

## Notes

- **Parallel safety**: Both static map files and per-replay HDF5 files use atomic writes (temp file + `os.rename`) to prevent corruption from parallel workers or crashes.
- **Error handling**: Failed replays are logged and skipped without stopping the batch.
- **Memory**: Pass 1 writes directly into pre-allocated numpy arrays (sized from `game_duration_loops`), avoiding the ~30 GB Python dict overhead per long replay. Arrays grow dynamically if needed and are trimmed to exact `(T, N)` after Pass 1.
- **Passengers**: Units inside transports are tracked with `is_carried=True`, inheriting the transport's position and heading.
- **Orders**: Only the first order per unit per frame is extracted. `target_id` holds the target unit's row index (`0xFFFFFFFF` means no target or unresolved target tag); `target_pos` holds the ground target position.
