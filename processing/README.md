# StarCraft Motion -- Data Processing Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the end-to-end data-processing pipeline used to build
the **StarCraft Motion** dataset from raw StarCraft II replays. It walks you
from a fresh checkout through:

1. installing the StarCraft II 3.16.1 Linux binary, ladder maps, and the
   official Replay Pack 1 (all from Blizzard's public CDN, the same artifacts
   referenced by [Blizzard/s2client-proto](https://github.com/Blizzard/s2client-proto));
2. extracting per-replay HDF5 files containing full-precision unit
   trajectories, fog-of-war state, dynamic map state, and player economy;
3. splitting those replays into fixed-length, Waymo-Motion-style scenario
   chunks; and
4. building a per-map train / val / test split with optional
   adversarial-weighted sampling.

The repository also ships two manually curated reference modules
(`utils/coarse_action_mapping.py`, `utils/unit_type_map.py`) that are useful
when consuming the dataset downstream.

---

## Pipeline at a glance

```
                                   make install
                                        |
                                        v
                          $SC2PATH/Replays-3.16.1-Pack_1-fix
                                        |
            Stage 1 -- run_all_replay.sh (extract_replay_level.py)
                                        |
                                        v
                                $DATA_DIR/*.h5     (per-replay HDF5)
                                $MAP_DATA_DIR/*.h5 (static map data)
                                        |
            Stage 2 -- run_split_scenarios.sh (split_scenarios.py)
                                        |
                                        v
                       $SCENARIO_DIR/{replay_hash}/{chunk_idx:04d}.h5
                                        |
        (optional) python compute_scenario_index.py --input_dir $SCENARIO_DIR
                                        |
                                        v
                       $SCENARIO_DIR/scenario_index.pkl
                                        |
       Stage 3 -- run_create_split_adversarial.sh (create_dataset_split.py)
                                        |
                                        v
                $SPLIT_DIR/{train,val,test}/{map_name}/*.h5  (symlinks)
                $SPLIT_DIR/manifests/{train,val,test}_manifest.txt
```

Each chunk file holds **145 frames at 16 fps** (16 past + 1 current + 128
future, i.e. 1 s past + 8 s future), following the temporal layout of the
Waymo Open Motion Dataset.

The full HDF5 schema -- attributes, datasets, dtypes, and the
`visible_status` encoding -- is documented in
[`docs/extract_replay_level.md`](docs/extract_replay_level.md). All scenario
chunks share the same per-replay schema, sliced in time and remapped to the
units alive within the chunk.

---

## Requirements

| | |
|---|---|
| OS | Linux x86_64 (the SC2 Linux 3.16.1 binary only runs here) |
| Python | 3.7 (pinned to match `pysc2==3.0`; see `environment.yml`) |
| Tools | `wget`, `unzip`, `make`, `bash`, `conda` (Miniconda is auto-installed if missing) |
| Disk -- install | ~30 GB for SC2 + maps + Replay Pack 1 |
| Disk -- output | ~hundreds of GB for the full extracted dataset (varies with replay count and replay length; see *Runtime expectations* below) |
| CPU | Many cores recommended -- Stage 1 spawns one SC2 engine instance per worker |
| GPU | Not required at any stage |

Stage 1 (replay extraction) is the only stage that requires the SC2 game
engine to be running. Stages 2 and 3 operate entirely on HDF5 files.

---

## Installation

```bash
git clone https://github.com/inouye-lab/StarCraftMotion.git
cd StarCraftMotion/processing

# Bootstrap conda env + SC2 + maps + Replay Pack 1 (defaults install into ./datasets/).
make install
```

By default `make install`:

| Step | Where it goes | URL |
|---|---|---|
| Miniconda3 (if missing) | `$HOME/miniconda3` | https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh |
| Conda env `starcraft` | `$HOME/miniconda3/envs/starcraft` | from `environment.yml` |
| StarCraft II 3.16.1 | `./datasets/StarCraftII/` | http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip |
| Map pack: Ladder2017Season1 | `./datasets/StarCraftII/Maps/Ladder2017Season1/` | http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2017Season1.zip |
| Map pack: Melee | `./datasets/StarCraftII/Maps/Melee/` | http://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip |
| Replay Pack 1 | `./datasets/StarCraftII/Replays-3.16.1-Pack_1-fix/` | http://blzdistsc2-a.akamaihd.net/ReplayPacks/3.16.1-Pack_1-fix.zip |

All Blizzard archives are extracted with the documented EULA password
`iagreetotheeula`.

Override paths if you need to install elsewhere:

```bash
make install INSTALL_DIR=/data/sc2 CONDA_INSTALL_DIR=/opt/conda
```

After the install, activate the environment and tell `pysc2` where the game
lives:

```bash
conda activate starcraft
export SC2PATH="$(pwd)/datasets/StarCraftII"   # or wherever INSTALL_DIR points
```

### Manual installation (if `make install` doesn't fit your environment)

The Makefile is just a thin wrapper around the standard Blizzard URLs above.
You can perform the equivalent steps by hand:

```bash
# Conda env
conda env create -f environment.yml
conda activate starcraft

# SC2 binary
wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip
unzip -P iagreetotheeula SC2.3.16.1.zip -d /your/install/dir
rm SC2.3.16.1.zip

# Map packs
for pack in Ladder2017Season1 Melee; do
  wget http://blzdistsc2-a.akamaihd.net/MapPacks/${pack}.zip
  unzip -P iagreetotheeula ${pack}.zip -d /your/install/dir/StarCraftII/Maps/
  rm ${pack}.zip
done

# Replay Pack 1
wget http://blzdistsc2-a.akamaihd.net/ReplayPacks/3.16.1-Pack_1-fix.zip
unzip -P iagreetotheeula 3.16.1-Pack_1-fix.zip -d /your/install/dir/StarCraftII/
mv /your/install/dir/StarCraftII/Replays /your/install/dir/StarCraftII/Replays-3.16.1-Pack_1-fix
rm 3.16.1-Pack_1-fix.zip

export SC2PATH=/your/install/dir/StarCraftII
```

---

## Stage 1 -- Extract per-replay HDF5

Runs the SC2 engine three times per replay (omniscient + each player's
fog-of-war perspective) and writes one HDF5 file per replay containing every
unit's trajectory, visibility state, orders, economy, and the dynamic creep /
FOW maps. Static map data (pathing grid, placement grid, height map) is
de-duplicated across replays into one HDF5 per map.

```bash
bash run_all_replay.sh
```

Tunable via environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SC2PATH` | *(required)* | Path to your StarCraftII install. |
| `REPLAY_DIR` | `$SC2PATH/Replays-3.16.1-Pack_1-fix` | Directory of `.SC2Replay` files. |
| `DATA_DIR` | `./data/replay_hdf5` | Output directory for per-replay HDF5. |
| `MAP_DATA_DIR` | `./data/map_data/static` | Output directory for static map HDF5. |
| `NUM_PARALLEL_JOBS` | `8` | Joblib workers (one SC2 engine per worker). |

Output layout:

```
$DATA_DIR/
├── {replay_hash}.h5             # one per replay (full schema in docs/extract_replay_level.md)
├── run_config.json              # metadata for this extraction run
└── logs/
    ├── extraction-{ts}.log
    └── verbose-{ts}.log

$MAP_DATA_DIR/
└── {Map_Name}.h5                # one per unique map (spaces -> underscores)
```

**Runtime expectation.** Stage 1 is the slowest stage. As a rough baseline,
extracting ~30,000 replays on a 24-core machine takes on the order of a week.
Increase `NUM_PARALLEL_JOBS` to scale with available cores (each worker holds
its own SC2 engine, so memory grows roughly linearly with `NUM_PARALLEL_JOBS`).

Failed replays are logged and skipped; you can re-run the same command to
fill in any gaps -- already-extracted replays are detected and skipped.

See [`docs/extract_replay_level.md`](docs/extract_replay_level.md) for the
complete HDF5 schema, the three-pass extraction strategy, the `visible_status`
encoding, and an example reader.

---

## Stage 2 -- Split into scenario chunks

Slices each replay into fixed-length 145-frame chunks, drops short replays,
filters to units alive within each chunk, and remaps `target_id` indices
into the chunk's unit space.

```bash
bash run_split_scenarios.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `./data/replay_hdf5` | Input: Stage 1 output. |
| `SCENARIO_DIR` | `./data/scenarios` | Output: scenario chunks. |
| `NUM_PARALLEL_JOBS` | `8` | Joblib workers. |

Default chunking parameters (set inside `split_scenarios.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `--chunk_size` | 145 | Frames per chunk (16 past + 1 current + 128 future). |
| `--stride` | 144 | Frames between successive chunks (chunks share 1 frame). |
| `--fps` | 16 | Game frames per chunk frame. |
| `--map_frame_skip` | 16 | Frames between dynamic-map snapshots. |
| `--min_duration` | 120 | Skip replays shorter than 120 s. |
| `--min_apm` | 1 | Skip replays where either player has APM < 1. |
| `--skip_incomplete` | true | Drop the trailing chunk if stride doesn't divide evenly. |

Output layout:

```
$SCENARIO_DIR/
└── {replay_hash}/
    ├── 0000.h5
    ├── 0001.h5
    └── ...
```

Each chunk's HDF5 schema mirrors the per-replay schema (see
`docs/extract_replay_level.md`), with new attributes
`current_frame_idx`, `chunk_start_frame`, `chunk_end_frame`, `chunk_idx`, and
`parent_replay`.

---

## Stage 3 -- Build train / val / test split

Splits replays per map into train / val / test, samples a fraction of
scenarios per replay, and writes the result as symlinks plus per-split
manifest files. Maps named via `--id_maps` (default: 5 ladder maps from
Replay Pack 1) are split 80 / 10 / 10; all other maps go into the
out-of-distribution test set only.

### 3a. Build the scenario interaction index (required for adversarial sampling)

```bash
python compute_scenario_index.py --input_dir ./data/scenarios
```

This scans every chunk and records, per scenario:
- `mutual_frame_cnt` -- number of frames with any mutually visible unit
- `mutual_unit_sum`  -- total (unit, frame) pairs with `visible_status == 8`
- `transition_cnt`   -- visibility transitions debounced with a 3-frame hold

Output: `./data/scenarios/scenario_index.pkl`.

### 3b. Build the split

```bash
bash run_create_split_adversarial.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `SCENARIO_DIR` | `./data/scenarios` | Input: Stage 2 output. |
| `SPLIT_DIR` | `./data/split_adversarial` | Output: split. |
| `SAMPLE_RATE` | `0.1` | Fraction of scenarios sampled per replay. |
| `ID_MAPS` | `Abyssal_Reef_LE Acolyte_LE Ascension_to_Aiur_LE Interloper_LE Mech_Depot_LE` | In-distribution map names. |

With the index present, sampling is weighted by
`log1p(mutual_unit_sum) + log1p(transition_cnt)`, which biases sampling
toward scenarios with meaningful inter-player interaction.

Output layout:

```
$SPLIT_DIR/
├── train/{map_name}/{replay_hash}_{scenario_idx}.h5    # symlinks
├── val/{map_name}/{replay_hash}_{scenario_idx}.h5
├── test/{map_name}/{replay_hash}_{scenario_idx}.h5
└── manifests/
    ├── train_manifest.txt
    ├── val_manifest.txt
    └── test_manifest.txt
```

### Uniform sampling (no adversarial index)

If you'd rather sample uniformly, skip 3a and call the underlying script
directly:

```bash
python create_dataset_split.py \
  --input_dir ./data/scenarios \
  --output_dir ./data/split_uniform \
  --sample_rate 0.1 \
  --id_maps Abyssal_Reef_LE Acolyte_LE Ascension_to_Aiur_LE Interloper_LE Mech_Depot_LE
```

---

## Reference data (`utils/`)

The pipeline scripts above don't import these modules -- they are shipped
alongside the dataset so downstream consumers have a canonical mapping for
the raw integer IDs stored in HDF5.

### `utils/coarse_action_mapping.py`

Maps each raw SC2 `ability_id` (the integer stored in
`unit_data/repeated/ability_id`) to one of 12 coarse action classes
(`NO_OP`, `MOVE`, `ATTACK`, `HARVEST`, `TRAIN`, `BUILD`, `RESEARCH`, `MORPH`,
`EFFECT`, `TRANSPORT`, `BURROW`, `UNKNOWN`). The mapping is built from
Blizzard's stable enum (~468 entries) plus 10 manually curated overrides for
ability IDs that appear in replays but are missing from the enum, validated
to 100 % coverage on a 13,957-replay reference set.

```python
from utils import get_coarse_action, COARSE_ACTION_NAMES

label = get_coarse_action(3674)               # -> integer label
name  = COARSE_ACTION_NAMES[label]            # -> "ATTACK", "MOVE", ...
```

### `utils/unit_type_map.py`

Static remap of every SC2 `unit_type` ID into a contiguous range
`0 .. NUM_UNIT_TYPES-1` (with index `0` reserved for "Unknown"), plus
race / movement metadata. Vendored from `pysc2.lib.units` so the public
release does not import `pysc2` outside of Stage 1.

```python
import numpy as np
from utils import remap_unit_type, UNIT_ID_TO_NAME, UNIT_ID_TO_RACE

dense = remap_unit_type(np.array([48, 341, 18]))   # raw -> contiguous
print(UNIT_ID_TO_NAME[48], UNIT_ID_TO_RACE[48])    # "Marine", "Terran"
```

---

## Reading a chunk

```python
import h5py
import numpy as np

with h5py.File("./data/scenarios/<replay_hash>/0000.h5", "r") as f:
    print(f.attrs["map_name"], f.attrs["total_frames"], f.attrs["current_frame_idx"])

    # All alive units at the "current" frame
    cur = int(f.attrs["current_frame_idx"])
    alive = f["unit_data/repeated/is_alive"][cur]      # (N,) bool
    coords = f["unit_data/repeated/coordinate"][cur]   # (N, 3) float16
    types = f["unit_data/repeated/unit_type"][cur]     # (N,) uint32
    owners = f["unit_data/global/unit_owner"][:]       # (N,) uint8

    for i in np.where(alive)[0][:5]:
        print(f"  unit {i}: type={types[i]} owner={owners[i]} pos={coords[i]}")
```

The full schema is in [`docs/extract_replay_level.md`](docs/extract_replay_level.md).

---

## Citing

Citation information will be added here.

```
@misc{starcraft_motion,
  title  = {{StarCraft Motion} dataset},
  year   = {2026},
  note   = {Citation forthcoming.}
}
```

---

## License

[MIT](LICENSE).

---

## Acknowledgements

This pipeline depends on
[StarCraft II](https://starcraft2.com/) (the Linux 3.16.1 game binary, ladder
maps, and Replay Pack 1 are distributed by Blizzard and referenced from
[Blizzard/s2client-proto](https://github.com/Blizzard/s2client-proto)) and
[`pysc2`](https://github.com/google-deepmind/pysc2) for replay decoding.

## Camera-ready data release

The scripts in [`release/`](release/) preserve the reviewed Parquet fields and add
the missing HDF5 metadata, including both player starting locations needed by HMART.
See [Parquet release and HDF5 reconstruction](docs/parquet_release.md) for conversion,
resumption and verification commands. Both directions share `release/schema.py`.
