#!/bin/bash
# Stage 1: Extract per-replay HDF5 files from .SC2Replay files.
#
# Required env:
#   SC2PATH               -- absolute path to your StarCraftII install (see README).
#
# Tunable env (defaults shown):
#   REPLAY_DIR            -- directory of .SC2Replay files
#                            [default: $SC2PATH/Replays-3.16.1-Pack_1-fix]
#   DATA_DIR              -- output directory for per-replay HDF5 files
#                            [default: ./data/replay_hdf5]
#   MAP_DATA_DIR          -- output directory for static map HDF5 files
#                            [default: ./data/map_data/static]
#   NUM_PARALLEL_JOBS     -- joblib worker count (one SC2 engine per worker)
#                            [default: 8]

set -e

: "${SC2PATH:?Set SC2PATH to your StarCraftII install (see README)}"
: "${REPLAY_DIR:=${SC2PATH}/Replays-3.16.1-Pack_1-fix}"
: "${DATA_DIR:=./data/replay_hdf5}"
: "${MAP_DATA_DIR:=./data/map_data/static}"
: "${NUM_PARALLEL_JOBS:=8}"

python extract_replay_level.py \
  --replay_dir "${REPLAY_DIR}" \
  --data_save_dir "${DATA_DIR}" \
  --map_data_dir "${MAP_DATA_DIR}" \
  --num_parallel_jobs "${NUM_PARALLEL_JOBS}"

echo
echo "=== Stage 1 (replay-level extraction) complete ==="
echo "Per-replay HDF5: ${DATA_DIR}"
echo "Static map data: ${MAP_DATA_DIR}"
