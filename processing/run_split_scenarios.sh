#!/bin/bash
# Stage 2: Split replay-level HDF5 files into 145-frame scenario chunks.
#
# Tunable env (defaults shown):
#   DATA_DIR              -- input dir of per-replay HDF5 files (Stage 1 output)
#                            [default: ./data/replay_hdf5]
#   SCENARIO_DIR          -- output dir for scenario chunks
#                            [default: ./data/scenarios]
#   NUM_PARALLEL_JOBS     -- joblib worker count [default: 8]
#
# Default chunking (set inside split_scenarios.py): chunk_size=145, stride=144,
# fps=16, map_frame_skip=16, min_duration=120s, min_apm=1, skip_incomplete=True.

set -e

: "${DATA_DIR:=./data/replay_hdf5}"
: "${SCENARIO_DIR:=./data/scenarios}"
: "${NUM_PARALLEL_JOBS:=8}"

python split_scenarios.py \
  --input_dir "${DATA_DIR}" \
  --output_dir "${SCENARIO_DIR}" \
  --num_parallel_jobs "${NUM_PARALLEL_JOBS}"

echo
echo "=== Stage 2 (scenario chunking) complete ==="
echo "Scenario chunks: ${SCENARIO_DIR}/{replay_hash}/{chunk_idx:04d}.h5"
