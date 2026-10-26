#!/bin/bash
# Stage 3: Build train/val/test split with adversarial-weighted sampling.
#
# Requires scenario_index.pkl in $SCENARIO_DIR (produced by
# `python compute_scenario_index.py --input_dir $SCENARIO_DIR`).
#
# Tunable env (defaults shown):
#   SCENARIO_DIR          -- input dir of scenario chunks (Stage 2 output)
#                            [default: ./data/scenarios]
#   SPLIT_DIR             -- output dir for the split
#                            [default: ./data/split_adversarial]
#   SAMPLE_RATE           -- fraction of scenarios to sample per replay
#                            [default: 0.1]
#   ID_MAPS               -- space-separated list of in-distribution map names;
#                            all other maps go into the OOD test set only
#                            [default: 5 ladder maps from Replay Pack 1]

set -e

: "${SCENARIO_DIR:=./data/scenarios}"
: "${SPLIT_DIR:=./data/split_adversarial}"
: "${SAMPLE_RATE:=0.1}"
: "${ID_MAPS:=Abyssal_Reef_LE Acolyte_LE Ascension_to_Aiur_LE Interloper_LE Mech_Depot_LE}"

INDEX="${SCENARIO_DIR}/scenario_index.pkl"
if [ ! -f "${INDEX}" ]; then
  echo "Missing ${INDEX}." >&2
  echo "Build it first with: python compute_scenario_index.py --input_dir ${SCENARIO_DIR}" >&2
  exit 1
fi

python create_dataset_split.py \
  --input_dir "${SCENARIO_DIR}" \
  --output_dir "${SPLIT_DIR}" \
  --adversarial_index "${INDEX}" \
  --sample_rate "${SAMPLE_RATE}" \
  --id_maps ${ID_MAPS}

echo
echo "=== Stage 3 (train/val/test split) complete ==="
echo "Split:     ${SPLIT_DIR}/{train,val,test}/{map_name}/"
echo "Manifests: ${SPLIT_DIR}/manifests/{train,val,test}_manifest.txt"
