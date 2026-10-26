# StarCraft Motion

Code and project page for **StarCraft Motion: A Dataset for Agent Simulation in Adversarial and Partially Observable Scenarios**. Includes replay preprocessing, SMART/HMART training, CAT-K fine-tuning, rollout generation, and simulation, intent, and target-position evaluation.

[Project page](https://inouye-lab.github.io/StarCraftMotion/) · [Dataset](https://huggingface.co/datasets/inouye-lab/StarCraftMotion) · [Sample](https://huggingface.co/datasets/inouye-lab/StarCraftMotion_sample) · [Checkpoints](https://huggingface.co/inouye-lab/StarCraftMotion-checkpoints)

## Install

Use Python 3.11.9. Run the following commands from `training/`:

```bash
conda create -n starcraft-motion python=3.11.9 pip -y
conda activate starcraft-motion
cd training
pip install -r requirements-cpu.txt
# For CUDA 12.1 instead: pip install -r requirements-gpu.txt
```

Raw replay extraction uses the separate [processing environment and instructions](processing/README.md).

## Data

Download Parquet and reconstruct HDF5 once; training reads HDF5:

```bash
hf download inouye-lab/StarCraftMotion_sample --repo-type dataset --local-dir data/parquet
python ../processing/release/parquet_to_hdf5.py --src data/parquet --out data/hdf5 --workers 8
hf download inouye-lab/StarCraftMotion-checkpoints --repo-type model --local-dir checkpoints
```

Replace the sample dataset name with `inouye-lab/StarCraftMotion` for the full dataset: 469,187 scenarios, approximately 31 GB Parquet and 90 GB reconstructed HDF5. The required motion dictionary and seven static map rasters are included in `training/assets/`. Change `paths.sc_dataset_root` to use another data location.

## Train

Use batch size ≥ 2 for training. HMART uses `num_concepts=16`; set it to `0` to train SMART.

```bash
# Behavior cloning:
python -m src.run experiment=sc_pre_bc trainer.max_epochs=16 data.train_batch_size=16 data.val_batch_size=16 paths.sc_dataset_root=data/hdf5
# CAT-K fine-tuning:
python -m src.run experiment=sc_clsft ckpt_path=checkpoints/hmart.ckpt data.train_batch_size=32 paths.sc_dataset_root=data/hdf5
```

For CPU training, append `trainer.accelerator=cpu trainer.devices=1`.

## Evaluate

The released files are `hmart.ckpt` (HMART) and `hmart_catk.ckpt` (HMART + CAT-K). Simulation tables report mean ± std over three seeds; these weights are from one of them, the run used for the single-run intent and target results.

```bash
# Observer-unit and opponent-modeling simulation tables:
python scripts/evaluate.py --table simulation --model hmart --checkpoint checkpoints/hmart_catk.ckpt --data-root data/hdf5 --output logs/hmart_catk_simulation --workers 8
# Per-action intent precision and recall:
python scripts/evaluate.py --table intent --model hmart --checkpoint checkpoints/hmart.ckpt --data-root data/hdf5 --output logs/hmart_intent --workers 8
# Target-position error:
python scripts/evaluate.py --table target --model hmart --checkpoint checkpoints/hmart.ckpt --data-root data/hdf5 --output logs/hmart_target --workers 8
```

Repeat with each checkpoint. Evaluation uses 32 rollouts, top-5 sampling, temperature 1.0, both observers, and both own/opponent modes. Add `--device gpu` for GPU execution, `--split ood` for OOD maps, or `--limit-scenarios 5 --rollouts 2` for a small smoke check.

Simulation results are in `composites.json`; intent and target results are in `intent/intent_metrics_table.csv` and `intent/intent_distance_pooled.csv`. The paper intent columns are move, attack, harvest, build, morph, effect, transport, and burrow; train is omitted. Target columns are move, attack, harvest, and build.

## Citation

```bibtex
@inproceedings{chen2026starcraftmotion,
  title = {{StarCraft Motion}: A Dataset for Agent Simulation in Adversarial and Partially Observable Scenarios},
  author = {Yi-Chung Chen and Mingyu Kim and Ruqi Bai and James Zachary Hare and Jing Gao and David I. Inouye},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```

## License

Training code and checkpoints use the [NVIDIA Source Code License](training/LICENSE), including non-commercial restrictions. SMART components retain their [upstream license](training/src/smart/LICENSE.SMART). Preprocessing uses [MIT](processing/LICENSE). Blizzard replay and game assets remain subject to their respective terms.
