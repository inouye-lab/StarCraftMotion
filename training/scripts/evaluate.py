"""Generate and score rollouts using the paper's original evaluation routines."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ID_MAPS = "Abyssal_Reef_LE,Acolyte_LE,Ascension_to_Aiur_LE,Interloper_LE,Mech_Depot_LE"
OOD_MAPS = "Catallena_LE_(Void),Odyssey_LE"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", choices=["simulation", "intent", "target"], required=True)
    p.add_argument("--model", choices=["smart", "hmart"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--split", choices=["id", "ood"], default="id")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rollouts", type=int, default=32)
    p.add_argument("--limit-scenarios", type=int, default=0)
    a = p.parse_args()
    if not 1 <= a.workers <= 32: p.error("workers must be between 1 and 32")
    out = a.output.resolve(); data = a.data_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    maps = ID_MAPS if a.split == "id" else OOD_MAPS
    def run(args):
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)
    args = ["-m", "src.run", "experiment=sc_pre_bc", "action=test",
            f"ckpt_path={a.checkpoint.resolve()}", f"paths.sc_dataset_root={data}",
            f"trainer.accelerator={a.device}", "trainer.devices=1", "logger=null",
            "data.test_batch_size=1", f"data.num_workers={a.workers}",
            f"data.test_map_names={json.dumps(maps.split(chr(44)))}", f"hydra.run.dir={out}",
            f"model.model_config.decoder.num_concepts={0 if a.model == 'smart' else 16}",
            "model.model_config.val_open_loop=false", "model.model_config.val_closed_loop=true",
            "model.model_config.save_closed_rollouts=true",
            f"model.model_config.n_rollout_closed_val={a.rollouts}"]
    if a.limit_scenarios: args.append(f"trainer.limit_test_batches={a.limit_scenarios}")
    run(args)
    if a.table == "simulation":
        run(["scripts/eval_sc_rollouts.py", "--rollouts_dir", out / "rollouts",
             "--replays_dir", data, "--map_dir", ROOT / "assets/static",
             "--map_names", maps, "--out_csv", out / "metrics.csv", "--n_workers", a.workers])
        sys.path.insert(0, str(ROOT / "scripts"))
        import aggregate_sweep_eval as agg
        rows = agg._read_rows(out / "metrics.csv", a.model, a.split, out.name)
        means = agg._weighted_mean(rows)
        result = {mode: list(agg._compute_composites(means, mode_filter=mode).values())[0]
                  for mode in ["own", "opponent"]}
        (out / "composites.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        run(["scripts/compute_intent_candidates.py", "--rollouts_dir", out / "rollouts",
             "--replays_dir", data, "--map_names", maps, "--out_dir", out / "intent",
             "--w_max", 48, "--target_merge_dist", 10, "--n_workers", a.workers])
        run(["scripts/eval_intent_metrics.py", "--candidates_dir", out / "intent",
             "--w", 16, "--d", "inf", "--require_action_match", "true"])

if __name__ == "__main__": main()
