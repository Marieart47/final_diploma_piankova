"""Запуск экспериментов по курикулум-обучению для задачи детекции объектов (YOLOv8 / VOC)."""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

from curriculum.strategies import Baseline, StageWise, Static, Online, AntiCurriculum
from training.detection_trainer import CurriculumDetectionTrainer

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", default="original",
                    choices=["original", "noisy", "imbalance", "artifacts"])
parser.add_argument("--strategies", default="baseline",
                    help="Comma-separated: baseline,stagewise,static,online,anticurriculum")
parser.add_argument("--model", default="yolov8n",
                    choices=["yolov8n", "yolov8s", "yolov8m"])
parser.add_argument("--data", default="VOC.yaml",
                    help="Data YAML для ultralytics (VOC.yaml или coco128.yaml).")
parser.add_argument("--total_steps", type=int, default=1000,
                    help="Total gradient steps per strategy run (budget_mode=steps).")
parser.add_argument("--budget_mode", default="steps", choices=["steps", "epochs"])
parser.add_argument("--max_epochs", type=int, default=100,
                    help="Upper bound on epochs (training stops at total_steps first).")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--device", default=None,
                    help="cuda / mps / cpu (auto-detected if omitted).")
parser.add_argument("--noise_level", type=float, default=0.3,
                    help="Label noise probability (dataset_type=noisy).")
parser.add_argument("--imbalance_factor", type=float, default=0.3,
                    help="Fraction of rare-class samples to remove (dataset_type=imbalance).")
parser.add_argument("--artifact_level", type=float, default=0.4,
                    help="Artifact intensity in [0, 1] (dataset_type=artifacts).")
parser.add_argument("--artifact_type", default="blur",
                    choices=["blur", "noise", "low_res", "mixed"],
                    help="Visual artifact type (dataset_type=artifacts).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--exp_suffix", default="")
args = parser.parse_args()

# ── Device ─────────────────────────────────────────────────────────────────────

if args.device is None:
    if torch.cuda.is_available():
        device = "0"       # ultralytics uses GPU index string for CUDA
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
else:
    device = args.device

print(f"Device: {device}")

# ── Strategies ─────────────────────────────────────────────────────────────────

ALL_STRATEGIES = {
    "baseline":       lambda: Baseline(),
    "stagewise":      lambda: StageWise(),
    "static":         lambda: Static(),
    "online":         lambda: Online(pct=0.3),
    "anticurriculum": lambda: AntiCurriculum(pct=0.3),
}

strategy_names = [s.strip() for s in args.strategies.split(",")
                  if s.strip() in ALL_STRATEGIES]
assert strategy_names, f"No valid strategies in: {args.strategies}"

# ── Experiment name ─────────────────────────────────────────────────────────────

def make_exp_name() -> str:
    ds_tag = args.data.replace('.yaml', '').lower()
    base = f"{ds_tag}_{args.dataset_type}"
    if args.dataset_type == "noisy":
        base += f"_noise{args.noise_level}"
    elif args.dataset_type == "imbalance":
        base += f"_im{args.imbalance_factor}"
    elif args.dataset_type == "artifacts":
        base += f"_{args.artifact_type}_art{args.artifact_level}"
    name = f"{base}_{args.model}"
    if args.exp_suffix:
        name += f"_{args.exp_suffix}"
    return name

exp_name = make_exp_name()
results_dir = Path("results") / exp_name
results_dir.mkdir(parents=True, exist_ok=True)

print(f"Experiment: {exp_name}")
print(f"Results:    {results_dir}")

# Save config
config = vars(args)
config["exp_name"] = exp_name
config["device"] = device
(results_dir / "experiment_config.json").write_text(json.dumps(config, indent=2))

# ── Run ────────────────────────────────────────────────────────────────────────

all_results = []

for strat_name in strategy_names:
    print(f"\n{'='*60}")
    print(f"  {args.model.upper()} | {strat_name.upper()}")
    print(f"  Dataset: voc ({args.dataset_type}), budget_mode={args.budget_mode}")
    print(f"{'='*60}")

    strategy = ALL_STRATEGIES[strat_name]()

    # Ultralytics trainer configuration
    overrides = {
        'model':     f'{args.model}.pt',
        'data':      args.data,
        'epochs':    args.max_epochs,
        'batch':     args.batch,
        'imgsz':     args.imgsz,
        'workers':   args.workers,
        'device':    device,
        'seed':      args.seed,
        'verbose':   True,
        'project':   str(results_dir),
        'name':      strat_name,
        'exist_ok':  True,
        'plots':     False,
        'save':      True,
        'val':       True,
        'cache':     False,
    }

    trainer = CurriculumDetectionTrainer(
        overrides=overrides,
        curriculum_strategy=strategy,
        total_steps=args.total_steps,
        budget_mode=args.budget_mode,
        difficulty_type=args.dataset_type,
        noise_level=args.noise_level,
        imbalance_factor=args.imbalance_factor,
        artifact_level=args.artifact_level,
        artifact_type=args.artifact_type,
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # Extract final metrics
    metrics = getattr(trainer, 'metrics', {}) or {}
    map50    = metrics.get('metrics/mAP50(B)',    0.0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)

    row = {
        'experiment':        exp_name,
        'model':             args.model,
        'strategy':          strat_name,
        'dataset_type':      args.dataset_type,
        'map50':             map50,
        'map50_95':          map50_95,
        'total_steps':       trainer._steps_done,
        'total_epochs':      trainer.epoch,
        'training_time_sec': round(elapsed, 1),
    }
    all_results.append(row)

    print(f"\n  map50={map50:.4f} | map50-95={map50_95:.4f} | "
          f"steps={trainer._steps_done} | epochs={trainer.epoch} | "
          f"time={elapsed:.0f}s")

# ── Save results ───────────────────────────────────────────────────────────────

if all_results:
    df = pd.DataFrame(all_results)
    out_path = results_dir / "detection_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")
    print(df[['strategy', 'map50', 'map50_95', 'total_steps', 'total_epochs']].to_string(index=False))
