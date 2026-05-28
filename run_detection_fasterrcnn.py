"""Запуск экспериментов по курикулум-обучению: Faster R-CNN / VOC."""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch

from curriculum.strategies import Baseline, StageWise, Static, Online, AntiCurriculum
from datasets.voc_fasterrcnn_dataset import (
    VOCFasterRCNNDataset, TRAIN_SPLITS, VAL_SPLITS
)
from training.fasterrcnn_trainer import FasterRCNNTrainer

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Faster R-CNN + Curriculum Learning / VOC")
parser.add_argument("--dataset_type", default="original",
                    choices=["original", "noisy", "imbalance", "artifacts"])
parser.add_argument("--strategies", default="baseline,stagewise,static,online")
parser.add_argument("--total_steps", type=int, default=1000)
parser.add_argument("--batch",       type=int, default=4)
parser.add_argument("--lr",          type=float, default=1e-4)
parser.add_argument("--imgsz",       type=int, default=640)
parser.add_argument("--workers",     type=int, default=2)
parser.add_argument("--device",      default=None)
parser.add_argument("--noise_level",     type=float, default=0.3)
parser.add_argument("--imbalance_factor",type=float, default=0.3)
parser.add_argument("--artifact_level",  type=float, default=0.4)
parser.add_argument("--artifact_type",   default="blur",
                    choices=["blur", "noise", "low_res", "mixed"])
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)

# ── Device ─────────────────────────────────────────────────────────────────────

if args.device is None:
    if torch.cuda.is_available():
        device = "cuda"
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
assert strategy_names, f"No valid strategies: {args.strategies}"

# ── Experiment name ────────────────────────────────────────────────────────────

def make_exp_name():
    tag = f"voc_{args.dataset_type}"
    if args.dataset_type == "noisy":
        tag += f"_noise{args.noise_level}"
    elif args.dataset_type == "imbalance":
        tag += f"_im{args.imbalance_factor}"
    elif args.dataset_type == "artifacts":
        tag += f"_{args.artifact_type}{args.artifact_level}"
    return tag + "_fasterrcnn"

exp_name    = make_exp_name()
results_dir = Path("results") / exp_name
results_dir.mkdir(parents=True, exist_ok=True)

print(f"\nExperiment: {exp_name}")
print(f"Strategies: {strategy_names}")
print(f"Budget: {args.total_steps} steps\n")

# ── Dataset kwargs ─────────────────────────────────────────────────────────────

ds_kwargs = dict(
    imgsz=args.imgsz,
    difficulty_type=args.dataset_type,
    noise_level=args.noise_level,
    imbalance_factor=args.imbalance_factor,
    artifact_level=args.artifact_level,
    artifact_type=args.artifact_type,
)

# Validation датасет создаётся один раз
ds_val = VOCFasterRCNNDataset(VAL_SPLITS, **{**ds_kwargs, 'difficulty_type': 'original'})

# ── Main loop ─────────────────────────────────────────────────────────────────

all_results = []

for strat_name in strategy_names:
    print(f"\n{'='*60}")
    print(f"  FASTER R-CNN | {strat_name.upper()}")
    print(f"  VOC / {args.dataset_type} | budget={args.total_steps} steps")
    print(f"{'='*60}")

    ds_train = VOCFasterRCNNDataset(TRAIN_SPLITS, **ds_kwargs)
    strategy = ALL_STRATEGIES[strat_name]()

    trainer = FasterRCNNTrainer(
        dataset_train=ds_train,
        dataset_val=ds_val,
        curriculum_strategy=strategy,
        total_steps=args.total_steps,
        batch_size=args.batch,
        lr=args.lr,
        device=device,
        num_classes=21,   # 20 VOC + фон
        workers=args.workers,
    )

    start = time.time()
    metrics = trainer.train()
    elapsed = time.time() - start

    map50    = metrics.get('metrics/mAP50(B)', 0.0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)

    row = {
        'experiment':        exp_name,
        'model':             'fasterrcnn_mobilenet_v3',
        'strategy':          strat_name,
        'dataset_type':      args.dataset_type,
        'map50':             map50,
        'map50_95':          map50_95,
        'total_steps':       trainer._steps_done,
        'total_epochs':      trainer.epoch,
        'training_time_sec': round(elapsed, 1),
    }
    all_results.append(row)

    print(f"\n  map50={map50:.4f} | steps={trainer._steps_done} | "
          f"epochs={trainer.epoch} | time={elapsed:.0f}s")

# ── Save ───────────────────────────────────────────────────────────────────────

if all_results:
    df = pd.DataFrame(all_results)
    out = results_dir / "detection_results.csv"
    df.to_csv(out, index=False)
    print(f"\nResults → {out}")
    print(df[['strategy', 'map50', 'total_steps', 'total_epochs']].to_string(index=False))

config = vars(args)
config['exp_name'] = exp_name
(results_dir / "experiment_config.json").write_text(json.dumps(config, indent=2))
