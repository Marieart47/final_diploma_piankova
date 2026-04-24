"""Запуск экспериментов по курикулум-обучению для задачи классификации."""

import torch
import pandas as pd
import os
import argparse
import time
import json
import numpy as np
from datetime import datetime
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, random_split

from datasets.difficult_datasets import create_dataset
from curriculum.strategies import Baseline, StageWise, Static, Online, AntiCurriculum
from models.classification import resnet18, swin_t
from training.classification_trainer import ClassificationTrainer

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "stl10"])
parser.add_argument("--dataset_type", default="original",
                    choices=["original", "imbalance", "noisy", "artifacts"])
parser.add_argument("--imbalance_factor", type=float, default=0.5)
parser.add_argument("--noise_level", type=float, default=0.3)
parser.add_argument("--artifact_level", type=float, default=0.5)
parser.add_argument("--artifact_type", default="blur",
                    choices=["mixed", "blur", "noise", "low_res"])
parser.add_argument("--total_steps", type=int, default=5000,
                    help="Total gradient steps per (model, strategy) run.")
parser.add_argument("--budget_mode", default="steps", choices=["steps", "epochs"],
                    help="'steps': equal gradient steps (recommended). "
                         "'epochs': equal epochs (biased, but simpler).")
parser.add_argument("--epochs", type=int, default=30,
                    help="Used only when --budget_mode=epochs.")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--models", default="resnet18")
parser.add_argument("--strategies", default="baseline,stagewise,static,online")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience (in epochs).")
parser.add_argument("--resize", type=int, default=None,
                    help="Resize images to this size. Default: 96 for STL-10/CIFAR, "
                         "224 recommended for Swin-T.")
args = parser.parse_args()

# ── Setup ──────────────────────────────────────────────────────────────────────

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")

if args.dataset_type == "original":
    exp_name = f"{args.dataset}_original"
elif args.dataset_type == "noisy":
    exp_name = f"{args.dataset}_noisy_noise{args.noise_level}"
elif args.dataset_type == "imbalance":
    exp_name = f"{args.dataset}_imbalance_im{args.imbalance_factor}"
elif args.dataset_type == "artifacts":
    exp_name = f"{args.dataset}_artifacts_{args.artifact_type}_art{args.artifact_level}"
else:
    exp_name = f"{args.dataset}_{args.dataset_type}"

# results_dir and checkpoints_dir are created per-model inside the main loop.

# ── Model & strategy registries ────────────────────────────────────────────────

ALL_MODELS = {"resnet18": resnet18, "swin_t": swin_t}
ALL_STRATEGIES = {
    "baseline": lambda: Baseline(),
    "stagewise": lambda: StageWise(),
    "static": lambda: Static(),
    "online": lambda: Online(pct=0.3),
    "anticurriculum": lambda: AntiCurriculum(pct=0.3),
}

model_registry = {k: v for k, v in ALL_MODELS.items() if k in args.models.split(",")}
strategy_registry = {k: ALL_STRATEGIES[k]() for k in args.strategies.split(",")
                     if k in ALL_STRATEGIES}

assert model_registry, f"No valid models in: {args.models}"
assert strategy_registry, f"No valid strategies in: {args.strategies}"

# ── Data ───────────────────────────────────────────────────────────────────────

# Default resize: 96 for STL-10 (native), 96 for CIFAR (upscale from 32).
# Pass --resize 224 when using Swin-T (required by the architecture).
if args.resize is not None:
    RESIZE = args.resize
elif args.dataset == "stl10":
    RESIZE = None   # native 96×96, no resize needed
else:
    RESIZE = 96

def _make_transform(resize):
    if resize is None:
        return ToTensor()
    return Compose([Resize(resize), ToTensor()])

if args.dataset == "cifar10":
    transform = _make_transform(RESIZE)
    train_full = CIFAR10("./data", train=True, download=True, transform=transform)
    test_full = CIFAR10("./data", train=False, download=True, transform=transform)
    num_classes = 10
elif args.dataset == "cifar100":
    transform = _make_transform(RESIZE)
    train_full = CIFAR100("./data", train=True, download=True, transform=transform)
    test_full = CIFAR100("./data", train=False, download=True, transform=transform)
    num_classes = 100
else:  # stl10
    transform = _make_transform(RESIZE)
    train_full = STL10("./data", split="train", download=True, transform=transform)
    test_full = STL10("./data", split="test", download=True, transform=transform)
    num_classes = 10

print(f"Image size: {RESIZE if RESIZE else 96} (native)")

val_size = int(len(train_full) * args.val_split)
train_size = len(train_full) - val_size
train_raw, val_raw = random_split(train_full, [train_size, val_size])

difficulty_params = dict(
    imbalance_factor=args.imbalance_factor,
    noise_level=args.noise_level,
    artifact_level=args.artifact_level,
    artifact_type=args.artifact_type,
)

class _ListDataset(torch.utils.data.Dataset):
    """Thin wrapper around a Subset that provides stable integer indices
    without materialising all images in RAM (avoids ~27 GB for CIFAR-10@224)."""
    def __init__(self, subset):
        self.subset = subset
    def __len__(self): return len(self.subset)
    def __getitem__(self, i):
        img, label = self.subset[i]
        return img if isinstance(img, torch.Tensor) else torch.tensor(img), int(label)

train_base = _ListDataset(train_raw)
val_base   = _ListDataset(val_raw)

train_dataset = create_dataset(train_base, args.dataset_type, **difficulty_params)

# Val and test use CLEAN (original) data — this measures true generalisation,
# not alignment with artificially corrupted labels or degraded images.
# For 'noisy': a perfect model predicting true labels would score <70% against
# noisy test labels — making strategy comparison meaningless.
val_loader  = DataLoader(val_base,  args.batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_full, args.batch_size, shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset)} | Val: {len(val_base)} | Test: {len(test_full)}")

# ── Helpers ────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, path: str, meta: dict):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **meta
    }, path)


def compute_steps_per_epoch(strategy, dataset, batch_size: int) -> int:
    """Estimate how many gradient steps one epoch gives for this strategy."""
    subset = strategy.get_dataset(dataset) if not isinstance(strategy, Static) else dataset
    return max(1, len(subset) // batch_size)


# ── Main loop ──────────────────────────────────────────────────────────────────

for model_name, model_fn in model_registry.items():
    # Per-model results directory: results/{exp_name}_{model_name}/
    model_exp_name = f"{exp_name}_{model_name}"
    results_dir    = f"results/{model_exp_name}"
    checkpoints_dir = f"checkpoints/{model_exp_name}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    model_rows = []
    model_test_rows = []

    # Fresh curriculum dataset per model (losses reset)
    from datasets.curriculum_dataset import CurriculumDataset
    curr_ds = CurriculumDataset(train_dataset)

    for strat_name, strat in strategy_registry.items():
        print(f"\n{'='*70}")
        print(f"  {model_name.upper()} | {strat_name.upper()}")
        print(f"  Dataset: {args.dataset} ({args.dataset_type}), "
              f"budget_mode={args.budget_mode}")
        print(f"{'='*70}")

        model = model_fn(num_classes)
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        trainer = ClassificationTrainer(model, optimizer, criterion, device)

        ckpt_dir = f"{checkpoints_dir}/{strat_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt_path = f"{ckpt_dir}/best_model.pth"

        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        total_steps = 0
        epoch = 0
        total_time = 0.0

        # ── Static strategy: warm-up pass to initialize difficulty groups ──────
        if strat_name == "static":
            print("  [Static] Running warm-up epoch to initialize difficulty groups...")
            warmup_loader = DataLoader(curr_ds, args.batch_size, shuffle=True, num_workers=0)
            trainer.train_epoch(warmup_loader, curr_ds)
            strat.initialize(curr_ds)
            # Count warm-up steps toward budget so it's still fair
            total_steps += len(warmup_loader)
            print(f"  [Static] Groups initialized. Warm-up steps: {len(warmup_loader)}")

        # ── StageWise: determine stage transition points ───────────────────────
        # Transitions happen at 1/3 and 2/3 of total training budget,
        # regardless of budget_mode. This makes transitions robust.
        stagewise_step_transitions: list[int] = []
        if strat_name == "stagewise" and args.budget_mode == "steps":
            stagewise_step_transitions = [
                args.total_steps // 3,
                2 * args.total_steps // 3,
            ]

        # ── Training loop ──────────────────────────────────────────────────────
        while True:
            epoch_start = time.time()

            # Select subset for this epoch
            if strat_name == "static":
                # Advance group every 1/3 of total budget
                if args.budget_mode == "steps":
                    stage = min(total_steps * 3 // args.total_steps, 2)
                else:
                    stage = min(epoch * 3 // max(args.epochs, 1), 2)
                subset = strat.get_dataset(curr_ds, stage)
            else:
                subset = strat.get_dataset(curr_ds)

            loader = DataLoader(subset, args.batch_size, shuffle=True, num_workers=0)
            steps_this_epoch = len(loader)

            # StageWise transitions in step mode
            if strat_name == "stagewise" and stagewise_step_transitions:
                for threshold in stagewise_step_transitions:
                    if total_steps < threshold <= total_steps + steps_this_epoch:
                        strat.step()
                        print(f"  [StageWise] Advanced to stage {strat.stage} "
                              f"at step {total_steps}")

            # StageWise transitions in epoch mode (at 1/3 and 2/3 of epochs)
            if strat_name == "stagewise" and args.budget_mode == "epochs":
                if epoch in [args.epochs // 3, 2 * args.epochs // 3]:
                    strat.step()
                    print(f"  [StageWise] Advanced to stage {strat.stage} at epoch {epoch}")

            train_loss, train_acc = trainer.train_epoch(loader, curr_ds)
            val_loss, val_acc = trainer.validate(val_loader)

            total_steps += steps_this_epoch
            epoch_time = time.time() - epoch_start
            total_time += epoch_time

            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(model, optimizer, best_ckpt_path, {
                    "epoch": epoch, "val_acc": val_acc,
                    "train_acc": train_acc, "total_steps": total_steps,
                })
                print(f"  ✓ epoch={epoch:3d} | steps={total_steps:6d} | "
                      f"train={train_acc:.4f} | val={val_acc:.4f}  ← best")
            else:
                patience_counter += 1
                if epoch % 5 == 0:
                    print(f"    epoch={epoch:3d} | steps={total_steps:6d} | "
                          f"train={train_acc:.4f} | val={val_acc:.4f} "
                          f"(patience {patience_counter}/{args.patience})")

            model_rows.append({
                "experiment": model_exp_name, "model": model_name,
                "strategy": strat_name, "epoch": epoch,
                "total_steps": total_steps,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "epoch_time": epoch_time, "total_time": total_time,
                "subset_size": len(subset),
                "dataset_type": args.dataset_type,
            })

            epoch += 1

            # ── Stopping conditions ───────────────────────────────────────────
            if args.budget_mode == "steps" and total_steps >= args.total_steps:
                print(f"\n  Budget exhausted ({total_steps} steps).")
                break
            if args.budget_mode == "epochs" and epoch >= args.epochs:
                print(f"\n  Epoch limit reached ({epoch} epochs).")
                break
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement in "
                      f"{args.patience} epochs).")
                break

        # ── Test ──────────────────────────────────────────────────────────────
        print(f"\n  Testing {model_name} / {strat_name} (loading best checkpoint)...")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        test_loss, test_acc = trainer.validate(test_loader)

        print(f"  Test accuracy: {test_acc:.4f} | best val: {best_val_acc:.4f} "
              f"(epoch {best_epoch}) | total steps: {total_steps}")

        model_test_rows.append({
            "experiment": model_exp_name, "model": model_name,
            "strategy": strat_name,
            "dataset_type": args.dataset_type,
            "imbalance_factor": args.imbalance_factor,
            "noise_level": args.noise_level,
            "artifact_level": args.artifact_level,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "best_val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
            "total_steps": total_steps,
            "total_epochs": epoch,
            "total_training_time": total_time,
        })

    # ── Save results (per model) ────────────────────────────────────────────────

    pd.DataFrame(model_rows).to_csv(
        f"{results_dir}/classification_results.csv", index=False)
    pd.DataFrame(model_test_rows).to_csv(
        f"{results_dir}/test_results.csv", index=False)

    df_test = pd.DataFrame(model_test_rows)
    print(f"\n{'='*70}")
    print(f"MODEL COMPLETE: {model_exp_name}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}")

    if not df_test.empty:
        print("\nFINAL TEST RESULTS (sorted by accuracy):")
        for _, row in df_test.sort_values("test_accuracy", ascending=False).iterrows():
            print(f"  {row['model']:15s} | {row['strategy']:15s} | "
                  f"test={row['test_accuracy']:.4f} | "
                  f"val={row['best_val_accuracy']:.4f} | "
                  f"steps={row['total_steps']} | epochs={row['total_epochs']}")
