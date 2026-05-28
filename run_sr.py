"""Запуск экспериментов по курикулум-обучению для задачи Super Resolution."""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from curriculum.strategies import Baseline, StageWise, Static, Online, AntiCurriculum
from datasets.curriculum_dataset import CurriculumDataset
from datasets.sr_dataset import SRDataset, DEGRADATION_TYPES
from models.sr_models import build_sr_model
from training.sr_trainer import SRTrainer

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Curriculum Learning для Super Resolution")
parser.add_argument("--hr_train_dir", required=True,
                    help="Папка с HR-изображениями для обучения")
parser.add_argument("--hr_val_dir", default=None,
                    help="Папка с HR для валидации. Если не указана — val_split от train")
parser.add_argument("--dataset_name", default="div2k",
                    help="Имя датасета (div2k, flickr2k, celeba, ...)")
parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4],
                    help="Коэффициент апсемплинга")
parser.add_argument("--patch_size", type=int, default=128,
                    help="Размер HR-патча при обучении")
parser.add_argument("--degradation", default="original", choices=list(DEGRADATION_TYPES),
                    help="Тип деградации LR: original, noise, blur, jpeg, mixed")
parser.add_argument("--deg_level", type=float, default=0.5,
                    help="Уровень деградации [0.0, 1.0]. Игнорируется при degradation=original")
parser.add_argument("--archs", default="srcnn,espcn,swinir",
                    help="Архитектуры через запятую: srcnn, espcn, swinir")
parser.add_argument("--strategies", default="baseline,stagewise,static,online",
                    help="Стратегии через запятую: baseline, stagewise, static, online, anticurriculum")
parser.add_argument("--total_steps", type=int, default=10000,
                    help="Бюджет в градиентных шагах на один прогон (arch × strategy)")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--val_split", type=float, default=0.1,
                    help="Доля train для валидации (если --hr_val_dir не задан)")
parser.add_argument("--patience", type=int, default=20,
                    help="Ранняя остановка: эпох без улучшения PSNR")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ── Setup ──────────────────────────────────────────────────────────────────────

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed(args.seed)
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")

# Имя эксперимента: {dataset}_x{scale}_{deg_type}[_deg{level}]
if args.degradation == "original":
    deg_suffix = "original"
else:
    deg_suffix = f"{args.degradation}_deg{args.deg_level}"

# ── Стратегии и архитектуры ────────────────────────────────────────────────────

ALL_STRATEGIES = {
    "baseline":       lambda: Baseline(),
    "stagewise":      lambda: StageWise(),
    "static":         lambda: Static(),
    "online":         lambda: Online(pct=0.3),
    "anticurriculum": lambda: AntiCurriculum(pct=0.3),
}

arch_list     = [a.strip() for a in args.archs.split(",")]
strategy_list = [s.strip() for s in args.strategies.split(",") if s.strip() in ALL_STRATEGIES]

assert arch_list,     f"Нет валидных архитектур: {args.archs}"
assert strategy_list, f"Нет валидных стратегий: {args.strategies}"

# ── Данные ─────────────────────────────────────────────────────────────────────

deg_kwargs = dict(degradation=args.degradation, deg_level=args.deg_level)

train_ds_full = SRDataset(args.hr_train_dir, scale=args.scale,
                          patch_size=args.patch_size, training=True, **deg_kwargs)

if args.hr_val_dir:
    val_ds        = SRDataset(args.hr_val_dir, scale=args.scale,
                              patch_size=args.patch_size, training=False, **deg_kwargs)
    train_ds_base = train_ds_full
else:
    val_size   = max(1, int(len(train_ds_full) * args.val_split))
    train_size = len(train_ds_full) - val_size
    train_sub, val_sub = random_split(train_ds_full, [train_size, val_size])

    class _SubsetDS(torch.utils.data.Dataset):
        def __init__(self, s): self.s = s
        def __len__(self): return len(self.s)
        def __getitem__(self, i): return self.s[i]

    train_ds_base = _SubsetDS(train_sub)
    val_ds_full   = SRDataset(args.hr_train_dir, scale=args.scale,
                               patch_size=args.patch_size, training=False, **deg_kwargs)
    val_ds = torch.utils.data.Subset(val_ds_full, val_sub.indices)

val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds_base)} | Val: {len(val_ds)} | "
      f"Scale: x{args.scale} | Patch: {args.patch_size} | "
      f"Degradation: {args.degradation} (level={args.deg_level})")

# ── Helpers ────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, path, meta):
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **meta,
    }, path)

# ── Главный цикл: архитектура × стратегия ─────────────────────────────────────

for arch in arch_list:
    exp_name    = f"{args.dataset_name}_x{args.scale}_{deg_suffix}_{arch}"
    results_dir = f"results/{exp_name}"
    ckpt_root   = f"checkpoints/{exp_name}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_root,   exist_ok=True)

    curr_ds    = CurriculumDataset(train_ds_base)
    epoch_rows = []
    test_rows  = []

    for strat_name in strategy_list:
        strat = ALL_STRATEGIES[strat_name]()

        print(f"\n{'='*70}")
        print(f"  {arch.upper()} | {strat_name.upper()}")
        print(f"  {args.dataset_name} x{args.scale} | deg={args.degradation}({args.deg_level}) | "
              f"budget={args.total_steps} steps")
        print(f"{'='*70}")

        model     = build_sr_model(arch, scale=args.scale)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        trainer   = SRTrainer(model, optimizer, device)

        ckpt_dir  = f"{ckpt_root}/{strat_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt = f"{ckpt_dir}/best_model.pth"

        best_psnr    = 0.0
        best_epoch   = 0
        patience_ctr = 0
        total_steps  = 0
        total_time   = 0.0
        epoch        = 0

        # Static: warm-up для инициализации групп сложности
        if strat_name == "static":
            print("  [Static] Warm-up epoch...")
            wu_loader = DataLoader(curr_ds, args.batch_size, shuffle=True, num_workers=0)
            trainer.train_epoch(wu_loader, curr_ds)
            strat.initialize(curr_ds)
            total_steps += len(wu_loader)
            print(f"  [Static] Groups ready. Warm-up: {len(wu_loader)} steps")

        # StageWise: точки переключения на 1/3 и 2/3 бюджета
        stagewise_transitions: list[int] = []
        if strat_name == "stagewise":
            stagewise_transitions = [args.total_steps // 3, 2 * args.total_steps // 3]

        # Обучение
        while True:
            t0 = time.time()

            if strat_name == "static":
                stage  = min(total_steps * 3 // args.total_steps, 2)
                subset = strat.get_dataset(curr_ds, stage)
            else:
                subset = strat.get_dataset(curr_ds)

            loader           = DataLoader(subset, args.batch_size, shuffle=True, num_workers=0)
            steps_this_epoch = len(loader)

            if strat_name == "stagewise":
                for thr in stagewise_transitions:
                    if total_steps < thr <= total_steps + steps_this_epoch:
                        strat.step()
                        print(f"  [StageWise] Stage {strat.stage} at step {total_steps}")

            train_loss, train_psnr = trainer.train_epoch(loader, curr_ds)
            val_loss,   val_psnr   = trainer.validate(val_loader)

            total_steps += steps_this_epoch
            epoch_time   = time.time() - t0
            total_time  += epoch_time

            if val_psnr > best_psnr:
                best_psnr    = val_psnr
                best_epoch   = epoch
                patience_ctr = 0
                save_checkpoint(model, optimizer, best_ckpt, {
                    "epoch": epoch, "val_psnr": val_psnr,
                    "train_psnr": train_psnr, "total_steps": total_steps,
                })
                print(f"  * epoch={epoch:3d} | steps={total_steps:6d} | "
                      f"train={train_psnr:.2f} dB | val={val_psnr:.2f} dB  <- best")
            else:
                patience_ctr += 1
                if epoch % 5 == 0:
                    print(f"    epoch={epoch:3d} | steps={total_steps:6d} | "
                          f"train={train_psnr:.2f} dB | val={val_psnr:.2f} dB "
                          f"(patience {patience_ctr}/{args.patience})")

            epoch_rows.append({
                "arch":        arch,
                "strategy":    strat_name,
                "dataset":     args.dataset_name,
                "scale":       args.scale,
                "degradation": args.degradation,
                "deg_level":   args.deg_level,
                "epoch":       epoch,
                "total_steps": total_steps,
                "train_loss":  train_loss,
                "train_psnr":  train_psnr,
                "val_loss":    val_loss,
                "val_psnr":    val_psnr,
                "subset_size": len(subset),
                "epoch_time":  epoch_time,
                "total_time":  total_time,
            })

            epoch += 1

            if total_steps >= args.total_steps:
                print(f"\n  Бюджет исчерпан ({total_steps} шагов).")
                break
            if patience_ctr >= args.patience:
                print(f"\n  Ранняя остановка на эпохе {epoch}.")
                break

        # Тест на лучшем чекпоинте
        print(f"\n  Test {arch}/{strat_name}...")
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loss, test_psnr = trainer.validate(val_loader)

        print(f"  Test PSNR: {test_psnr:.2f} dB | best val: {best_psnr:.2f} dB "
              f"(epoch {best_epoch}) | steps: {total_steps}")

        test_rows.append({
            "arch":          arch,
            "strategy":      strat_name,
            "dataset":       args.dataset_name,
            "scale":         args.scale,
            "degradation":   args.degradation,
            "deg_level":     args.deg_level,
            "test_psnr":     test_psnr,
            "test_loss":     test_loss,
            "best_val_psnr": best_psnr,
            "best_epoch":    best_epoch,
            "total_steps":   total_steps,
            "total_epochs":  epoch,
            "total_time":    total_time,
        })

    pd.DataFrame(epoch_rows).to_csv(f"{results_dir}/sr_results.csv",      index=False)
    pd.DataFrame(test_rows).to_csv(f"{results_dir}/sr_test_results.csv",   index=False)

    df = pd.DataFrame(test_rows)
    print(f"\n{'='*70}")
    print(f"ARCH COMPLETE: {exp_name}")
    print(f"{'='*70}")
    if not df.empty:
        print("\nFINAL TEST PSNR:")
        for _, row in df.sort_values("test_psnr", ascending=False).iterrows():
            print(f"  {row['arch']:8s} | {row['strategy']:15s} | "
                  f"PSNR={row['test_psnr']:.2f} dB | "
                  f"val={row['best_val_psnr']:.2f} dB | "
                  f"steps={row['total_steps']}")
