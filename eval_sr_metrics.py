"""
Переоценка всех сохранённых SR-моделей с метриками PSNR, SSIM, LPIPS.
Читает чекпоинты из checkpoints/, тестирует на val-датасете,
дописывает колонки ssim/lpips в sr_test_results.csv.
"""

import os, sys
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_msssim import ssim as ssim_fn
import lpips

sys.path.insert(0, str(Path(__file__).parent))
from datasets.sr_dataset import SRDataset
from models.sr_models import build_sr_model

# ── конфиг ────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    dict(name="stl10_x2_original",       dataset="stl10", scale=2, deg="original", lvl=0.0,
         val_dir="data/sr_hr/stl10/val",  archs=["srcnn","espcn","swinir"]),
    dict(name="stl10_x2_noise_deg0.5",    dataset="stl10", scale=2, deg="noise",    lvl=0.5,
         val_dir="data/sr_hr/stl10/val",  archs=["srcnn","espcn","swinir"]),
    dict(name="div2k_x4_original",        dataset="div2k", scale=4, deg="original", lvl=0.0,
         val_dir="data/sr_hr/div2k/val",  archs=["srcnn","espcn","swinir"]),
    dict(name="div2k_x4_noise_deg0.5",    dataset="div2k", scale=4, deg="noise",    lvl=0.5,
         val_dir="data/sr_hr/div2k/val",  archs=["srcnn","espcn"]),
]

STRATEGIES = ["baseline", "stagewise", "static", "online"]

device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# LPIPS использует сеть AlexNet
lpips_fn = lpips.LPIPS(net="alex").to(device)
lpips_fn.eval()


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_psnr(pred, target):
    pred = pred.clamp(0, 1); target = target.clamp(0, 1)
    mse = ((pred - target) ** 2).mean(dim=[1,2,3])
    return (10.0 * torch.log10(1.0 / (mse + 1e-8))).mean().item()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    psnr_sum = ssim_sum = lpips_sum = n = 0

    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr).clamp(0, 1)

        bs = lr.size(0)
        psnr_sum  += compute_psnr(sr, hr) * bs
        ssim_sum  += ssim_fn(sr, hr, data_range=1.0, size_average=True).item() * bs
        # LPIPS ожидает [-1, 1]
        lpips_sum += lpips_fn(sr * 2 - 1, hr * 2 - 1).mean().item() * bs
        n += bs

    return psnr_sum/n, ssim_sum/n, lpips_sum/n


# ── main ──────────────────────────────────────────────────────────────────────

for exp in EXPERIMENTS:
    val_ds = SRDataset(exp["val_dir"], scale=exp["scale"],
                       patch_size=128, training=False,
                       degradation=exp["deg"], deg_level=exp["lvl"])
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"\n{'='*60}")
    print(f"  {exp['name']} | val={len(val_ds)} images")
    print(f"{'='*60}")

    rows = []
    for arch in exp["archs"]:
        for strat in STRATEGIES:
            ckpt_path = Path(f"checkpoints/{exp['name']}_{arch}/{strat}/best_model.pth")
            if not ckpt_path.exists():
                print(f"  SKIP {arch}/{strat} — checkpoint not found")
                continue

            model = build_sr_model(arch, scale=exp["scale"]).to(device)
            ckpt  = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

            psnr_v, ssim_v, lpips_v = evaluate(model, loader)
            print(f"  {arch:8s} | {strat:10s} | PSNR={psnr_v:.2f} dB | SSIM={ssim_v:.4f} | LPIPS={lpips_v:.4f}")

            rows.append(dict(
                arch=arch, strategy=strat,
                dataset=exp["dataset"], scale=exp["scale"],
                degradation=exp["deg"], deg_level=exp["lvl"],
                psnr=round(psnr_v, 4),
                ssim=round(ssim_v, 4),
                lpips=round(lpips_v, 4),
            ))

    if rows:
        out_path = Path(f"results/{exp['name']}_metrics_extended.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")

print("\nDone.")
