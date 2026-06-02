"""Re-evaluate all saved SR checkpoints with PSNR, SSIM, and LPIPS.
Appends ssim / lpips columns to each results/*/sr_test_results.csv in-place.

DIV2K images are 2K resolution — for CPU speed we center-crop to DIV2K_CROP×DIV2K_CROP
and evaluate on DIV2K_MAX_IMAGES images only (standard SR evaluation practice).
"""

import sys
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as TF
from PIL import Image
from pytorch_msssim import ssim as ssim_fn
import lpips as lpips_lib

sys.path.insert(0, str(Path(__file__).parent))
from datasets.sr_dataset import SRDataset
from models.sr_models import build_sr_model

# ── Config ────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    dict(name="stl10_x2_original",     dataset="stl10", scale=2, deg="original", lvl=0.0,
         val_dir="data/sr_hr/stl10/val",  archs=["srcnn", "espcn", "swinir"]),
    dict(name="stl10_x2_noise_deg0.5", dataset="stl10", scale=2, deg="noise",    lvl=0.5,
         val_dir="data/sr_hr/stl10/val",  archs=["srcnn", "espcn", "swinir"]),
    dict(name="stl10_x2_blur_deg0.5",  dataset="stl10", scale=2, deg="blur",     lvl=0.5,
         val_dir="data/sr_hr/stl10/val",  archs=["srcnn", "espcn", "swinir"]),
    dict(name="div2k_x4_original",     dataset="div2k", scale=4, deg="original", lvl=0.0,
         val_dir="data/sr_hr/div2k/val",  archs=["srcnn", "espcn", "swinir"]),
    dict(name="div2k_x4_noise_deg0.5", dataset="div2k", scale=4, deg="noise",    lvl=0.5,
         val_dir="data/sr_hr/div2k/val",  archs=["srcnn", "espcn", "swinir"]),
    dict(name="div2k_x4_blur_deg0.5",  dataset="div2k", scale=4, deg="blur",     lvl=0.5,
         val_dir="data/sr_hr/div2k/val",  archs=["srcnn", "espcn", "swinir"]),
]

STRATEGIES       = ["baseline", "stagewise", "static", "online"]
CKPT_DIR         = Path("checkpoints")
RESULTS_DIR      = Path("results")
BATCH_SIZE       = 4
DIV2K_CROP       = 256   # HR center-crop size for DIV2K (LR = 64×64 at ×4)
DIV2K_MAX_IMAGES = 20    # use first N images from DIV2K val set

# ── Device ────────────────────────────────────────────────────────────────────

device = "cpu"
print(f"Device: {device}\n")

lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
lpips_fn.eval()

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred   = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])
    return (10.0 * torch.log10(1.0 / (mse + 1e-8))).mean().item()


class _Div2kCropDataset(torch.utils.data.Dataset):
    """DIV2K eval: center-crop to fixed size, limit to max_images."""
    _EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(self, hr_dir, scale, degradation, deg_level, crop, max_images):
        from datasets.sr_dataset import _apply_degradation as _deg
        self._deg      = _deg
        self.scale     = scale
        self.deg_type  = degradation
        self.deg_level = deg_level
        self.crop      = (crop // scale) * scale  # align to scale
        paths = sorted(p for p in Path(hr_dir).rglob("*")
                       if p.suffix.lower() in self._EXTS)
        self.paths = paths[:max_images]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        hr = Image.open(self.paths[idx]).convert("RGB")
        w, h = hr.size
        left = (w - self.crop) // 2
        top  = (h - self.crop) // 2
        hr   = hr.crop((left, top, left + self.crop, top + self.crop))
        lr   = hr.resize((self.crop // self.scale, self.crop // self.scale), Image.BILINEAR)
        lr_t = TF.to_tensor(lr)
        hr_t = TF.to_tensor(hr)
        if self.deg_type != "original" and self.deg_level > 0.0:
            lr_t = self._deg(lr_t, self.deg_type, self.deg_level)
        return lr_t, hr_t


def make_loader(exp):
    """Build DataLoader; for DIV2K use fixed center-crop + limit images."""
    if exp["dataset"] == "div2k":
        ds = _Div2kCropDataset(
            exp["val_dir"], exp["scale"], exp["deg"], exp["lvl"],
            crop=DIV2K_CROP, max_images=DIV2K_MAX_IMAGES,
        )
        print(f"  [DIV2K] {len(ds)} images, {DIV2K_CROP}×{DIV2K_CROP} center-crop")
    else:
        ds = SRDataset(
            str(Path(exp["val_dir"])), scale=exp["scale"],
            degradation=exp["deg"], deg_level=exp["lvl"],
            training=False, augment=False,
        )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=False)


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
        lpips_sum += lpips_fn(sr * 2 - 1, hr * 2 - 1).mean().item() * bs
        n += bs

    return psnr_sum / n, ssim_sum / n, lpips_sum / n


# ── Main ──────────────────────────────────────────────────────────────────────

for exp in EXPERIMENTS:
    name    = exp["name"]
    val_dir = Path(exp["val_dir"])

    if not val_dir.exists():
        print(f"[SKIP] val dir not found: {val_dir}")
        continue

    ldr = make_loader(exp)

    for arch in exp["archs"]:
        csv_path = RESULTS_DIR / f"{name}_{arch}" / "sr_test_results.csv"
        if not csv_path.exists():
            print(f"[SKIP] no CSV: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "ssim" in df.columns and "lpips" in df.columns:
            print(f"[SKIP] already evaluated: {csv_path.parent.name}")
            continue

        print(f"\n── {name} / {arch} ──")
        rows_updated = 0

        for strat in STRATEGIES:
            ckpt = CKPT_DIR / f"{name}_{arch}" / strat / "best_model.pth"
            if not ckpt.exists():
                print(f"  [missing ckpt] {strat}")
                continue

            model = build_sr_model(arch, scale=exp["scale"]).to(device)
            ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
            state = ckpt_data.get("model_state_dict", ckpt_data)
            model.load_state_dict(state)

            psnr, ssim, lpips_val = evaluate(model, ldr)
            print(f"  {strat:10s} | PSNR={psnr:.3f} | SSIM={ssim:.4f} | LPIPS={lpips_val:.4f}")

            mask = df["strategy"] == strat
            df.loc[mask, "ssim"]  = round(ssim,      4)
            df.loc[mask, "lpips"] = round(lpips_val, 4)
            rows_updated += mask.sum()

        if rows_updated:
            df.to_csv(csv_path, index=False)
            print(f"  → saved {csv_path}")

print("\nDone.")
