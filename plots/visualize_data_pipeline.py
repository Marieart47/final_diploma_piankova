"""Визуализация всех типов деградации данных используемых в экспериментах."""

import io
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torchvision.datasets import CIFAR10, STL10
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT = Path(__file__).parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = "./data"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "axes.titlesize": 10, "axes.labelsize": 9,
})

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]
STL10_CLASSES   = ["airplane","bird","car","cat","deer",
                   "dog","horse","monkey","ship","truck"]

# ── helpers ──────────────────────────────────────────────────────────────────

def _ax_off(ax):
    ax.axis("off")

def _border(ax, color, lw=3):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_visible(True)

def _title(ax, t, color="black", size=9):
    ax.set_title(t, color=color, fontsize=size, pad=3)

def to_pil(tensor):
    return TF.to_pil_image(tensor.clamp(0, 1))

def to_np(tensor):
    return np.array(to_pil(tensor))

# ── degradation functions (точно как в коде) ──────────────────────────────────

def apply_blur(pil_img, level):
    return pil_img.filter(ImageFilter.GaussianBlur(radius=1 + level * 4))

def apply_noise(pil_img, level):
    arr = np.array(pil_img).astype(np.float32)
    noise = np.random.normal(0, level * 50, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

def apply_sr_noise(lr_t, level):
    std = level * 0.10
    return (lr_t + torch.randn_like(lr_t) * std).clamp(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# График 1: Визуальные артефакты (blur + noise) на CIFAR-10
# ══════════════════════════════════════════════════════════════════════════════

def plot_visual_artifacts():
    ds = CIFAR10(DATA_ROOT, train=True, download=True,
                 transform=transforms.ToTensor())

    # по одному примеру каждого класса
    class_to_img = {}
    for img, label in ds:
        if label not in class_to_img:
            class_to_img[label] = img
        if len(class_to_img) == 10:
            break

    blur_levels  = [0.0, 0.2, 0.4, 0.6]
    noise_levels = [0.0, 0.2, 0.4, 0.6]

    n_classes = 5            # показываем первые 5 классов
    n_cols_blur  = len(blur_levels)
    n_cols_noise = len(noise_levels)
    n_cols = n_cols_blur + n_cols_noise + 1   # +1 разделитель
    n_rows = n_classes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.35, n_rows * 1.5))

    blur_labels  = ["Original", "Blur 0.2", "Blur 0.4", "Blur 0.6"]
    noise_labels = ["Original", "Noise 0.2", "Noise 0.4", "Noise 0.6"]

    for row in range(n_rows):
        label_id = row
        pil_orig = to_pil(class_to_img[label_id])

        # Blur
        for c, lv in enumerate(blur_levels):
            ax = axes[row][c]
            img = pil_orig if lv == 0.0 else apply_blur(pil_orig, lv)
            ax.imshow(np.array(img), interpolation="nearest")
            _ax_off(ax)
            if row == 0:
                _title(ax, blur_labels[c],
                       color="#1565C0" if lv > 0 else "black", size=8.5)
            if c == 0:
                ax.set_ylabel(CIFAR10_CLASSES[label_id],
                              fontsize=8, rotation=0, labelpad=40,
                              va="center", ha="right")

        # разделитель
        axes[row][n_cols_blur].axis("off")

        # Noise
        for c, lv in enumerate(noise_levels):
            ax = axes[row][n_cols_blur + 1 + c]
            img = pil_orig if lv == 0.0 else apply_noise(pil_orig, lv)
            ax.imshow(np.array(img), interpolation="nearest")
            _ax_off(ax)
            if row == 0:
                _title(ax, noise_labels[c],
                       color="#C62828" if lv > 0 else "black", size=8.5)

    # группирующие надписи
    fig.text(0.22, 1.01, "Gaussian Blur", ha="center", fontsize=12,
             fontweight="bold", color="#1565C0",
             transform=fig.transFigure)
    fig.text(0.72, 1.01, "Pixel Noise", ha="center", fontsize=12,
             fontweight="bold", color="#C62828",
             transform=fig.transFigure)

    fig.suptitle("CIFAR-10: Visual Artifacts at Different Degradation Levels",
                 fontsize=12, fontweight="bold", y=1.05)
    plt.tight_layout(pad=0.3)
    path = OUT / "data_01_visual_artifacts.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# График 2: Шум в метках
# ══════════════════════════════════════════════════════════════════════════════

def plot_noisy_labels():
    ds = CIFAR10(DATA_ROOT, train=True, download=True,
                 transform=transforms.ToTensor())

    noise_level = 0.6
    random.seed(42)

    # берём 10 изображений разных классов
    samples = []
    seen = set()
    for img, label in ds:
        if label not in seen:
            seen.add(label)
            noisy = label
            if random.random() < noise_level:
                others = [i for i in range(10) if i != label]
                noisy = random.choice(others)
            samples.append((img, label, noisy))
        if len(samples) == 10:
            break

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 3.2))

    for i, (img, true_lbl, noisy_lbl) in enumerate(samples):
        pil = to_pil(img)
        corrupted = (true_lbl != noisy_lbl)

        # верхний ряд — оригинал с правильной меткой
        axes[0][i].imshow(np.array(pil), interpolation="nearest")
        _ax_off(axes[0][i])
        _title(axes[0][i], CIFAR10_CLASSES[true_lbl], color="black")
        _border(axes[0][i], "#2E7D32", lw=2.5)

        # нижний ряд — та же картинка с возможно зашумлённой меткой
        axes[1][i].imshow(np.array(pil), interpolation="nearest")
        _ax_off(axes[1][i])
        color = "#C62828" if corrupted else "#2E7D32"
        marker = "✗" if corrupted else "✓"
        _title(axes[1][i],
               f"{marker} {CIFAR10_CLASSES[noisy_lbl]}",
               color=color)
        _border(axes[1][i], color, lw=2.5)

    axes[0][0].set_ylabel("True\nLabel", fontsize=9, rotation=0,
                          labelpad=55, va="center")
    axes[1][0].set_ylabel(f"Noisy Label\n(p={noise_level})", fontsize=9,
                          rotation=0, labelpad=55, va="center")

    green_patch = mpatches.Patch(color="#2E7D32", label="Correct label")
    red_patch   = mpatches.Patch(color="#C62828", label="Noisy label")
    fig.legend(handles=[green_patch, red_patch], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle(f"Label Noise (noise_level={noise_level}): image is unchanged, label may be replaced",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(pad=0.4)
    path = OUT / "data_02_noisy_labels.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# График 3: Дисбаланс классов
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_imbalance():
    ds = CIFAR10(DATA_ROOT, train=True, download=True,
                 transform=transforms.ToTensor())

    labels_all = [ds[i][1] for i in range(len(ds))]
    counts_orig = np.bincount(labels_all, minlength=10)

    rare_classes   = [0, 1, 2]
    imb_factors    = [0.3, 0.6]
    colors_orig = ["#C62828" if i in rare_classes else "#1565C0" for i in range(10)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    # оригинал
    ax = axes[0]
    bars = ax.bar(range(10), counts_orig, color=colors_orig, edgecolor="white", alpha=0.9)
    ax.set_title("Original (Balanced)", fontsize=11, fontweight="bold")
    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Number of images")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for bar, cnt in zip(bars, counts_orig):
        ax.text(bar.get_x() + bar.get_width()/2, cnt + 30,
                str(cnt), ha="center", va="bottom", fontsize=7)

    for ax_i, factor in enumerate(imb_factors):
        counts_imb = counts_orig.copy().astype(float)
        for c in rare_classes:
            counts_imb[c] = int(counts_orig[c] * (1 - factor))

        ax = axes[ax_i + 1]
        bars = ax.bar(range(10), counts_imb, color=colors_orig, edgecolor="white", alpha=0.9)
        ax.set_title(f"Imbalance factor={factor}\n(rare classes: {[CIFAR10_CLASSES[c] for c in rare_classes]})",
                     fontsize=10, fontweight="bold")
        ax.set_xticks(range(10))
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Number of images")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for bar, cnt in zip(bars, counts_imb):
            ax.text(bar.get_x() + bar.get_width()/2, cnt + 30,
                    f"{int(cnt)}", ha="center", va="bottom", fontsize=7)

        # горизонтальная линия — уровень оригинала для редких
        orig_rare = counts_orig[rare_classes[0]]
        ax.axhline(orig_rare, color="#C62828", linestyle="--", alpha=0.6, linewidth=1.5,
                   label=f"Original level ({orig_rare})")
        ax.legend(fontsize=8)

    red_p  = mpatches.Patch(color="#C62828", alpha=0.9, label=f"Rare classes ({', '.join(CIFAR10_CLASSES[c] for c in rare_classes)})")
    blue_p = mpatches.Patch(color="#1565C0", alpha=0.9, label="Common classes")
    fig.legend(handles=[red_p, blue_p], loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("CIFAR-10: Artificial Class Imbalance",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(pad=1.0)
    path = OUT / "data_03_class_imbalance.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# График 4: SR-пайплайн — HR → LR → деградация → восстановление
# ══════════════════════════════════════════════════════════════════════════════

def plot_sr_pipeline():
    # берём реальные изображения из STL-10 SR данных
    sr_dir = Path("data/sr_hr/stl10/train")
    img_paths = sorted(sr_dir.glob("*.png"))[:8]

    scale  = 2
    levels = [0.0, 0.5]   # original / noise

    n_imgs = min(4, len(img_paths))
    if n_imgs == 0:
        print("STL-10 SR data not found, skipping SR pipeline plot")
        return

    fig, axes = plt.subplots(n_imgs, 3, figsize=(7.5, n_imgs * 2.6))
    col_titles = ["HR (96×96)", "LR original (48×48)", "LR + noise 0.5 (48×48)"]
    colors_t   = ["#2E7D32", "#1565C0", "#C62828"]

    for row, path in enumerate(img_paths[:n_imgs]):
        hr_pil = Image.open(path).convert("RGB")
        hr_t   = TF.to_tensor(hr_pil)

        lr_pil  = hr_pil.resize((hr_pil.width // scale, hr_pil.height // scale), Image.BILINEAR)
        lr_t    = TF.to_tensor(lr_pil)
        lr_noisy = apply_sr_noise(lr_t, 0.5)

        imgs  = [hr_t, lr_t, lr_noisy]
        sizes = [f"{hr_pil.width}×{hr_pil.height}",
                 f"{lr_pil.width}×{lr_pil.height}",
                 f"{lr_pil.width}×{lr_pil.height}"]

        for col, (img_t, sz) in enumerate(zip(imgs, sizes)):
            ax = axes[row][col]
            ax.imshow(to_np(img_t), interpolation="nearest")
            _ax_off(ax)
            _border(ax, colors_t[col], lw=2)
            if row == 0:
                _title(ax, f"{col_titles[col]}\n({sz})", color=colors_t[col], size=8.5)

    for x_frac in [0.36, 0.67]:
        fig.text(x_frac, 0.5, "→", ha="center", va="center",
                 fontsize=18, color="gray", transform=fig.transFigure)

    fig.suptitle("SR Pipeline: HR → LR (bilinear ×2) → LR degradation (Gaussian noise)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.3, rect=[0, 0, 1, 0.97])
    path_out = OUT / "data_04_sr_pipeline.png"
    plt.savefig(path_out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_out}")


# ══════════════════════════════════════════════════════════════════════════════
# График 5: Сводная схема — все деградации на одном листе (STL-10)
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary_grid():
    ds = STL10(DATA_ROOT, split="train", download=True,
               transform=transforms.ToTensor())

    # 4 класса × 1 изображение
    class_imgs = {}
    for img, label in ds:
        if label not in class_imgs:
            class_imgs[label] = img
        if len(class_imgs) == 4:
            break

    selected = list(class_imgs.items())[:4]  # (label, tensor)

    # колонки: original | blur0.4 | blur0.6 | noise0.4 | noise0.6
    configs = [
        ("Original",   None,    None),
        ("Blur\n0.4",  "blur",  0.4),
        ("Blur\n0.6",  "blur",  0.6),
        ("Noise\n0.4", "noise", 0.4),
        ("Noise\n0.6", "noise", 0.6),
    ]
    col_colors = ["black", "#1565C0", "#1565C0", "#C62828", "#C62828"]

    n_rows = len(selected)
    n_cols = len(configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.7))

    for row, (lbl, img_t) in enumerate(selected):
        pil_orig = to_pil(img_t)
        for col, (title, atype, alevel) in enumerate(configs):
            ax = axes[row][col]
            if atype is None:
                pil = pil_orig
            elif atype == "blur":
                pil = apply_blur(pil_orig, alevel)
            else:
                pil = apply_noise(pil_orig, alevel)

            ax.imshow(np.array(pil), interpolation="nearest")
            _ax_off(ax)
            _border(ax, col_colors[col], lw=2)
            if row == 0:
                _title(ax, title, color=col_colors[col], size=9)
            if col == 0:
                ax.set_ylabel(STL10_CLASSES[lbl], fontsize=9,
                              rotation=0, labelpad=42, va="center", ha="right")

    fig.suptitle("STL-10: All Types of Visual Degradation (as in experiments)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.3, rect=[0, 0, 1, 0.97])
    path = OUT / "data_05_stl10_summary.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# График 6: Curriculum Learning — схема сортировки по сложности
# ══════════════════════════════════════════════════════════════════════════════

def plot_curriculum_concept():
    ds = CIFAR10(DATA_ROOT, train=True, download=True,
                 transform=transforms.ToTensor())

    # симулируем loss: чистые изображения → низкий лосс, зашумлённые → высокий
    random.seed(42)
    np.random.seed(42)

    samples = []
    seen_labels = set()
    for img, label in ds:
        if label not in seen_labels:
            seen_labels.add(label)
            samples.append((img, label))
        if len(samples) == 10:
            break

    # создаём 4 "уровня сложности" с разным шумом
    levels = [
        ("Easy\n(low loss)",   0.0,  0.05),
        ("Medium",             0.35, 0.18),
        ("Hard\n(high loss)",  0.70, 0.45),
    ]

    n_imgs_per_level = 5
    n_rows = len(levels)
    n_cols = n_imgs_per_level + 2  # +2: заголовок + loss bar

    fig = plt.figure(figsize=(14, n_rows * 2.2))
    gs  = GridSpec(n_rows, n_cols, figure=fig,
                   wspace=0.1, hspace=0.35,
                   left=0.01, right=0.99)

    level_colors = ["#2E7D32", "#F57F17", "#C62828"]

    for row, (level_name, noise_lv, loss_val) in enumerate(levels):
        color = level_colors[row]

        # label
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.set_xlim(0, 1); ax_lbl.set_ylim(0, 1)
        ax_lbl.text(0.5, 0.55, level_name, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color,
                    transform=ax_lbl.transAxes)
        ax_lbl.text(0.5, 0.2, f"loss ≈ {loss_val:.2f}", ha="center", va="center",
                    fontsize=8.5, color="gray", transform=ax_lbl.transAxes)
        ax_lbl.axis("off")

        # изображения
        for col in range(n_imgs_per_level):
            img_t, lbl = samples[col]
            pil = to_pil(img_t)
            if noise_lv > 0:
                pil = apply_noise(pil, noise_lv)
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(np.array(pil), interpolation="nearest")
            ax.axis("off")
            _border(ax, color, lw=2.5)
            ax.set_title(CIFAR10_CLASSES[lbl], fontsize=7, pad=1)

        # loss bar
        ax_bar = fig.add_subplot(gs[row, -1])
        bar_h  = 0.5
        ax_bar.barh(0, loss_val, bar_h, color=color, alpha=0.85)
        ax_bar.set_xlim(0, 0.7)
        ax_bar.set_ylim(-0.5, 0.5)
        ax_bar.set_xlabel("Loss", fontsize=8)
        ax_bar.set_yticks([])
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_visible(False)
        ax_bar.text(loss_val + 0.02, 0, f"{loss_val:.2f}",
                    va="center", fontsize=9, color=color, fontweight="bold")

    # стрелка "порядок подачи"
    fig.text(0.5, -0.01, "Feeding Order in Stagewise CL: Easy → Medium → Hard",
             ha="center", fontsize=11, fontweight="bold",
             color="#1565C0",
             bbox=dict(boxstyle="rarrow,pad=0.4", fc="#E3F2FD", ec="#1565C0", lw=1.5))

    fig.suptitle("Curriculum Learning Concept: Sorting Examples by Difficulty (Loss)",
                 fontsize=13, fontweight="bold", y=1.02)
    path = OUT / "data_06_curriculum_concept.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating dataset visualization plots...\n")
    plot_visual_artifacts()
    plot_noisy_labels()
    plot_class_imbalance()
    plot_sr_pipeline()
    plot_summary_grid()
    plot_curriculum_concept()
    print(f"\nAll plots saved to: {OUT}")
