"""Visualization of Faster R-CNN / VOC object detection experiment results."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT = Path(__file__).parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

STRATEGY_ORDER  = ["baseline", "stagewise", "static", "online"]
STRATEGY_LABELS = {
    "baseline":  "Baseline",
    "stagewise": "Stagewise CL",
    "static":    "Static CL",
    "online":    "Online CL",
}
STRATEGY_COLORS = {
    "baseline":  "#2196F3",
    "stagewise": "#4CAF50",
    "static":    "#FF9800",
    "online":    "#E91E63",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

FOLDER_META = {
    "voc_original_fasterrcnn":            {"label": "Original",           "group": "main",      "sort": 0},
    "voc_noisy_noise0.3_fasterrcnn":      {"label": "Noisy\n(p=0.3)",     "group": "main",      "sort": 1},
    "voc_noisy_noise0.6_fasterrcnn":      {"label": "Noisy\n(p=0.6)",     "group": "main",      "sort": 2},
    "voc_imbalance_im0.3_fasterrcnn":     {"label": "Imbalance\n(f=0.3)", "group": "main",      "sort": 3},
    "voc_imbalance_im0.6_fasterrcnn":     {"label": "Imbalance\n(f=0.6)", "group": "main",      "sort": 4},
    "voc_artifacts_blur0.4_fasterrcnn":   {"label": "Blur\n(α=0.4)",      "group": "artifacts", "sort": 0},
    "voc_artifacts_noise0.4_fasterrcnn":  {"label": "Noise art.\n(α=0.4)","group": "artifacts", "sort": 1},
}


def load_all():
    rows = []
    for folder, meta in FOLDER_META.items():
        csv = RESULTS_DIR / folder / "detection_results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df["exp_label"] = meta["label"]
        df["exp_group"] = meta["group"]
        df["sort_key"]  = meta["sort"]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _axis_style(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Plot 1: mAP@50 overview — all experiments ─────────────────────────────────

def plot_map50_overview(df):
    main = df[df["exp_group"] == "main"].copy()
    main = main.sort_values("sort_key")
    labels = main["exp_label"].unique()

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(labels))
    n, w = len(STRATEGY_ORDER), 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    for i, strat in enumerate(STRATEGY_ORDER):
        heights = []
        for lbl in labels:
            row = main[(main["exp_label"] == lbl) & (main["strategy"] == strat)]
            heights.append(float(row["map50"].values[0]) if not row.empty else 0.0)
        bars = ax.bar(x + offsets[i], heights, w,
                      label=STRATEGY_LABELS[strat],
                      color=STRATEGY_COLORS[strat], alpha=0.85, edgecolor="white")
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=10)
    ax.set_ylabel("mAP@50")
    ax.set_title("Faster R-CNN: mAP@50 across Dataset Conditions", fontweight="bold")
    ax.legend(fontsize=9)
    _axis_style(ax)

    plt.tight_layout()
    out = OUT / "frcnn_01_map50_overview.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 2: CL gain over baseline ─────────────────────────────────────────────

def plot_cl_gain(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    groups = [("main", "Main Conditions"), ("artifacts", "Artifact Conditions")]

    for ax, (grp, title) in zip(axes, groups):
        sub = df[df["exp_group"] == grp].sort_values("sort_key")
        labels = sub["exp_label"].unique()
        cl_strats = ["stagewise", "static", "online"]
        x = np.arange(len(labels))
        n, w = len(cl_strats), 0.22
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

        for i, strat in enumerate(cl_strats):
            deltas = []
            for lbl in labels:
                base = sub[(sub["exp_label"] == lbl) & (sub["strategy"] == "baseline")]
                cl   = sub[(sub["exp_label"] == lbl) & (sub["strategy"] == strat)]
                if not base.empty and not cl.empty:
                    deltas.append(float(cl["map50"].values[0]) - float(base["map50"].values[0]))
                else:
                    deltas.append(0.0)
            bars = ax.bar(x + offsets[i], deltas, w,
                          label=STRATEGY_LABELS[strat],
                          color=STRATEGY_COLORS[strat], alpha=0.85, edgecolor="white")
            for bar, d in zip(bars, deltas):
                va = "bottom" if d >= 0 else "top"
                offset = 0.001 if d >= 0 else -0.001
                ax.text(bar.get_x() + bar.get_width() / 2, d + offset,
                        f"{d:+.3f}", ha="center", va=va, fontsize=7.5)

        ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=9)
        ax.set_ylabel("ΔmAP@50 vs Baseline")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        _axis_style(ax)

    fig.suptitle("Faster R-CNN: CL Gain over Baseline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT / "frcnn_02_cl_gain.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 3: Heatmap — mAP@50 по всем конфигурациям ───────────────────────────

def plot_heatmap(df):
    all_labels = [FOLDER_META[k]["label"] for k in FOLDER_META if
                  (RESULTS_DIR / k / "detection_results.csv").exists()]
    pivot = df.pivot_table(index="exp_label", columns="strategy", values="map50")
    # sort rows by original experiment order
    order = [m["label"] for m in FOLDER_META.values()]
    pivot = pivot.reindex([l for l in order if l in pivot.index])
    pivot = pivot[STRATEGY_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="mAP@50")

    ax.set_xticks(range(len(STRATEGY_ORDER)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER], fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([l.replace("\n", " ") for l in pivot.index], fontsize=10)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if v < 0.10 else "white")

    ax.set_title("Faster R-CNN: mAP@50 Heatmap (Strategy × Condition)",
                 fontweight="bold")
    plt.tight_layout()
    out = OUT / "frcnn_03_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 4: Training time ──────────────────────────────────────────────────────

def plot_training_time(df):
    fig, ax = plt.subplots(figsize=(13, 5))

    all_labels = df["exp_label"].unique()
    # sort by original order
    order_map = {m["label"]: m["sort"] + (0 if m["group"] == "main" else 10)
                 for m in FOLDER_META.values()}
    all_labels = sorted(all_labels, key=lambda l: order_map.get(l, 99))

    x = np.arange(len(all_labels))
    n, w = len(STRATEGY_ORDER), 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    for i, strat in enumerate(STRATEGY_ORDER):
        heights = []
        for lbl in all_labels:
            row = df[(df["exp_label"] == lbl) & (df["strategy"] == strat)]
            hours = float(row["training_time_sec"].values[0]) / 3600 if not row.empty else 0.0
            heights.append(hours)
        ax.bar(x + offsets[i], heights, w,
               label=STRATEGY_LABELS[strat],
               color=STRATEGY_COLORS[strat], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("\n", " ") for l in all_labels], fontsize=9)
    ax.set_ylabel("Training time (hours)")
    ax.set_title("Faster R-CNN: Training Time per Strategy", fontweight="bold")
    ax.legend(fontsize=9)
    _axis_style(ax)

    plt.tight_layout()
    out = OUT / "frcnn_04_training_time.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading Faster R-CNN results...")
    df = load_all()
    if df.empty:
        print("No data found.")
    else:
        print(f"  Rows: {len(df)}\n")
        plot_map50_overview(df)
        plot_cl_gain(df)
        plot_heatmap(df)
        plot_training_time(df)
        print(f"\nAll plots saved to: {OUT}")
