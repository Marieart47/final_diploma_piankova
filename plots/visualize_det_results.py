"""Visualization of object detection experiment results (YOLOv8 / VOC)."""

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

# ── Data loading ───────────────────────────────────────────────────────────────

FOLDER_META = {
    "voc_original_yolov8n":             {"label": "Original",          "type": "main",      "sort": 0},
    "voc_noisy_noise0.3_yolov8n":       {"label": "Noisy\n(p=0.3)",    "type": "main",      "sort": 1},
    "voc_noisy_noise0.6_yolov8n":       {"label": "Noisy\n(p=0.6)",    "type": "main",      "sort": 2},
    "voc_imbalance_im0.3_yolov8n":      {"label": "Imbalance\n(f=0.3)","type": "main",      "sort": 3},
    "voc_imbalance_im0.6_yolov8n":      {"label": "Imbalance\n(f=0.6)","type": "main",      "sort": 4},
    "voc_artifacts_blur_art0.4_yolov8n": {"label": "Blur (α=0.4)",     "type": "artifacts", "sort": 0},
    "voc_artifacts_noise_art0.4_yolov8n":{"label": "Noise (α=0.4)",    "type": "artifacts", "sort": 1},
}

def load_all():
    rows = []
    for folder, meta in FOLDER_META.items():
        csv = RESULTS_DIR / folder / "detection_results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df["exp_label"] = meta["label"]
        df["exp_type"]  = meta["type"]
        df["sort_key"]  = meta["sort"]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _axis_style(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Plot 1: mAP@50 overview (no artifacts) ────────────────────────────────────

def plot_map50_overview(df):
    d = df[df["exp_type"] == "main"].copy()
    exp_labels = d.drop_duplicates("exp_label").sort_values("sort_key")["exp_label"].tolist()

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(exp_labels)), 6))
    x = np.arange(len(exp_labels))
    n, width = len(STRATEGY_ORDER), 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, strat in enumerate(STRATEGY_ORDER):
        heights = []
        for exp in exp_labels:
            row = d[(d["strategy"] == strat) & (d["exp_label"] == exp)]
            heights.append(float(row["map50"].values[0]) if not row.empty else 0.0)
        bars = ax.bar(x + offsets[i], heights, width,
                      label=STRATEGY_LABELS[strat],
                      color=STRATEGY_COLORS[strat], alpha=0.85, edgecolor="white")
        for bar, h in zip(bars, heights):
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=9)
    ax.set_ylabel("mAP@50")
    ax.set_ylim(0, d["map50"].max() * 1.15 + 0.01)
    ax.legend(loc="upper right", framealpha=0.9)
    _axis_style(ax)
    fig.suptitle(
        "Pascal VOC — YOLOv8n: mAP@50 by Degradation and CL Strategy\n"
        "(blur/noise artifacts shown in a separate plot)",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT / "det_01_map50_overview.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 2: CL gain over baseline ─────────────────────────────────────────────

def plot_cl_gain(df):
    d = df[df["exp_type"] == "main"].copy()
    exp_labels = d.drop_duplicates("exp_label").sort_values("sort_key")["exp_label"].tolist()
    cl_strats = [s for s in STRATEGY_ORDER if s != "baseline"]

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(exp_labels)), 5))
    ax.axhspan(0, 0.15, color="#E8F5E9", alpha=0.6, zorder=0)
    ax.axhspan(-0.17, 0, color="#FFEBEE", alpha=0.6, zorder=0)

    x = np.arange(len(exp_labels))
    n, width = len(cl_strats), 0.22
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, strat in enumerate(cl_strats):
        deltas = []
        for exp in exp_labels:
            base = d[(d["strategy"] == "baseline") & (d["exp_label"] == exp)]
            cl   = d[(d["strategy"] == strat)      & (d["exp_label"] == exp)]
            if not base.empty and not cl.empty:
                deltas.append(float(cl["map50"].values[0]) - float(base["map50"].values[0]))
            else:
                deltas.append(0.0)
        bars = ax.bar(x + offsets[i], deltas, width,
                      label=STRATEGY_LABELS[strat],
                      color=STRATEGY_COLORS[strat], alpha=0.85, edgecolor="white")
        for bar, dv in zip(bars, deltas):
            if abs(dv) >= 0.005:
                va = "bottom" if dv >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        dv + (0.003 if dv >= 0 else -0.003),
                        f"{dv:+.3f}", ha="center", va=va, fontsize=7)

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=9)
    ax.set_ylabel("ΔmAP@50 vs Baseline")
    ax.legend(fontsize=8.5)
    _axis_style(ax)
    fig.suptitle("Pascal VOC — YOLOv8n: CL Gain over Baseline (ΔmAP@50)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT / "det_02_cl_gain.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 3: Artifacts collapse ─────────────────────────────────────────────────

def plot_artifacts_collapse(df):
    d = df[df["exp_type"] == "artifacts"].copy()
    art_labels = d.drop_duplicates("exp_label").sort_values("sort_key")["exp_label"].tolist()
    art_titles = {
        "Blur (α=0.4)":  "Gaussian Blur (α=0.4)",
        "Noise (α=0.4)": "Pixel Noise (α=0.4)",
    }

    fig, axes = plt.subplots(1, len(art_labels), figsize=(7 * len(art_labels), 5))
    if len(art_labels) == 1:
        axes = [axes]

    for ax, exp in zip(axes, art_labels):
        ed = d[d["exp_label"] == exp]
        heights, strat_names = [], []
        for strat in STRATEGY_ORDER:
            row = ed[ed["strategy"] == strat]
            if not row.empty:
                heights.append(max(float(row["map50"].values[0]), 1e-7))
                strat_names.append(strat)

        bars = ax.bar(range(len(strat_names)), heights,
                      color=[STRATEGY_COLORS[s] for s in strat_names],
                      alpha=0.85, edgecolor="white")
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.3,
                    f"{h:.2e}", ha="center", va="bottom", fontsize=8, rotation=90)

        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1e-1)
        ax.set_xticks(range(len(strat_names)))
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in strat_names], fontsize=9)
        ax.set_ylabel("mAP@50 (log)")
        ax.set_title(art_titles.get(exp, exp), fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.text(0.97, 0.97,
                "Complete detector collapse\n(mAP ≈ 0 for all strategies)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="#C62828", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C62828", lw=1.5))

    fig.suptitle("VOC YOLOv8n: Detector Collapse under Visual Artifacts",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT / "det_03_artifacts_collapse.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 4: Heatmap ────────────────────────────────────────────────────────────

def plot_heatmap(df):
    d = df[df["exp_type"] == "main"].copy()
    exp_labels = d.drop_duplicates("exp_label").sort_values("sort_key")["exp_label"].tolist()

    pivot = pd.DataFrame(
        index=[STRATEGY_LABELS[s] for s in STRATEGY_ORDER],
        columns=exp_labels,
        dtype=float,
    )
    for strat in STRATEGY_ORDER:
        for exp in exp_labels:
            row = d[(d["strategy"] == strat) & (d["exp_label"] == exp)]
            if not row.empty:
                pivot.loc[STRATEGY_LABELS[strat], exp] = float(row["map50"].values[0])

    vals = pivot.values.astype(float)
    col_best = np.nanargmax(vals, axis=0)

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(exp_labels)), 3.5))
    vmin = np.nanmin(vals) - 0.02
    vmax = np.nanmax(vals) + 0.01
    im = ax.imshow(vals, aspect="auto", cmap="YlGn", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(exp_labels)))
    ax.set_xticklabels(exp_labels, rotation=0, ha="center", fontsize=9)
    ax.set_yticks(range(len(STRATEGY_ORDER)))
    ax.set_yticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER], fontsize=9)

    for i in range(len(STRATEGY_ORDER)):
        for j in range(len(exp_labels)):
            v = vals[i, j]
            if not np.isnan(v):
                is_best = (col_best[j] == i)
                text_color = "white" if v > np.nanmax(vals) * 0.93 else "black"
                fw = "bold" if is_best else "normal"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8.5, color=text_color, fontweight=fw)

    fig.colorbar(im, ax=ax, label="mAP@50", shrink=0.85)
    fig.suptitle("Pascal VOC — mAP@50 Heatmap (bold = best per condition)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT / "det_04_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 5: mAP@50 vs mAP@50-95 ──────────────────────────────────────────────

def plot_map50_vs_map9595(df):
    d = df[df["exp_type"] == "main"].copy()
    exp_labels = d.drop_duplicates("exp_label").sort_values("sort_key")["exp_label"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    metrics = [
        ("map50",    "mAP@50",    "mAP@50 — object classification"),
        ("map50_95", "mAP@50-95", "mAP@50-95 — localization precision"),
    ]

    x = np.arange(len(exp_labels))
    n, width = len(STRATEGY_ORDER), 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for i, strat in enumerate(STRATEGY_ORDER):
            heights = []
            for exp in exp_labels:
                row = d[(d["strategy"] == strat) & (d["exp_label"] == exp)]
                heights.append(float(row[metric].values[0]) if not row.empty else 0.0)
            ax.bar(x + offsets[i], heights, width,
                   label=STRATEGY_LABELS[strat],
                   color=STRATEGY_COLORS[strat], alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, d[metric].max() * 1.15 + 0.01)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5)
        _axis_style(ax)

    fig.suptitle("VOC YOLOv8n: mAP@50 vs mAP@50-95",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT / "det_05_map50_vs_map9595.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading detection results...")
    df = load_all()
    if df.empty:
        print("No data found.")
    else:
        print(f"  Rows: {len(df)}\n")
        plot_map50_overview(df)
        plot_cl_gain(df)
        plot_artifacts_collapse(df)
        plot_heatmap(df)
        plot_map50_vs_map9595(df)
        print(f"\nAll plots saved to: {OUT}")
