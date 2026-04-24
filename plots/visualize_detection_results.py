"""Визуализация результатов экспериментов по детекции объектов (YOLOv8 / VOC)."""

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR   = Path(__file__).parent / "output"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_COLORS = {
    "baseline":  "#2196F3",
    "stagewise": "#4CAF50",
    "static":    "#FF9800",
    "online":    "#E91E63",
}
STRATEGY_ORDER  = ["baseline", "stagewise", "static", "online"]
STRATEGY_LABELS = {
    "baseline":  "Baseline",
    "stagewise": "Stagewise CL",
    "static":    "Static CL",
    "online":    "Online CL",
}

plt.rcParams.update({
    "figure.dpi":     150,
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


# ── Folder parsing ──────────────────────────────────────────────────────────────

def parse_folder(name: str, suffix: str = "") -> dict | None:
    pattern = (
        r"voc_(?P<etype>original|noisy|imbalance|artifacts)"
        r"(?:_noise(?P<noise>[\d.]+))?"
        r"(?:_im(?P<imb>[\d.]+))?"
        r"(?:_(?P<art_type>blur|noise|low_res|mixed)_art(?P<art_level>[\d.]+))?"
        r"_(?P<model>yolov8[nsm])"
        + (f"_{re.escape(suffix)}" if suffix else "") + r"$"
    )
    m = re.match(pattern, name)
    if not m:
        return None
    return {
        "folder":    name,
        "exp_type":  m.group("etype"),
        "noise_p":   float(m.group("noise"))     if m.group("noise")     else None,
        "imb_p":     float(m.group("imb"))       if m.group("imb")       else None,
        "art_type":  m.group("art_type"),
        "art_level": float(m.group("art_level")) if m.group("art_level") else None,
        "model":     m.group("model"),
    }


def load_results(results_dir: Path, suffix: str = "") -> pd.DataFrame:
    rows = []
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        meta = parse_folder(folder.name, suffix)
        if not meta:
            continue
        csv = folder / "detection_results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        for k, v in meta.items():
            df[k] = v
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def make_exp_label(row) -> str:
    if row["exp_type"] == "original":  return "Original"
    if row["exp_type"] == "noisy":     return f"Noisy\n(p={row['noise_p']:.1f})"
    if row["exp_type"] == "imbalance": return f"Imbalance\n(f={row['imb_p']:.1f})"
    if row["exp_type"] == "artifacts": return f"Art-{row['art_type']}\n(a={row['art_level']:.1f})"
    return row["exp_type"]


def exp_sort_key(label: str) -> tuple:
    prefixes = {"original": 0, "noisy": 1, "imbalance": 2, "art": 3}
    low = label.lower().replace("\n", " ")
    for k, v in prefixes.items():
        if low.startswith(k):
            return (v, label)
    return (9, label)


def _axis_style(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Plot 1: mAP50 overview ──────────────────────────────────────────────────────

def plot_map50_overview(df: pd.DataFrame, tag: str):
    df = df.copy()
    df["exp_label"] = df.apply(make_exp_label, axis=1)
    exp_labels = sorted(df["exp_label"].unique(), key=exp_sort_key)

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(exp_labels)), 6))
    x = np.arange(len(exp_labels))
    n, width = len(STRATEGY_ORDER), 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, strat in enumerate(STRATEGY_ORDER):
        heights = []
        for exp in exp_labels:
            row = df[(df["strategy"] == strat) & (df["exp_label"] == exp)]
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
    ax.set_ylabel("mAP50")
    ax.set_ylim(0, df["map50"].max() * 1.15 + 0.01)
    ax.legend(loc="upper right", framealpha=0.9)
    _axis_style(ax)
    fig.suptitle(f"YOLOv8 / VOC — mAP50 by Experiment and CL Strategy{tag}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"detection{tag}_01_map50_overview.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 2: Heatmap ─────────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, tag: str):
    df = df.copy()
    df["exp_label"] = df.apply(make_exp_label, axis=1)

    pivot = df.pivot_table(index="strategy", columns="exp_label",
                           values="map50", aggfunc="mean")
    pivot = pivot.reindex([s for s in STRATEGY_ORDER if s in pivot.index])
    cols = sorted(pivot.columns, key=exp_sort_key)
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(cols)), 3.5))
    vals = pivot.values
    vmin = np.nanmin(vals) - 0.02
    vmax = np.nanmax(vals) + 0.01
    im = ax.imshow(vals, aspect="auto", cmap="YlGn", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([STRATEGY_LABELS[s] for s in pivot.index], fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                text_color = "white" if v > np.nanmax(vals) * 0.93 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8.5, color=text_color)

    fig.colorbar(im, ax=ax, label="mAP50", shrink=0.85)
    fig.suptitle(f"YOLOv8 / VOC — mAP50 Heatmap (Strategy × Experiment){tag}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"detection{tag}_02_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 3: CL gain over baseline ───────────────────────────────────────────────

def plot_curriculum_gain(df: pd.DataFrame, tag: str):
    df = df.copy()
    df["exp_label"] = df.apply(make_exp_label, axis=1)
    exp_labels = sorted(df["exp_label"].unique(), key=exp_sort_key)
    cl_strats = [s for s in STRATEGY_ORDER if s != "baseline"]

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(exp_labels)), 5))
    x = np.arange(len(exp_labels))
    n, width = len(cl_strats), 0.22
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, strat in enumerate(cl_strats):
        deltas = []
        for exp in exp_labels:
            base = df[(df["strategy"] == "baseline") & (df["exp_label"] == exp)]
            cl   = df[(df["strategy"] == strat)      & (df["exp_label"] == exp)]
            if not base.empty and not cl.empty:
                deltas.append(float(cl["map50"].values[0]) - float(base["map50"].values[0]))
            else:
                deltas.append(0.0)
        bars = ax.bar(x + offsets[i], deltas, width,
                      label=STRATEGY_LABELS[strat],
                      color=STRATEGY_COLORS[strat], alpha=0.85, edgecolor="white")
        for bar, d in zip(bars, deltas):
            if abs(d) >= 0.005:
                va = "bottom" if d >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        d + (0.003 if d >= 0 else -0.003),
                        f"{d:+.3f}", ha="center", va=va, fontsize=7)

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=9)
    ax.set_ylabel("Δ mAP50 vs Baseline")
    ax.legend(fontsize=8.5)
    _axis_style(ax)
    fig.suptitle(f"YOLOv8 / VOC — CL Gain over Baseline{tag}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"detection{tag}_03_curriculum_gain.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 4: mAP50 vs difficulty level ──────────────────────────────────────────

def plot_difficulty_scaling(df: pd.DataFrame, tag: str):
    cases = [
        ("noisy",     "noise_p",  "Noise level (p)"),
        ("imbalance", "imb_p",    "Imbalance factor"),
    ]
    fig, axes = plt.subplots(1, len(cases), figsize=(7 * len(cases), 5), squeeze=False)

    for ci, (etype, col, xlabel) in enumerate(cases):
        ax = axes[0][ci]
        edf = df[df["exp_type"] == etype]
        if edf.empty:
            ax.set_visible(False)
            continue
        levels = sorted(edf[col].dropna().unique())
        for strat in STRATEGY_ORDER:
            sdf = edf[edf["strategy"] == strat]
            ys = []
            for lvl in levels:
                row = sdf[sdf[col] == lvl]
                ys.append(float(row["map50"].values[0]) if not row.empty else np.nan)
            ax.plot(levels, ys, "o-",
                    color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat],
                    linewidth=2, markersize=7)
        ax.set_title(etype.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mAP50")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"YOLOv8 / VOC — mAP50 vs Difficulty Level{tag}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"detection{tag}_04_difficulty_scaling.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="",
                        help="Folder suffix to filter (e.g. 'v2' for voc_*_yolov8n_v2)")
    args = parser.parse_args()

    suffix = args.suffix
    tag = f"_{suffix}" if suffix else ""

    print(f"Loading detection results from: {RESULTS_DIR}")
    df = load_results(RESULTS_DIR, suffix)
    if df.empty:
        print("No detection_results.csv files found. Nothing to plot.")
        return

    print(f"  Rows loaded: {len(df)}")
    print(f"  Experiments: {sorted(df['exp_type'].unique())}")
    print(f"  Strategies:  {sorted(df['strategy'].unique())}\n")

    plot_map50_overview(df, tag)
    plot_heatmap(df, tag)
    plot_curriculum_gain(df, tag)
    plot_difficulty_scaling(df, tag)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
