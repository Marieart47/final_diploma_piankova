"""Визуализация результатов экспериментов по классификации (CIFAR-10 и STL-10)."""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

MODEL_COLORS = {
    "resnet18": "#C62828",
    "swin_t":   "#1565C0",
}
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "swin_t":   "Swin-T",
}
MODEL_ORDER = ["resnet18", "swin_t"]

_MODEL_SUFFIXES = [
    ("_resnet18", "resnet18"),
    ("_swin_t",   "swin_t"),
]

plt.rcParams.update({
    "figure.dpi":    150,
    "font.size":     11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def infer_model(name: str) -> str:
    for suffix, model in _MODEL_SUFFIXES:
        if name.endswith(suffix):
            return model
    return "swin_t"


def strip_model(name: str) -> str:
    for suffix, _ in _MODEL_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def parse_folder(name: str) -> dict | None:
    base = strip_model(name)
    m = re.match(
        r"(?P<dataset>cifar10|stl10)_(?P<etype>original|noisy|imbalance|artifacts)"
        r"(?:_noise(?P<noise>[\d.]+))?"
        r"(?:_im(?P<imb>[\d.]+))?"
        r"(?:_(?P<art_type>blur|noise)_art(?P<art_level>[\d.]+))?$",
        base,
    )
    if not m:
        return None
    return {
        "folder":    name,
        "dataset":   m.group("dataset"),
        "exp_type":  m.group("etype"),
        "noise_p":   float(m.group("noise"))     if m.group("noise")     else None,
        "imb_p":     float(m.group("imb"))       if m.group("imb")       else None,
        "art_type":  m.group("art_type"),
        "art_level": float(m.group("art_level")) if m.group("art_level") else None,
        "model":     infer_model(name),
    }


def load_results(results_dir: Path):
    test_recs, train_recs = [], []
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        meta = parse_folder(folder.name)
        if not meta:
            continue
        for csv_name, lst in [("test_results.csv", test_recs),
                               ("classification_results.csv", train_recs)]:
            path = folder / csv_name
            if path.exists():
                df = pd.read_csv(path)
                for k, v in meta.items():
                    df[k] = v
                lst.append(df)

    test_df  = pd.concat(test_recs,  ignore_index=True) if test_recs  else pd.DataFrame()
    train_df = pd.concat(train_recs, ignore_index=True) if train_recs else pd.DataFrame()
    return test_df, train_df


def make_exp_label(row) -> str:
    if row["exp_type"] == "original":  return "Original"
    if row["exp_type"] == "noisy":     return f"Noisy\n(p={row['noise_p']:.1f})"
    if row["exp_type"] == "imbalance": return f"Imbalance\n(f={row['imb_p']:.1f})"
    if row["exp_type"] == "artifacts": return f"Art-{row['art_type']}\n(a={row['art_level']:.1f})"
    return row["exp_type"]


def exp_sort_key(label: str) -> tuple:
    prefixes = {"original": 0, "noisy": 1, "imbalance": 2, "art-blur": 3, "art-noise": 4}
    low = label.lower().replace("\n", " ")
    for k, v in prefixes.items():
        if low.startswith(k):
            return (v, label)
    return (9, label)


def _axis_style(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ylim(ax, series, pad_lo=5, pad_hi=4):
    vals = series.dropna()
    if not vals.empty:
        ax.set_ylim(max(0, vals.min() - pad_lo), min(100, vals.max() + pad_hi))


# ── Plot 1: Accuracy overview — x=experiment, bars=strategy, panels=model ──────

def plot_accuracy_overview(test_df: pd.DataFrame, dataset: str):
    df = test_df[test_df["dataset"] == dataset].copy()
    if df.empty:
        return
    df["exp_label"] = df.apply(make_exp_label, axis=1)
    exp_labels = sorted(df["exp_label"].unique(), key=exp_sort_key)
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    fig, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 6), squeeze=False)
    for ax, model in zip(axes[0], models):
        mdf = df[df["model"] == model]
        x = np.arange(len(exp_labels))
        n, width = len(STRATEGY_ORDER), 0.18
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

        for i, strat in enumerate(STRATEGY_ORDER):
            heights = []
            for exp in exp_labels:
                row = mdf[(mdf["strategy"] == strat) & (mdf["exp_label"] == exp)]
                heights.append(row["test_accuracy"].values[0] * 100 if not row.empty else 0)
            bars = ax.bar(x + offsets[i], heights, width,
                          label=STRATEGY_LABELS[strat],
                          color=STRATEGY_COLORS[strat], alpha=0.85, edgecolor="white")
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                            f"{h:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_title(MODEL_LABELS[model], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, fontsize=9)
        ax.set_ylabel("Test Accuracy (%)")
        _ylim(ax, mdf["test_accuracy"] * 100)
        ax.legend(loc="lower right", framealpha=0.9)
        _axis_style(ax)

    fig.suptitle(f"{dataset.upper()} — Test Accuracy by Experiment and CL Strategy",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"{dataset}_01_accuracy_overview.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 2: Model comparison — x=experiment, bars=model, panels=strategy ───────

def plot_model_comparison(test_df: pd.DataFrame, dataset: str):
    df = test_df[test_df["dataset"] == dataset].copy()
    if df.empty:
        return
    df["exp_label"] = df.apply(make_exp_label, axis=1)
    exp_labels = sorted(df["exp_label"].unique(), key=exp_sort_key)
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    fig, axes = plt.subplots(1, len(STRATEGY_ORDER),
                             figsize=(5.5 * len(STRATEGY_ORDER), 5), squeeze=False)
    for ax, strat in zip(axes[0], STRATEGY_ORDER):
        sdf = df[df["strategy"] == strat]
        x = np.arange(len(exp_labels))
        n, width = len(models), 0.22
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

        for i, model in enumerate(models):
            heights = []
            for exp in exp_labels:
                row = sdf[(sdf["model"] == model) & (sdf["exp_label"] == exp)]
                heights.append(row["test_accuracy"].values[0] * 100 if not row.empty else 0)
            bars = ax.bar(x + offsets[i], heights, width,
                          label=MODEL_LABELS[model],
                          color=MODEL_COLORS[model], alpha=0.85, edgecolor="white")
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                            f"{h:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_title(STRATEGY_LABELS[strat], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, fontsize=8.5)
        ax.set_ylabel("Test Accuracy (%)")
        _ylim(ax, sdf["test_accuracy"] * 100)
        ax.legend(fontsize=8.5)
        _axis_style(ax)

    fig.suptitle(f"{dataset.upper()} — Model Comparison per CL Strategy",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"{dataset}_02_model_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 3: Heatmap — strategy × experiment, panels=model ──────────────────────

def plot_heatmap(test_df: pd.DataFrame, dataset: str):
    df = test_df[test_df["dataset"] == dataset].copy()
    if df.empty:
        return
    df["exp_label"] = df.apply(make_exp_label, axis=1)
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    fig, axes = plt.subplots(1, len(models),
                             figsize=(6.5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        mdf = df[df["model"] == model]
        pivot = mdf.pivot_table(index="strategy", columns="exp_label",
                                values="test_accuracy", aggfunc="mean")
        pivot = pivot.reindex([s for s in STRATEGY_ORDER if s in pivot.index])
        cols = sorted(pivot.columns, key=exp_sort_key)
        pivot = pivot[cols]

        vals = pivot.values * 100
        vmin = np.nanmin(vals) - 3
        vmax = np.nanmax(vals) + 1
        im = ax.imshow(vals, aspect="auto", cmap="YlGn", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([STRATEGY_LABELS[s] for s in pivot.index], fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    text_color = "white" if v > (np.nanmax(pivot.values) * 0.95) else "black"
                    ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center",
                            fontsize=8.5, color=text_color)

        fig.colorbar(im, ax=ax, label="Test Accuracy (%)", shrink=0.85)

    fig.suptitle(f"{dataset.upper()} — Accuracy Heatmap (Strategy × Experiment)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"{dataset}_03_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 4: Curriculum gain over baseline ──────────────────────────────────────

def plot_curriculum_gain(test_df: pd.DataFrame, dataset: str):
    df = test_df[test_df["dataset"] == dataset].copy()
    if df.empty:
        return
    df["exp_label"] = df.apply(make_exp_label, axis=1)
    exp_labels = sorted(df["exp_label"].unique(), key=exp_sort_key)
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    cl_strats = [s for s in STRATEGY_ORDER if s != "baseline"]

    fig, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 5),
                             squeeze=False, sharey=True)
    for ax, model in zip(axes[0], models):
        mdf = df[df["model"] == model]
        x = np.arange(len(exp_labels))
        n, width = len(cl_strats), 0.22
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

        for i, strat in enumerate(cl_strats):
            deltas = []
            for exp in exp_labels:
                base_row = mdf[(mdf["strategy"] == "baseline") & (mdf["exp_label"] == exp)]
                cl_row   = mdf[(mdf["strategy"] == strat)      & (mdf["exp_label"] == exp)]
                if not base_row.empty and not cl_row.empty:
                    delta = (cl_row["test_accuracy"].values[0] -
                             base_row["test_accuracy"].values[0]) * 100
                else:
                    delta = 0
                deltas.append(delta)
            bars = ax.bar(x + offsets[i], deltas, width,
                          label=STRATEGY_LABELS[strat],
                          color=STRATEGY_COLORS[strat], alpha=0.85, edgecolor="white")
            for bar, d in zip(bars, deltas):
                if abs(d) >= 0.3:
                    va = "bottom" if d >= 0 else "top"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            d + (0.15 if d >= 0 else -0.15),
                            f"{d:+.1f}", ha="center", va=va, fontsize=7)

        ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, fontsize=9)
        ax.set_ylabel("Accuracy Δ vs Baseline (%)")
        ax.legend(fontsize=8.5)
        _axis_style(ax)

    fig.suptitle(f"{dataset.upper()} — Curriculum Learning Gain over Baseline",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"{dataset}_04_curriculum_gain.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 5: Training curves — panels=experiment, lines=strategy ────────────────

def plot_training_curves(train_df: pd.DataFrame, dataset: str):
    df = train_df[train_df["dataset"] == dataset].copy()
    if df.empty:
        return
    df["exp_label"] = df.apply(make_exp_label, axis=1)
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    for model in models:
        mdf = df[df["model"] == model]
        exp_labels = sorted(mdf["exp_label"].unique(), key=exp_sort_key)
        n = len(exp_labels)
        if n == 0:
            continue
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)

        for idx, exp in enumerate(exp_labels):
            ax = axes[idx // cols][idx % cols]
            edf = mdf[mdf["exp_label"] == exp]
            for strat in STRATEGY_ORDER:
                sdf = edf[edf["strategy"] == strat].sort_values("epoch")
                if sdf.empty:
                    continue
                ax.plot(sdf["epoch"], sdf["val_acc"] * 100,
                        color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat],
                        linewidth=2, alpha=0.9)
            ax.set_title(exp.replace("\n", " "), fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Val Accuracy (%)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle(f"{dataset.upper()} — Training Curves ({MODEL_LABELS[model]})",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = PLOTS_DIR / f"{dataset}_05_training_curves_{model}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# ── Plot 6: Difficulty scaling — accuracy vs severity, per model ───────────────

def plot_difficulty_scaling(test_df: pd.DataFrame, dataset: str):
    """Line chart: how accuracy changes with increasing difficulty level."""
    df = test_df[test_df["dataset"] == dataset].copy()
    if df.empty:
        return
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    cases = [
        ("noisy",     "noise_p",  "Noise level (p)"),
        ("imbalance", "imb_p",    "Imbalance factor"),
    ]

    n_cases = len(cases)
    fig, axes = plt.subplots(len(models), n_cases,
                             figsize=(6 * n_cases, 4.5 * len(models)), squeeze=False)

    for mi, model in enumerate(models):
        mdf = df[df["model"] == model]
        for ci, (etype, col, xlabel) in enumerate(cases):
            ax = axes[mi][ci]
            edf = mdf[mdf["exp_type"] == etype]
            if edf.empty:
                ax.set_visible(False)
                continue
            levels = sorted(edf[col].dropna().unique())
            for strat in STRATEGY_ORDER:
                sdf = edf[edf["strategy"] == strat]
                ys = []
                for lvl in levels:
                    row = sdf[sdf[col] == lvl]
                    ys.append(row["test_accuracy"].values[0] * 100 if not row.empty else np.nan)
                ax.plot(levels, ys, "o-",
                        color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat],
                        linewidth=2, markersize=7)
            ax.set_title(f"{MODEL_LABELS[model]} — {etype.capitalize()}", fontsize=11, fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Test Accuracy (%)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(f"{dataset.upper()} — Accuracy vs Difficulty Level",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"{dataset}_06_difficulty_scaling.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading results from: {RESULTS_DIR}")
    test_df, train_df = load_results(RESULTS_DIR)
    print(f"  Test rows: {len(test_df)},  Train rows: {len(train_df)}\n")

    for dataset in ["cifar10", "stl10"]:
        ds_test  = test_df[test_df["dataset"]  == dataset] if not test_df.empty  else pd.DataFrame()
        ds_train = train_df[train_df["dataset"] == dataset] if not train_df.empty else pd.DataFrame()
        if ds_test.empty:
            print(f"  No data for {dataset}, skipping.\n")
            continue

        models_found = ds_test["model"].unique().tolist()
        exps_found   = ds_test["exp_type"].unique().tolist()
        print(f"── {dataset.upper()} ({'·'.join(models_found)}) ─────────────────────")
        print(f"   Experiments: {', '.join(sorted(exps_found))}")

        plot_accuracy_overview(test_df,  dataset)
        plot_model_comparison(test_df,   dataset)
        plot_heatmap(test_df,            dataset)
        plot_curriculum_gain(test_df,    dataset)
        plot_training_curves(train_df,   dataset)
        plot_difficulty_scaling(test_df, dataset)
        print()

    print(f"All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
