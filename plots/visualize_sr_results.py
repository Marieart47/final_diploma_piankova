"""Визуализация результатов экспериментов по Super Resolution."""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR   = Path(__file__).parent / "output"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_COLORS = {
    "baseline":       "#2196F3",
    "stagewise":      "#4CAF50",
    "static":         "#FF9800",
    "online":         "#E91E63",
    "anticurriculum": "#9C27B0",
}
STRATEGY_ORDER  = ["baseline", "stagewise", "static", "online", "anticurriculum"]
STRATEGY_LABELS = {
    "baseline":       "Baseline",
    "stagewise":      "Stagewise CL",
    "static":         "Static CL",
    "online":         "Online CL",
    "anticurriculum": "Anti-Curriculum",
}

ARCH_COLORS = {
    "srcnn":  "#C62828",
    "espcn":  "#1565C0",
    "swinir": "#2E7D32",
}
ARCH_LABELS = {
    "srcnn":  "SRCNN",
    "espcn":  "ESPCN",
    "swinir": "SwinIR",
}
ARCH_ORDER = ["srcnn", "espcn", "swinir"]

_ARCH_SUFFIXES = [("_swinir", "swinir"), ("_espcn", "espcn"), ("_srcnn", "srcnn")]

plt.rcParams.update({
    "figure.dpi":     150,
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def _infer_arch(name: str) -> str:
    for suffix, arch in _ARCH_SUFFIXES:
        if name.endswith(suffix):
            return arch
    return "srcnn"


def _strip_arch(name: str) -> str:
    for suffix, _ in _ARCH_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def parse_folder(name: str) -> dict | None:
    base = _strip_arch(name)
    m = re.match(
        r"(?P<dataset>[^_]+)_x(?P<scale>\d+)_"
        r"(?P<deg_type>original|noise|blur|jpeg|mixed)"
        r"(?:_deg(?P<deg_level>[\d.]+))?$",
        base,
    )
    if not m:
        return None
    return {
        "folder":    name,
        "dataset":   m.group("dataset"),
        "scale":     int(m.group("scale")),
        "deg_type":  m.group("deg_type"),
        "deg_level": float(m.group("deg_level")) if m.group("deg_level") else 0.0,
        "arch":      _infer_arch(name),
    }


def load_results(results_dir: Path):
    test_recs, train_recs = [], []
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        meta = parse_folder(folder.name)
        if not meta:
            continue
        for fname, lst in [("sr_test_results.csv", test_recs),
                           ("sr_results.csv",      train_recs)]:
            path = folder / fname
            if path.exists():
                df = pd.read_csv(path)
                for k, v in meta.items():
                    df[k] = v
                lst.append(df)

    test_df  = pd.concat(test_recs,  ignore_index=True) if test_recs  else pd.DataFrame()
    train_df = pd.concat(train_recs, ignore_index=True) if train_recs else pd.DataFrame()
    return test_df, train_df


def make_deg_label(row) -> str:
    t = row["deg_type"]
    if t == "original":
        return "Original"
    lvl = row.get("deg_level", 0.0)
    labels = {"noise": "Noise", "blur": "Blur", "jpeg": "JPEG", "mixed": "Mixed"}
    return f"{labels.get(t, t)}\n(l={lvl:.1f})"


def deg_sort_key(label: str) -> tuple:
    order = {"original": 0, "noise": 1, "blur": 2, "jpeg": 3, "mixed": 4}
    low = label.lower().replace("\n", " ").split("(")[0].strip()
    return (order.get(low, 9), label)


def _axis_style(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Plot 1: PSNR overview — x=degradation, bars=strategy, panels=arch ──────────

def plot_psnr_overview(test_df: pd.DataFrame, dataset: str, scale: int):
    df = test_df[(test_df["dataset"] == dataset) & (test_df["scale"] == scale)].copy()
    if df.empty:
        return
    df["deg_label"] = df.apply(make_deg_label, axis=1)
    deg_labels = sorted(df["deg_label"].unique(), key=deg_sort_key)
    archs = [a for a in ARCH_ORDER if a in df["arch"].unique()]
    strats = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]

    fig, axes = plt.subplots(1, len(archs), figsize=(7 * len(archs), 6), squeeze=False)
    for ax, arch in zip(axes[0], archs):
        adf = df[df["arch"] == arch]
        x = np.arange(len(deg_labels))
        n, width = len(strats), 0.18
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

        for i, strat in enumerate(strats):
            heights = []
            for lbl in deg_labels:
                row = adf[(adf["strategy"] == strat) & (adf["deg_label"] == lbl)]
                heights.append(row["test_psnr"].values[0] if not row.empty else 0)
            bars = ax.bar(x + offsets[i], heights, width,
                          label=STRATEGY_LABELS[strat],
                          color=STRATEGY_COLORS[strat], alpha=0.85, edgecolor="white")
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                            f"{h:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_title(ARCH_LABELS[arch], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(deg_labels, fontsize=9)
        ax.set_ylabel("Test PSNR (dB)")
        ax.legend(loc="lower right", framealpha=0.9)
        _axis_style(ax)

    tag = f"{dataset}_x{scale}"
    fig.suptitle(f"{tag.upper()} — Test PSNR by Degradation and CL Strategy",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"{tag}_01_psnr_overview.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 2: Architecture comparison — x=degradation, bars=arch, panels=strategy ─

def plot_arch_comparison(test_df: pd.DataFrame, dataset: str, scale: int):
    df = test_df[(test_df["dataset"] == dataset) & (test_df["scale"] == scale)].copy()
    if df.empty:
        return
    df["deg_label"] = df.apply(make_deg_label, axis=1)
    deg_labels = sorted(df["deg_label"].unique(), key=deg_sort_key)
    archs = [a for a in ARCH_ORDER if a in df["arch"].unique()]
    strats = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]

    fig, axes = plt.subplots(1, len(strats), figsize=(5.5 * len(strats), 5), squeeze=False)
    for ax, strat in zip(axes[0], strats):
        sdf = df[df["strategy"] == strat]
        x = np.arange(len(deg_labels))
        n, width = len(archs), 0.22
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

        for i, arch in enumerate(archs):
            heights = []
            for lbl in deg_labels:
                row = sdf[(sdf["arch"] == arch) & (sdf["deg_label"] == lbl)]
                heights.append(row["test_psnr"].values[0] if not row.empty else 0)
            bars = ax.bar(x + offsets[i], heights, width,
                          label=ARCH_LABELS[arch],
                          color=ARCH_COLORS[arch], alpha=0.85, edgecolor="white")
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                            f"{h:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)

        ax.set_title(STRATEGY_LABELS[strat], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(deg_labels, fontsize=8.5)
        ax.set_ylabel("Test PSNR (dB)")
        ax.legend(fontsize=8.5)
        _axis_style(ax)

    tag = f"{dataset}_x{scale}"
    fig.suptitle(f"{tag.upper()} — Architecture Comparison per CL Strategy",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"{tag}_02_arch_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 3: Heatmap — strategy × degradation, panels=arch ──────────────────────

def plot_heatmap(test_df: pd.DataFrame, dataset: str, scale: int):
    df = test_df[(test_df["dataset"] == dataset) & (test_df["scale"] == scale)].copy()
    if df.empty:
        return
    df["deg_label"] = df.apply(make_deg_label, axis=1)
    archs = [a for a in ARCH_ORDER if a in df["arch"].unique()]
    strats = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]

    fig, axes = plt.subplots(1, len(archs), figsize=(6.5 * len(archs), 4), squeeze=False)
    for ax, arch in zip(axes[0], archs):
        adf = df[df["arch"] == arch]
        pivot = adf.pivot_table(index="strategy", columns="deg_label",
                                values="test_psnr", aggfunc="mean")
        pivot = pivot.reindex([s for s in strats if s in pivot.index])
        cols  = sorted(pivot.columns, key=deg_sort_key)
        pivot = pivot[cols]

        vals = pivot.values
        vmin = np.nanmin(vals) - 1
        vmax = np.nanmax(vals) + 0.5
        im = ax.imshow(vals, aspect="auto", cmap="YlGn", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([STRATEGY_LABELS.get(s, s) for s in pivot.index], fontsize=9)
        ax.set_title(ARCH_LABELS[arch], fontsize=12, fontweight="bold")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    tc = "white" if v > (np.nanmax(vals) * 0.97) else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8.5, color=tc)

        fig.colorbar(im, ax=ax, label="Test PSNR (dB)", shrink=0.85)

    tag = f"{dataset}_x{scale}"
    fig.suptitle(f"{tag.upper()} — PSNR Heatmap (Strategy × Degradation)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"{tag}_03_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 4: Curriculum gain — ΔPSNR vs baseline ────────────────────────────────

def plot_curriculum_gain(test_df: pd.DataFrame, dataset: str, scale: int):
    df = test_df[(test_df["dataset"] == dataset) & (test_df["scale"] == scale)].copy()
    if df.empty:
        return
    df["deg_label"] = df.apply(make_deg_label, axis=1)
    deg_labels = sorted(df["deg_label"].unique(), key=deg_sort_key)
    archs      = [a for a in ARCH_ORDER if a in df["arch"].unique()]
    cl_strats  = [s for s in STRATEGY_ORDER if s != "baseline" and s in df["strategy"].unique()]

    fig, axes = plt.subplots(1, len(archs), figsize=(7 * len(archs), 5),
                             squeeze=False, sharey=True)
    for ax, arch in zip(axes[0], archs):
        adf = df[df["arch"] == arch]
        x = np.arange(len(deg_labels))
        n, width = len(cl_strats), 0.22
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

        for i, strat in enumerate(cl_strats):
            deltas = []
            for lbl in deg_labels:
                base_row = adf[(adf["strategy"] == "baseline") & (adf["deg_label"] == lbl)]
                cl_row   = adf[(adf["strategy"] == strat)      & (adf["deg_label"] == lbl)]
                if not base_row.empty and not cl_row.empty:
                    d = cl_row["test_psnr"].values[0] - base_row["test_psnr"].values[0]
                else:
                    d = 0.0
                deltas.append(d)
            bars = ax.bar(x + offsets[i], deltas, width,
                          label=STRATEGY_LABELS[strat],
                          color=STRATEGY_COLORS[strat], alpha=0.85, edgecolor="white")
            for bar, d in zip(bars, deltas):
                if abs(d) >= 0.05:
                    va = "bottom" if d >= 0 else "top"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            d + (0.02 if d >= 0 else -0.02),
                            f"{d:+.2f}", ha="center", va=va, fontsize=7)

        ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
        ax.set_title(ARCH_LABELS[arch], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(deg_labels, fontsize=9)
        ax.set_ylabel("PSNR Δ vs Baseline (dB)")
        ax.legend(fontsize=8.5)
        _axis_style(ax)

    tag = f"{dataset}_x{scale}"
    fig.suptitle(f"{tag.upper()} — Curriculum Learning Gain over Baseline",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"{tag}_04_curriculum_gain.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 5: Training curves — panels=degradation, lines=strategy ───────────────

def plot_training_curves(train_df: pd.DataFrame, dataset: str, scale: int):
    df = train_df[(train_df["dataset"] == dataset) & (train_df["scale"] == scale)].copy()
    if df.empty:
        return
    df["deg_label"] = df.apply(make_deg_label, axis=1)
    archs  = [a for a in ARCH_ORDER if a in df["arch"].unique()]
    strats = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]

    for arch in archs:
        adf = df[df["arch"] == arch]
        deg_labels = sorted(adf["deg_label"].unique(), key=deg_sort_key)
        n = len(deg_labels)
        if n == 0:
            continue
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)

        for idx, lbl in enumerate(deg_labels):
            ax  = axes[idx // cols][idx % cols]
            ldf = adf[adf["deg_label"] == lbl]
            for strat in strats:
                sdf = ldf[ldf["strategy"] == strat].sort_values("epoch")
                if sdf.empty:
                    continue
                ax.plot(sdf["epoch"], sdf["val_psnr"],
                        color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat],
                        linewidth=2, alpha=0.9)
            ax.set_title(lbl.replace("\n", " "), fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Val PSNR (dB)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        tag = f"{dataset}_x{scale}"
        fig.suptitle(f"{tag.upper()} — Training Curves ({ARCH_LABELS[arch]})",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = PLOTS_DIR / f"{tag}_05_training_curves_{arch}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# ── Plot 6: Difficulty scaling — PSNR vs deg_level ─────────────────────────────

def plot_difficulty_scaling(test_df: pd.DataFrame, dataset: str, scale: int):
    df = test_df[(test_df["dataset"] == dataset) & (test_df["scale"] == scale)].copy()
    if df.empty:
        return
    archs  = [a for a in ARCH_ORDER if a in df["arch"].unique()]
    strats = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]

    # Только типы с числовым уровнем (не original)
    deg_types = [t for t in ("noise", "blur", "jpeg", "mixed")
                 if t in df["deg_type"].unique()]
    if not deg_types:
        return

    fig, axes = plt.subplots(len(archs), len(deg_types),
                             figsize=(6 * len(deg_types), 4.5 * len(archs)),
                             squeeze=False)

    for mi, arch in enumerate(archs):
        adf = df[df["arch"] == arch]
        for ci, dt in enumerate(deg_types):
            ax  = axes[mi][ci]
            edf = adf[adf["deg_type"] == dt]
            if edf.empty:
                ax.set_visible(False)
                continue
            levels = sorted(edf["deg_level"].dropna().unique())
            for strat in strats:
                sdf = edf[edf["strategy"] == strat]
                ys  = []
                for lvl in levels:
                    row = sdf[sdf["deg_level"] == lvl]
                    ys.append(row["test_psnr"].values[0] if not row.empty else np.nan)
                ax.plot(levels, ys, "o-",
                        color=STRATEGY_COLORS[strat], label=STRATEGY_LABELS[strat],
                        linewidth=2, markersize=7)
            ax.set_title(f"{ARCH_LABELS[arch]} — {dt.capitalize()}", fontsize=11,
                         fontweight="bold")
            ax.set_xlabel("Degradation level")
            ax.set_ylabel("Test PSNR (dB)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    tag = f"{dataset}_x{scale}"
    fig.suptitle(f"{tag.upper()} — PSNR vs Degradation Level",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"{tag}_06_difficulty_scaling.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading results from: {RESULTS_DIR}")
    test_df, train_df = load_results(RESULTS_DIR)
    print(f"  Test rows: {len(test_df)},  Train rows: {len(train_df)}\n")

    if test_df.empty:
        print("No SR results found. Run run_sr.py first.")
        return

    # Группируем по (dataset, scale)
    for (ds, sc), grp in test_df.groupby(["dataset", "scale"]):
        archs_found = grp["arch"].unique().tolist()
        degs_found  = grp["deg_type"].unique().tolist()
        print(f"── {ds.upper()} x{sc} ({' · '.join(archs_found)}) ─────────────────────")
        print(f"   Degradations: {', '.join(sorted(degs_found))}")

        plot_psnr_overview(test_df,       ds, sc)
        plot_arch_comparison(test_df,     ds, sc)
        plot_heatmap(test_df,             ds, sc)
        plot_curriculum_gain(test_df,     ds, sc)
        plot_training_curves(train_df,    ds, sc)
        plot_difficulty_scaling(test_df,  ds, sc)
        print()

    print(f"All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
