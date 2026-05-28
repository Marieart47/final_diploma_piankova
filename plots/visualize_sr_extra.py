"""Visualization of extra SR experiment analysis plots."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT = Path(__file__).parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

ARCH_COLORS  = {"srcnn": "#C62828", "espcn": "#1565C0", "swinir": "#2E7D32"}
ARCH_LABELS  = {"srcnn": "SRCNN",   "espcn": "ESPCN",   "swinir": "SwinIR"}
ARCH_ORDER   = ["srcnn", "espcn", "swinir"]

STRAT_COLORS = {
    "baseline":  "#2196F3",
    "stagewise": "#4CAF50",
    "static":    "#FF9800",
    "online":    "#E91E63",
}
STRAT_LABELS = {
    "baseline":  "Baseline",
    "stagewise": "Stagewise CL",
    "static":    "Static CL",
    "online":    "Online CL",
}
STRAT_ORDER  = ["baseline", "stagewise", "static", "online"]
CL_STRATS    = ["stagewise", "static", "online"]

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
})

# ── Data loading ───────────────────────────────────────────────────────────────

def load_sr_results():
    rows = []
    for csv in RESULTS_DIR.glob("*/sr_test_results.csv"):
        rows.append(pd.read_csv(csv))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _axis_style(ax):
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Plot 1: Baseline PSNR by arch and dataset ──────────────────────────────────

def plot_baseline_arch_dataset(df):
    configs = [
        ("stl10", 2,  "STL-10 ×2"),
        ("div2k", 4,  "DIV2K ×4"),
    ]
    deg_map = {"original": "Original", "noise": "Noise 0.5", "blur": "Blur 0.5"}
    degs    = ["original", "noise", "blur"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, (ds, scale, title) in zip(axes, configs):
        d = df[(df["dataset"] == ds) & (df["scale"] == scale) &
               (df["strategy"] == "baseline") &
               (df["degradation"].isin(degs))].copy()
        if d.empty:
            ax.set_visible(False)
            continue

        archs_here = [a for a in ARCH_ORDER if a in d["arch"].unique()]
        degs_here  = [dg for dg in degs if dg in d["degradation"].unique()]
        x = np.arange(len(degs_here))
        n, width = len(archs_here), 0.22
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

        for i, arch in enumerate(archs_here):
            heights = []
            for deg in degs_here:
                row = d[(d["arch"] == arch) & (d["degradation"] == deg)]
                heights.append(float(row["test_psnr"].values[0]) if not row.empty else 0.0)
            bars = ax.bar(x + offsets[i], heights, width,
                          label=ARCH_LABELS[arch],
                          color=ARCH_COLORS[arch], alpha=0.85, edgecolor="white")
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                            f"{h:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([deg_map[dg] for dg in degs_here], fontsize=10)
        ax.set_ylabel("Test PSNR (dB)")
        ax.set_ylim(0, d["test_psnr"].max() * 1.12 + 0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        _axis_style(ax)

    fig.suptitle("Baseline PSNR: Architectures and Datasets",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT / "sr_extra_01_baseline_arch_dataset.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 2: CL gain over baseline by arch ─────────────────────────────────────

def plot_cl_gain_all_archs(df):
    configs = [
        ("stl10", 2, "STL-10 ×2"),
        ("div2k", 4, "DIV2K ×4"),
    ]
    degs    = ["original", "noise", "blur"]
    deg_map = {"original": "Original", "noise": "Noise 0.5", "blur": "Blur 0.5"}
    hatches = {"stagewise": "///", "static": "\\\\\\", "online": ""}
    arch_base_colors = {
        "srcnn":  {"stagewise": "#EF9A9A", "static": "#FFCC80", "online": "#F48FB1"},
        "espcn":  {"stagewise": "#81C784", "static": "#4DB6AC", "online": "#64B5F6"},
        "swinir": {"stagewise": "#A5D6A7", "static": "#FFB74D", "online": "#F06292"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    for ax, (ds, scale, title) in zip(axes, configs):
        d = df[(df["dataset"] == ds) & (df["scale"] == scale) &
               (df["degradation"].isin(degs))].copy()
        if d.empty:
            ax.set_visible(False)
            continue

        archs_here = [a for a in ARCH_ORDER if a in d["arch"].unique()]
        degs_here  = [dg for dg in degs if dg in d["degradation"].unique()]
        combos = [(arch, strat) for arch in archs_here for strat in CL_STRATS]
        x = np.arange(len(degs_here))
        n, width = len(combos), 0.07
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

        handles = []
        for i, (arch, strat) in enumerate(combos):
            deltas = []
            for deg in degs_here:
                base = d[(d["arch"] == arch) & (d["strategy"] == "baseline") & (d["degradation"] == deg)]
                cl   = d[(d["arch"] == arch) & (d["strategy"] == strat)      & (d["degradation"] == deg)]
                if not base.empty and not cl.empty:
                    deltas.append(float(cl["test_psnr"].values[0]) - float(base["test_psnr"].values[0]))
                else:
                    deltas.append(0.0)
            color   = arch_base_colors[arch][strat]
            hatch   = hatches[strat]
            bars = ax.bar(x + offsets[i], deltas, width,
                          color=color, hatch=hatch, edgecolor="gray",
                          linewidth=0.5, alpha=0.9)
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, hatch=hatch,
                                         ec="gray", lw=0.5,
                                         label=f"{ARCH_LABELS[arch]} / {STRAT_LABELS[strat]}"))

        ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([deg_map[dg] for dg in degs_here], fontsize=10)
        ax.set_ylabel("ΔPSNR vs Baseline (dB)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(handles=handles, fontsize=7, ncol=2,
                  loc="upper right" if ds == "stl10" else "lower right")
        _axis_style(ax)

    fig.suptitle("CL Gain over Baseline by Architecture",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT / "sr_extra_02_cl_gain_all_archs.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 3: Degradation effect on PSNR ────────────────────────────────────────

def plot_degradation_effect(df):
    configs = [
        ("stl10", 2, "STL-10 ×2"),
        ("div2k", 4, "DIV2K ×4"),
    ]
    degs    = ["original", "noise", "blur"]
    deg_map = {"original": "Original", "noise": "Noise 0.5", "blur": "Blur 0.5"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, (ds, scale, title) in zip(axes, configs):
        d = df[(df["dataset"] == ds) & (df["scale"] == scale) &
               (df["degradation"].isin(degs))].copy()
        if d.empty:
            ax.set_visible(False)
            continue

        archs_here = [a for a in ARCH_ORDER if a in d["arch"].unique()]
        degs_here  = [dg for dg in degs if dg in d["degradation"].unique()]
        x_pos = np.arange(len(degs_here))

        arch_handles = []
        for arch in archs_here:
            for strat in STRAT_ORDER:
                ad = d[(d["arch"] == arch) & (d["strategy"] == strat)]
                ys = []
                for deg in degs_here:
                    row = ad[ad["degradation"] == deg]
                    ys.append(float(row["test_psnr"].values[0]) if not row.empty else np.nan)
                lw = 2.5 if strat == "baseline" else 1.2
                ls = "-"  if strat == "baseline" else "--"
                ax.plot(x_pos, ys, ls, color=ARCH_COLORS[arch],
                        linewidth=lw, marker="o", markersize=5, alpha=0.9)
                if strat == "baseline":
                    arch_handles.append(plt.Line2D([0], [0], color=ARCH_COLORS[arch],
                                                   lw=2.5, label=f"{ARCH_LABELS[arch]}"))

        strat_handles = [
            plt.Line2D([0], [0], color="gray", lw=2.5, ls="-",  label="Baseline"),
            plt.Line2D([0], [0], color="gray", lw=1.2, ls="--", label="CL strategies"),
        ]
        leg1 = ax.legend(handles=arch_handles, loc="upper right", fontsize=8,
                         title="Architecture")
        ax.add_artist(leg1)
        ax.legend(handles=strat_handles, loc="lower left", fontsize=8, title="Strategy")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([deg_map[dg] for dg in degs_here], fontsize=10)
        ax.set_xlabel("Degradation type")
        ax.set_ylabel("Test PSNR (dB)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Effect of Degradation on PSNR (all strategies, baseline highlighted)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT / "sr_extra_03_degradation_effect.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 4: Computational efficiency (steps to stopping) ──────────────────────

def plot_efficiency(df):
    configs = [
        ("stl10", 2, "STL-10 ×2"),
        ("div2k", 4, "DIV2K ×4"),
    ]
    degs    = ["original", "noise", "blur"]
    deg_lbl = {"original": "orig", "noise": "noise", "blur": "blur"}

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    for ax, (ds, scale, title) in zip(axes, configs):
        d = df[(df["dataset"] == ds) & (df["scale"] == scale) &
               (df["degradation"].isin(degs))].copy()
        if d.empty:
            ax.set_visible(False)
            continue

        archs_here = [a for a in ARCH_ORDER if a in d["arch"].unique()]
        degs_here  = [dg for dg in degs if dg in d["degradation"].unique()]
        groups = [(arch, deg) for arch in archs_here for deg in degs_here]
        group_labels = [f"{ARCH_LABELS[a]}\n({deg_lbl[deg]})" for a, deg in groups]
        x = np.arange(len(groups))
        n, width = len(STRAT_ORDER), 0.15
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

        for i, strat in enumerate(STRAT_ORDER):
            heights = []
            for arch, deg in groups:
                row = d[(d["arch"] == arch) & (d["strategy"] == strat) & (d["degradation"] == deg)]
                heights.append(float(row["total_steps"].values[0]) if not row.empty else 0.0)  # noqa
            ax.bar(x + offsets[i], heights, width,
                   label=STRAT_LABELS[strat],
                   color=STRAT_COLORS[strat], alpha=0.85, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=8)
        ax.set_ylabel("Total steps to stopping")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.5)
        _axis_style(ax)

    fig.suptitle("Computational Cost: Steps to Early Stopping",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT / "sr_extra_04_efficiency.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading SR results...")
    df = load_sr_results()
    if df.empty:
        print("No data found.")
    else:
        print(f"  Rows: {len(df)}\n")
        plot_baseline_arch_dataset(df)
        plot_cl_gain_all_archs(df)
        plot_degradation_effect(df)
        plot_efficiency(df)
        print(f"\nAll plots saved to: {OUT}")
