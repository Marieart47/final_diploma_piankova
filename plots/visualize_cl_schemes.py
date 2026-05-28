"""Diagrams illustrating each Curriculum Learning strategy."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
EASY_C   = "#2E7D32"
MED_C    = "#F57F17"
HARD_C   = "#C62828"
ALL_C    = "#1565C0"
SEL_C    = "#6A1B9A"
BOX_C    = "#ECEFF1"
STEP_C   = "#263238"
ARROW_C  = "#546E7A"
WHITE    = "#FFFFFF"

np.random.seed(42)

# ── Helpers ────────────────────────────────────────────────────────────────────

def fbox(ax, cx, cy, w, h, label, fc=BOX_C, ec="#90A4AE",
         tc=STEP_C, fs=9, lw=1.5, pad=0.08, bold=True):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle=f"round,pad={pad}",
                          facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            multialignment="center", zorder=4)

def arr(ax, x1, y1, x2, y2, lbl="", lbl_offset=(0, 0.07), lw=1.8, color=ARROW_C):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14), zorder=5)
    if lbl:
        ax.text((x1+x2)/2 + lbl_offset[0], (y1+y2)/2 + lbl_offset[1],
                lbl, ha="center", va="bottom", fontsize=7.5, color=color)

def scatter_dots(ax, cx, cy, n, color, spread_x=0.28, spread_y=0.16):
    xs = cx + (np.random.rand(n) - 0.5) * 2 * spread_x
    ys = cy + (np.random.rand(n) - 0.5) * 2 * spread_y
    ax.scatter(xs, ys, s=18, color=color, alpha=0.80, zorder=6, edgecolors="white", lw=0.4)

def strategy_title(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=13,
            fontweight="bold", color=STEP_C, zorder=6,
            bbox=dict(boxstyle="round,pad=0.35", fc="#E3F2FD", ec="#1565C0", lw=1.5))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE  –  3 rows, one per strategy
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.patch.set_facecolor("white")
plt.subplots_adjust(hspace=0.18, top=0.94, bottom=0.03)

# ─────────────────────────────────────────────────────────────────────────────
# Row 0 : Stagewise CL
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0.3, 3.5)
ax.axis("off")

strategy_title(ax, 5, 3.32, "Stagewise CL")

# Budget bar — taller so text fits larger
bar_y, bar_h = 2.45, 0.80
bx0, bw = 0.7, 8.6
stage_colors = [EASY_C, MED_C, HARD_C]
stage_labels = ["Stage 1\nEasy examples\n0 → ⅓ budget",
                "Stage 2\nMedium examples\n⅓ → ⅔ budget",
                "Stage 3\nHard examples\n⅔ → 1 budget"]
sw = bw / 3

for i, (col, lbl) in enumerate(zip(stage_colors, stage_labels)):
    x = bx0 + i * sw
    rect = FancyBboxPatch((x + 0.05, bar_y - bar_h/2), sw - 0.10, bar_h,
                          boxstyle="round,pad=0.05",
                          facecolor=col, edgecolor="white", linewidth=2, alpha=0.88, zorder=3)
    ax.add_patch(rect)
    ax.text(x + sw/2, bar_y, lbl, ha="center", va="center",
            fontsize=11, color=WHITE, fontweight="bold", multialignment="center", zorder=4)

# Tick marks + budget arrow below bar
tick_y = bar_y - bar_h/2 - 0.08
for i in range(4):
    ax.plot([bx0 + i*sw, bx0 + i*sw], [tick_y, tick_y - 0.18],
            color=ARROW_C, lw=1.3)
for i, tl in enumerate(["0", "⅓", "⅔", "1"]):
    ax.text(bx0 + i*sw, tick_y - 0.24, tl, ha="center", va="top",
            fontsize=10, color=ARROW_C, fontweight="bold")
ax.annotate("", xy=(bx0 + bw + 0.15, tick_y - 0.12),
            xytext=(bx0 - 0.15, tick_y - 0.12),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=1.5, mutation_scale=13))
ax.text(bx0 + bw/2, tick_y - 0.50, "Training budget",
        ha="center", va="top", fontsize=9.5, color=ARROW_C, style="italic")

# Sample dots
for i, col in enumerate(stage_colors):
    scatter_dots(ax, bx0 + i*sw + sw/2, 1.12, 12, col, spread_x=0.32, spread_y=0.14)

# Legend
patches = [mpatches.Patch(color=c, label=l)
           for c, l in zip(stage_colors, ["Easy", "Medium", "Hard"])]
ax.legend(handles=patches, loc="upper left", fontsize=9,
          ncol=3, bbox_to_anchor=(0.01, 1.0), framealpha=0.85)

# ─────────────────────────────────────────────────────────────────────────────
# Row 1 : Static CL
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0.7, 3.5)
ax.axis("off")

strategy_title(ax, 5, 3.32, "Static CL")

boxes = [
    (1.1,  2.35, "Warm-up\n(1 epoch,\nall data)",             ALL_C,      WHITE),
    (3.3,  2.35, "Compute loss\nper sample",                   BOX_C,      STEP_C),
    (5.5,  2.35, "Sort & split\ninto fixed groups\nE | M | H", BOX_C,      STEP_C),
    (7.9,  2.35, "Cycle training\nE → M → H\n(fixed groups)",  "#E8F5E9",  EASY_C),
]
for cx, cy, lbl, fc, tc in boxes:
    fbox(ax, cx, cy, 1.65, 0.85, lbl, fc=fc, tc=tc, fs=9)

for (cx1,_,__,___,____), (cx2,_,__,___,____) in zip(boxes, boxes[1:]):
    arr(ax, cx1 + 1.65/2 + 0.04, 2.35, cx2 - 1.65/2 - 0.04, 2.35)

# Color group boxes under "Sort & split"
gx0, gy = 4.68, 1.42
for i, (col, lbl) in enumerate(zip([EASY_C, MED_C, HARD_C], ["Easy", "Med", "Hard"])):
    gx = gx0 + i * 0.63
    rect = FancyBboxPatch((gx, gy - 0.18), 0.57, 0.36,
                          boxstyle="round,pad=0.04",
                          facecolor=col, edgecolor="white", linewidth=1.5, alpha=0.85, zorder=3)
    ax.add_patch(rect)
    ax.text(gx + 0.285, gy, lbl, ha="center", va="center",
            fontsize=8, color=WHITE, fontweight="bold", zorder=4)
ax.annotate("", xy=(gx0 + 0.285, gy + 0.18),
            xytext=(5.5, 2.35 - 0.425),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=1.2, mutation_scale=10))

# Cycle arrow
arr(ax, 7.9 + 1.65/2, 1.93, 7.9 + 1.65/2 + 0.12, 1.5)
ax.annotate("", xy=(7.9 - 1.65/2, 1.5),
            xytext=(7.9 + 1.65/2 + 0.12, 1.5),
            arrowprops=dict(arrowstyle="-|>", color=EASY_C, lw=1.5, mutation_scale=12))
ax.text(7.9, 1.30, "repeat each epoch", ha="center", va="top",
        fontsize=8, color=EASY_C, style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# Row 2 : Online CL
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(0, 10); ax.set_ylim(0.4, 3.5)
ax.axis("off")

strategy_title(ax, 5, 3.32, "Online CL")

loop_boxes = [
    (1.2,  2.5, "All\nsamples",                    ALL_C,     WHITE),
    (3.1,  2.5, "Rank by\ncurrent loss\n(ascending)", BOX_C,  STEP_C),
    (5.2,  2.5, "Select\neasiest 30%\n(low loss)",  SEL_C,    WHITE),
    (7.3,  2.5, "Train one\nepoch",                "#E8F5E9", EASY_C),
    (9.1,  2.5, "Update\nlosses",                   BOX_C,    STEP_C),
]
for cx, cy, lbl, fc, tc in loop_boxes:
    fbox(ax, cx, cy, 1.5, 0.82, lbl, fc=fc, tc=tc, fs=9)

for (cx1,_,__,___,____), (cx2,_,__,___,____) in zip(loop_boxes, loop_boxes[1:]):
    arr(ax, cx1 + 1.5/2 + 0.04, 2.5, cx2 - 1.5/2 - 0.04, 2.5)

# Feedback loop
ax.annotate("",
            xy=(3.1 - 1.5/2, 2.09),
            xytext=(9.1 + 1.5/2, 2.09),
            arrowprops=dict(arrowstyle="-|>", color=SEL_C, lw=1.8, mutation_scale=14,
                            connectionstyle="arc3,rad=-0.32"))
ax.text(6.15, 0.95, "← repeat each epoch  (selection adapts as model improves) →",
        ha="center", va="center", fontsize=8.5, color=SEL_C, style="italic")

# Selection bar illustration
bx, by = 3.95, 1.55
bar_total = 2.5
ax.text(bx + bar_total/2, by + 0.22, "ranked samples →",
        ha="center", va="bottom", fontsize=7.5, color=ARROW_C)
n_easy, n_rest = 9, 21
frac = n_easy / (n_easy + n_rest)
rect_e = FancyBboxPatch((bx, by - 0.13), bar_total * frac, 0.26,
                         boxstyle="round,pad=0.02",
                         facecolor=SEL_C, edgecolor="white", lw=1.2, alpha=0.85, zorder=3)
rect_r = FancyBboxPatch((bx + bar_total * frac, by - 0.13), bar_total * (1 - frac), 0.26,
                         boxstyle="round,pad=0.02",
                         facecolor="#BDBDBD", edgecolor="white", lw=1.2, alpha=0.5, zorder=3)
ax.add_patch(rect_e); ax.add_patch(rect_r)
ax.text(bx + bar_total * frac / 2, by,
        "30% selected", ha="center", va="center",
        fontsize=7, color=WHITE, fontweight="bold", zorder=4)
ax.text(bx + bar_total * frac + bar_total * (1 - frac) / 2, by,
        "70% skipped", ha="center", va="center",
        fontsize=7, color="#616161", fontweight="bold", zorder=4)
ax.annotate("", xy=(5.2, 2.5 - 0.41), xytext=(bx + bar_total * 0.18, by + 0.13),
            arrowprops=dict(arrowstyle="-|>", color=SEL_C, lw=1.1, mutation_scale=9))

# ── Save ───────────────────────────────────────────────────────────────────────
fig.suptitle("Curriculum Learning Strategies: Visual Schemes",
             fontsize=15, fontweight="bold")
out = OUT / "cl_strategies_schemes.png"
plt.savefig(out, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
