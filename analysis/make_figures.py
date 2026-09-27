#!/usr/bin/env python3
"""Generate publication-ready RA-CoFuzz figures from public aggregate data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


METHODS = [
    "ra_cofuzz",
    "strict_gptfuzzer",
    "pair",
    "tap",
    "renellm",
    "deepinception",
]
METHOD_LABELS = {
    "ra_cofuzz": "RA-CoFuzz",
    "strict_gptfuzzer": "Strict GPTFuzzer",
    "pair": "PAIR",
    "tap": "TAP",
    "renellm": "ReNeLLM",
    "deepinception": "DeepInception",
}
DATASETS = ["gptfuzzer", "advbench", "jailbreakbench"]
DATASET_LABELS = {
    "gptfuzzer": "GPTFuzzer",
    "advbench": "AdvBench",
    "jailbreakbench": "JailbreakBench",
}
MODELS = ["llama32_3b", "qwen25_1_5b", "qwen25_3b", "qwen25_7b", "vicuna_7b"]
MODEL_LABELS = {
    "llama32_3b": "Llama-3.2-3B-Instruct",
    "qwen25_1_5b": "Qwen2.5-1.5B-Instruct",
    "qwen25_3b": "Qwen2.5-3B-Instruct",
    "qwen25_7b": "Qwen2.5-7B-Instruct",
    "vicuna_7b": "Vicuna-7B-v1.5",
}
COLORS = {
    "ra_cofuzz": "#007F7B",
    "strict_gptfuzzer": "#4C78A8",
    "pair": "#F28E2B",
    "tap": "#E15759",
    "renellm": "#8F6BB3",
    "deepinception": "#7A7A7A",
}
LABEL_COLORS = {0: "#D9DEE7", 1: "#F2C14E", 2: "#2A9D8F"}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    sns.set_theme(style="whitegrid", context="paper", font="Nimbus Roman")


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg"]:
        fig.savefig(output_dir / f"{stem}.{suffix}", format=suffix)
    # A 300-dpi raster is sufficient for manuscript review and avoids fragile,
    # very large PNG streams; PDF/SVG remain the camera-ready master formats.
    fig.savefig(output_dir / f"{stem}.png", format="png", dpi=300)
    plt.close(fig)


def lookup(rows: list[dict], *keys: str) -> dict[tuple, dict]:
    return {tuple(row[k] for k in keys): row for row in rows}


def mean(row: dict, metric: str, scale: float = 1.0) -> float:
    return float(row[metric]["mean"]) * scale


def sd(row: dict, metric: str, scale: float = 1.0) -> float:
    return float(row[metric]["sample_sd"]) * scale


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=9)


def rounded_box(ax, xy, width, height, text, fc, ec="#1A1A1A", fontsize=8, lw=0.8):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, color="#333333", style="-|>", connectionstyle="arc3", lw=1.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=lw,
            color=color,
            linestyle=ls,
            connectionstyle=connectionstyle,
        )
    )


def draw_node_icon(ax, kind: str, cx: float, cy: float, scale: float = 0.033) -> None:
    """Draw a compact vector pictogram in data coordinates."""
    color = "#263238"
    lw = 0.9
    if kind == "pool":
        ax.add_patch(Rectangle((cx - scale, cy - scale * 0.55), 2 * scale, scale * 1.1, facecolor="#DCE8F3", edgecolor=color, lw=lw))
        ax.add_patch(Ellipse((cx, cy + scale * 0.55), 2 * scale, scale * 0.55, facecolor="#EDF4FA", edgecolor=color, lw=lw))
        ax.add_patch(Ellipse((cx, cy - scale * 0.55), 2 * scale, scale * 0.55, facecolor="none", edgecolor=color, lw=lw))
        for offset in [-0.15, 0.18]:
            ax.add_patch(Arc((cx, cy + scale * offset), 2 * scale, scale * 0.5, theta1=180, theta2=360, color=color, lw=0.6))
    elif kind == "selector":
        top = (cx, cy + scale * 0.55)
        leaves = [(cx - scale * 0.8, cy - scale * 0.45), (cx, cy - scale * 0.45), (cx + scale * 0.8, cy - scale * 0.45)]
        for leaf in leaves:
            ax.plot([top[0], leaf[0]], [top[1], leaf[1]], color=color, lw=0.8)
        ax.add_patch(Circle(top, scale * 0.22, facecolor="#4C78A8", edgecolor=color, lw=0.6))
        for leaf, fc in zip(leaves, ["#D6E7F5", "#FCE3C7", "#E7DDF2"]):
            ax.add_patch(Circle(leaf, scale * 0.23, facecolor=fc, edgecolor=color, lw=0.6))
    elif kind == "mutation":
        arrow(ax, (cx - scale * 0.8, cy + scale * 0.1), (cx + scale * 0.65, cy + scale * 0.25), color="#C47B18", connectionstyle="arc3,rad=-0.55", lw=0.9)
        arrow(ax, (cx + scale * 0.8, cy - scale * 0.1), (cx - scale * 0.65, cy - scale * 0.25), color="#C47B18", connectionstyle="arc3,rad=-0.55", lw=0.9)
    elif kind == "model":
        ax.add_patch(Rectangle((cx - scale * 0.68, cy - scale * 0.62), scale * 1.36, scale * 1.24, facecolor="#F8D9DC", edgecolor=color, lw=lw))
        ax.add_patch(Rectangle((cx - scale * 0.34, cy - scale * 0.30), scale * 0.68, scale * 0.60, facecolor="white", edgecolor=color, lw=0.7))
        for offset in [-0.42, 0, 0.42]:
            ax.plot([cx - scale * 0.92, cx - scale * 0.68], [cy + scale * offset, cy + scale * offset], color=color, lw=0.7)
            ax.plot([cx + scale * 0.68, cx + scale * 0.92], [cy + scale * offset, cy + scale * offset], color=color, lw=0.7)
    elif kind == "response":
        ax.add_patch(Rectangle((cx - scale * 0.7, cy - scale * 0.62), scale * 1.4, scale * 1.24, facecolor="white", edgecolor=color, lw=lw))
        for offset, width in [(0.30, 0.85), (0.0, 0.95), (-0.30, 0.65)]:
            ax.plot([cx - scale * 0.45, cx - scale * 0.45 + scale * width], [cy + scale * offset, cy + scale * offset], color="#708090", lw=0.8)
    elif kind == "judge":
        shield = Polygon(
            [(cx, cy + scale * 0.75), (cx + scale * 0.72, cy + scale * 0.42), (cx + scale * 0.55, cy - scale * 0.45), (cx, cy - scale * 0.78), (cx - scale * 0.55, cy - scale * 0.45), (cx - scale * 0.72, cy + scale * 0.42)],
            closed=True,
            facecolor="#EDF7F6",
            edgecolor=color,
            lw=lw,
        )
        ax.add_patch(shield)
        for dx, fc in [(-0.36, LABEL_COLORS[0]), (0.0, LABEL_COLORS[1]), (0.36, LABEL_COLORS[2])]:
            ax.add_patch(Circle((cx + scale * dx, cy), scale * 0.17, facecolor=fc, edgecolor=color, lw=0.45))
    elif kind == "fitness":
        ax.add_patch(Arc((cx, cy - scale * 0.1), 2 * scale, 1.35 * scale, theta1=0, theta2=180, color=color, lw=lw))
        ax.plot([cx, cx + scale * 0.48], [cy - scale * 0.1, cy + scale * 0.28], color="#8F6BB3", lw=1.0)
        ax.add_patch(Circle((cx, cy - scale * 0.1), scale * 0.12, facecolor="#8F6BB3", edgecolor=color, lw=0.5))
    elif kind == "update":
        arrow(ax, (cx - scale * 0.72, cy), (cx + scale * 0.55, cy + scale * 0.18), color="#8F6BB3", connectionstyle="arc3,rad=-0.65", lw=0.9)
        arrow(ax, (cx + scale * 0.72, cy), (cx - scale * 0.55, cy - scale * 0.18), color="#8F6BB3", connectionstyle="arc3,rad=-0.65", lw=0.9)
    elif kind == "assembly":
        ax.add_patch(Circle((cx, cy), scale * 0.72, facecolor="white", edgecolor=color, lw=lw))
        ax.plot([cx - scale * 0.34, cx + scale * 0.34], [cy, cy], color=color, lw=1.0)
        ax.plot([cx, cx], [cy - scale * 0.34, cy + scale * 0.34], color=color, lw=1.0)
    elif kind == "offline":
        ax.plot([cx - scale * 0.7, cx - scale * 0.7, cx + scale * 0.72], [cy + scale * 0.55, cy - scale * 0.6, cy - scale * 0.6], color=color, lw=0.75)
        for xoff, height, fc in [(-0.4, 0.45, "#D9DEE7"), (0.0, 0.75, "#F2C14E"), (0.4, 1.0, "#2A9D8F")]:
            ax.add_patch(Rectangle((cx + scale * xoff - scale * 0.13, cy - scale * 0.58), scale * 0.26, scale * height, facecolor=fc, edgecolor=color, lw=0.45))


def icon_card(ax, center, label, subtitle, facecolor, icon_kind, width=0.17, height=0.145):
    cx, cy = center
    patch = FancyBboxPatch(
        (cx - width / 2, cy - height / 2),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.020",
        facecolor=facecolor,
        edgecolor="#1A1A1A",
        linewidth=0.8,
        zorder=0.8,
    )
    ax.add_patch(patch)
    draw_node_icon(ax, icon_kind, cx, cy + height * 0.22, scale=0.025)
    ax.text(cx, cy - height * 0.10, label, ha="center", va="center", fontsize=7.2, fontweight="semibold", zorder=3)
    if subtitle:
        ax.text(cx, cy - height * 0.34, subtitle, ha="center", va="center", fontsize=5.9, color="#555555", zorder=3)
    return patch


def inline_icon_card(ax, center, label, subtitle, facecolor, icon_kind, width=0.18, height=0.09):
    cx, cy = center
    patch = FancyBboxPatch(
        (cx - width / 2, cy - height / 2),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.016",
        facecolor=facecolor,
        edgecolor="#4C5661",
        linewidth=0.7,
        zorder=0.8,
    )
    ax.add_patch(patch)
    icon_x = cx - width * 0.32
    draw_node_icon(ax, icon_kind, icon_x, cy, scale=0.017)
    text_x = cx - width * 0.15
    ax.text(text_x, cy + (0.012 if subtitle else 0), label, ha="left", va="center", fontsize=6.4, fontweight="semibold", zorder=3)
    if subtitle:
        ax.text(text_x, cy - 0.018, subtitle, ha="left", va="center", fontsize=5.2, color="#5D6670", zorder=3)
    return patch


def figure_framework(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.16, 4.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = [
        ((0.16, 0.79), "Template pool", "77 seeds + offspring", "#EAF1F8", "pool", 0.18),
        ((0.39, 0.87), "Hybrid selector", "MCTS 0.50 · DP 0.30 · Elite 0.20", "#E8F5F4", "selector", 0.20),
        ((0.62, 0.80), "Mutation", "5 operators", "#FFF2D8", "mutation", 0.17),
        ((0.85, 0.49), "Target model", "black-box", "#FDE8E8", "model", 0.17),
        ((0.83, 0.25), "Response", r"candidate $r_i$", "#F3F3F3", "response", 0.17),
        ((0.64, 0.16), "Two-stage judge", "weak → selective | L0/L1/L2", "#E8F5F4", "judge", 0.19),
        ((0.39, 0.12), "Candidate fitness", "label + attributes", "#F3EAF8", "fitness", 0.18),
        ((0.15, 0.31), "State update", "MCTS • DP • Elite", "#EEE7F5", "update", 0.18),
    ]
    for center, label, subtitle, fc, icon, width in nodes:
        icon_card(ax, center, label, subtitle, fc, icon, width=width)

    # External query input and the prompt-assembly junction.
    inline_icon_card(ax, (0.86, 0.86), "Evaluation query", r"$q_i$", "#F5F7FA", "response", width=0.15, height=0.082)
    inline_icon_card(ax, (0.75, 0.65), "Assembly", r"template + $q_i$", "#F5F7FA", "assembly", width=0.135, height=0.075)

    # Clockwise online loop. Three colors distinguish search, feedback and update.
    search_edges = [
        ((0.25, 0.81), (0.29, 0.85), -0.10),
        ((0.49, 0.85), (0.535, 0.82), 0.08),
        ((0.68, 0.727), (0.70, 0.686), 0.00),
        ((0.79, 0.613), (0.80, 0.565), -0.03),
        ((0.85, 0.417), (0.835, 0.325), 0.05),
    ]
    for start, end, rad in search_edges:
        arrow(ax, start, end, color="#333333", connectionstyle=f"arc3,rad={rad}", lw=1.15)
    arrow(ax, (0.825, 0.825), (0.785, 0.686), color="#4C78A8", connectionstyle="arc3,rad=0.08", lw=0.95)
    feedback_edges = [
        ((0.755, 0.205), (0.705, 0.185), 0.10),
        ((0.545, 0.14), (0.48, 0.12), 0.06),
    ]
    for start, end, rad in feedback_edges:
        arrow(ax, start, end, color="#007F7B", connectionstyle=f"arc3,rad={rad}", lw=1.2)
    update_edges = [((0.30, 0.13), (0.23, 0.27), -0.12)]
    for start, end, rad in update_edges:
        arrow(ax, start, end, color="#8F6BB3", connectionstyle=f"arc3,rad={rad}", lw=1.2)

    # Explicit stopping decision closes the loop or releases the result set.
    stop = Polygon([(0.105, 0.61), (0.145, 0.56), (0.105, 0.51), (0.065, 0.56)], closed=True, facecolor="#FFFFFF", edgecolor="#8F6BB3", lw=0.9, zorder=2)
    ax.add_patch(stop)
    ax.text(0.105, 0.56, "Stop?", ha="center", va="center", fontsize=5.8, fontweight="semibold", zorder=3)
    arrow(ax, (0.14, 0.385), (0.10, 0.51), color="#8F6BB3", connectionstyle="arc3,rad=-0.08", lw=1.2)
    arrow(ax, (0.105, 0.61), (0.135, 0.715), color="#8F6BB3", connectionstyle="arc3,rad=-0.08", lw=1.2)
    ax.text(0.062, 0.665, "continue", fontsize=5.4, color="#8F6BB3", rotation=68)

    # Center identifies the method; the completed branch is outside the online loop.
    ax.text(0.47, 0.60, "RA-CoFuzz", ha="center", va="center", fontsize=12, fontweight="bold", color="#31445A")
    ax.text(0.47, 0.565, "refusal-aware hybrid co-fuzzing", ha="center", va="center", fontsize=6.5, color="#66717E")
    inline_icon_card(ax, (0.30, 0.45), "Candidate set", "completed search", "#F5F7FA", "pool", width=0.17, height=0.09)
    inline_icon_card(ax, (0.54, 0.43), "Offline evaluation", "StrongJudge + RoBERTa", "#F5F7FA", "offline", width=0.21, height=0.09)
    arrow(ax, (0.145, 0.56), (0.215, 0.47), color="#666666", connectionstyle="arc3,rad=0.06", lw=0.9, ls="--")
    ax.text(0.165, 0.515, "finish", fontsize=5.4, color="#666666", rotation=-30)
    arrow(ax, (0.385, 0.45), (0.435, 0.435), color="#666666", lw=0.9, ls="--")

    legend_handles = [
        Line2D([0], [0], color="#333333", lw=1.4, label="Search"),
        Line2D([0], [0], color="#4C78A8", lw=1.2, label="Query input"),
        Line2D([0], [0], color="#007F7B", lw=1.4, label="Feedback"),
        Line2D([0], [0], color="#8F6BB3", lw=1.4, label="Update"),
        Line2D([0], [0], color="#666666", lw=1.2, ls="--", label="Post hoc"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=5, frameon=False, fontsize=6.2, handlelength=1.8, columnspacing=1.0)
    save(fig, output_dir, "fig01_framework_overview")


def figure_main(data: dict, output_dir: Path) -> None:
    rows = sorted(data["main_comparison"], key=lambda r: METHODS.index(r["method"]))

    fig, ax = plt.subplots(figsize=(4.1, 3.15))
    y = np.arange(len(rows))
    values = [mean(r, "question_label2_rate", 100) for r in rows]
    errors = [sd(r, "question_label2_rate", 100) for r in rows]
    ax.barh(y, values, color=[COLORS[r["method"]] for r in rows], alpha=0.90, height=0.62)
    ax.errorbar(values, y, xerr=errors, fmt="none", ecolor="#222222", capsize=2.5, lw=0.8)
    ax.set_yticks(y, [METHOD_LABELS[r["method"]] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("Question-level Label-2 ASR (%)")
    ax.set_title("Coverage effectiveness")
    for yi, value in zip(y, values):
        ax.text(min(value + 2, 102), yi, f"{value:.1f}", va="center", fontsize=7)
    fig.tight_layout()
    save(fig, output_dir, "fig02a_main_question_asr")

    fig, ax = plt.subplots(figsize=(5.2, 3.15))
    x = np.arange(len(rows))
    width = 0.23
    metrics = [
        ("response_label2_rate", "Label-2", "#2A9D8F"),
        ("response_label1_or_2_rate", "Label-(1+2)", "#F2C14E"),
        ("roberta_response_rate", "RoBERTa", "#4C78A8"),
    ]
    for j, (metric, label, color) in enumerate(metrics):
        vals = [mean(r, metric, 100) for r in rows]
        errs = [sd(r, metric, 100) for r in rows]
        ax.bar(x + (j - 1) * width, vals, width, label=label, color=color, alpha=0.90, yerr=errs, capsize=2, linewidth=0)
    short_labels = ["RA-CoFuzz", "Strict\nGPTFuzzer", "PAIR", "TAP", "ReNeLLM", "Deep-\nInception"]
    ax.set_xticks(x, short_labels)
    ax.set_ylim(0, 70)
    ax.set_ylabel("Response-level ASR / rate (%)")
    ax.set_title("Response-level outcomes")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    save(fig, output_dir, "fig02b_main_response_rates")


def heat_matrix(rows: list[dict], columns: list[str], column_key: str, metric: str) -> np.ndarray:
    table = lookup(rows, "method", column_key)
    return np.array([[mean(table[(m, c)], metric, 100) for c in columns] for m in METHODS])


def figure_cross_dataset(data: dict, output_dir: Path) -> None:
    rows = data["cross_dataset"]
    figures = [
        ("question_label2_rate", "Question-level Label-2 ASR", "fig03a_cross_dataset_question_asr"),
        ("response_label2_rate", "Response-level Label-2 ASR", "fig03b_cross_dataset_response_asr"),
    ]
    for metric, title, stem in figures:
        fig, ax = plt.subplots(figsize=(4.9, 3.6))
        mat = heat_matrix(rows, DATASETS, "dataset", metric)
        sns.heatmap(
            mat,
            ax=ax,
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 7},
            cbar=True,
            cbar_kws={"label": "%", "shrink": 0.82},
            linewidths=0.7,
            linecolor="white",
        )
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], rotation=25, ha="right")
        ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], rotation=0)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.tight_layout()
        save(fig, output_dir, stem)


def figure_cross_model(data: dict, output_dir: Path) -> None:
    rows = data["cross_model"]
    figures = [
        ("question_label2_rate", "Question-level Label-2 ASR across target models", "fig04a_cross_model_question_asr"),
        ("response_label2_rate", "Response-level Label-2 ASR across target models", "fig04b_cross_model_response_asr"),
    ]
    for metric, title, stem in figures:
        fig, ax = plt.subplots(figsize=(7.16, 3.35))
        mat = heat_matrix(rows, MODELS, "model", metric)
        sns.heatmap(
            mat,
            ax=ax,
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 7},
            cbar=True,
            cbar_kws={"label": "%", "shrink": 0.82},
            linewidths=0.7,
            linecolor="white",
        )
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=16, ha="right")
        ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], rotation=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title)
        fig.tight_layout()
        save(fig, output_dir, stem)


def ablation_panel(ax, rows, variants, metric, title, labels):
    table = {r["variant"]: r for r in rows}
    vals = [mean(table[v], metric, 100) for v in variants]
    errs = [sd(table[v], metric, 100) for v in variants]
    colors = ["#007F7B"] + ["#9FBACF"] * (len(variants) - 1)
    x = np.arange(len(variants))
    ax.bar(x, vals, color=colors, yerr=errs, capsize=2.5, width=0.68)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Rate (%)")
    ax.set_title(title)
    full = vals[0]
    for xi, value in enumerate(vals):
        text = f"{value:.1f}" if xi == 0 else f"{value:.1f}\n({value-full:+.1f})"
        ax.text(xi, min(value + errs[xi] + 3, 108), text, ha="center", va="bottom", fontsize=7)


def figure_selection_ablation(data: dict, output_dir: Path) -> None:
    variants = ["full", "hybrid_wo_dp", "hybrid_wo_elite", "hybrid_wo_mcts"]
    labels = ["Full", "w/o DP", "w/o Elite", "w/o MCTS"]
    rows = data["selection_ablations"]
    specs = [
        ("question_label2_rate", "Question-level Label-2 ASR", "fig05a_selection_question_asr"),
        ("response_label2_rate", "Response-level Label-2 ASR", "fig05b_selection_response_asr"),
    ]
    for metric, title, stem in specs:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ablation_panel(ax, rows, variants, metric, title, labels)
        fig.tight_layout()
        save(fig, output_dir, stem)


def figure_feedback_ablation(data: dict, output_dir: Path) -> None:
    variants = ["full", "wo_label1_reward", "wo_online_strongjudge_feedback", "full_online_review"]
    labels = ["Full", "w/o L1 reward", "w/o online judge", "Full online review"]
    table = {r["variant"]: r for r in data["feedback_ablations"]}
    specs = [
        ("question_label2_rate", "Question-level Label-2 ASR", 100, "ASR (%)", "fig06a_feedback_question_asr"),
        ("response_label2_rate", "Response-level Label-2 ASR", 100, "ASR (%)", "fig06b_feedback_response_asr"),
        ("online_judge_calls", "Online StrongJudge calls", 1, "Calls", "fig06c_feedback_judge_calls"),
    ]
    for metric, title, scale, ylabel, stem in specs:
        fig, ax = plt.subplots(figsize=(5.0, 3.05))
        vals = [mean(table[v], metric, scale) for v in variants]
        errs = [sd(table[v], metric, scale) for v in variants]
        y = np.arange(4)
        ax.barh(y, vals, color=["#007F7B", "#F2C14E", "#4C78A8", "#E15759"], xerr=errs, capsize=2.2, height=0.64)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(ylabel)
        upper = 115 if scale == 100 else 235
        ax.set_xlim(0, upper)
        for yi, value in enumerate(vals):
            ax.text(min(value + errs[yi] + upper * 0.025, upper * 0.93), yi, f"{value:.1f}", va="center", fontsize=6.8)
        fig.tight_layout()
        save(fig, output_dir, stem)


def figure_efficiency(data: dict, output_dir: Path) -> None:
    rows = sorted(data["main_comparison"], key=lambda r: METHODS.index(r["method"]))
    offsets = {
        "ra_cofuzz": (4, 1.5), "strict_gptfuzzer": (-55, -14), "pair": (-32, 8),
        "tap": (-38, -13), "renellm": (4, 2), "deepinception": (4, 2),
    }
    specs = [
        ("target_calls", "Mean target calls", False, "Effectiveness vs. target-query use", "fig07a_efficiency_target_calls"),
        ("runtime_seconds", "Mean runtime (s, log scale)", True, "Effectiveness vs. wall-clock runtime", "fig07b_efficiency_runtime"),
    ]
    for xmetric, xlabel, logx, title, stem in specs:
        fig, ax = plt.subplots(figsize=(4.8, 3.25))
        for row in rows:
            method = row["method"]
            x = mean(row, xmetric)
            y = mean(row, "question_label2_rate", 100)
            size = 30 + min(mean(row, "recorded_responses"), 200) * 0.45
            ax.scatter(x, y, s=size, color=COLORS[method], edgecolor="white", linewidth=0.7, zorder=3)
            dx, dy = offsets[method]
            ax.annotate(METHOD_LABELS[method], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Question-level Label-2 ASR (%)")
        ax.set_ylim(-4, 105)
        if logx:
            ax.set_xscale("log")
        ax.axhline(50, color="#BBBBBB", lw=0.7, ls="--")
        ax.set_title(title)
        fig.tight_layout()
        save(fig, output_dir, stem)


def figure_evaluator_alignment(data: dict, output_dir: Path) -> None:
    rows = data["evaluator_alignment"]
    markers = {"main": "o", "dataset": "s", "model": "^"}
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    for method in METHODS:
        for row in [r for r in rows if r["method"] == method]:
            if row["dataset"] == "gptfuzzer" and row["model"] == "llama32_3b":
                scope = "main"
            elif row["model"] == "llama32_3b":
                scope = "dataset"
            else:
                scope = "model"
            ax.scatter(
                mean(row, "question_label2_rate", 100),
                mean(row, "roberta_question_rate", 100),
                marker=markers[scope],
                s=42,
                color=COLORS[method],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.5,
            )
    ax.plot([0, 100], [0, 100], color="#555555", ls="--", lw=0.9, label="Agreement line")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("StrongJudge question-level success rate (%)")
    ax.set_ylabel("RoBERTa question-level success rate (%)")
    ax.set_title("Evaluator alignment across 42 method–condition aggregates")
    method_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[m], markeredgecolor="white", markersize=6, label=METHOD_LABELS[m]) for m in METHODS]
    scope_handles = [Line2D([0], [0], marker=markers[s], color="#555555", linestyle="none", markersize=5, label={"main":"Main", "dataset":"Cross-dataset", "model":"Cross-model"}[s]) for s in markers]
    leg1 = ax.legend(handles=method_handles, frameon=False, loc="lower right", ncol=2, title="Method")
    ax.add_artist(leg1)
    ax.legend(handles=scope_handles, frameon=False, loc="upper left", title="Condition")
    fig.tight_layout()
    save(fig, output_dir, "fig08_evaluator_alignment")


def figure_seed_stability(data: dict, output_dir: Path) -> None:
    cells = [r for r in data["cells"] if r["phase"] == "main"]
    table = {(r["method"], r["seed"]): r for r in cells}
    specs = [
        ("question_label2_rate", "Question-level Label-2 ASR stability", 105, "fig09a_question_stability"),
        ("response_label2_rate", "Response-level Label-2 ASR stability", 45, "fig09b_response_stability"),
    ]
    for metric, title, upper, stem in specs:
        fig, ax = plt.subplots(figsize=(4.8, 3.45))
        for method in METHODS:
            values = [table[(method, seed)][metric] * 100 for seed in [100, 200, 300]]
            ax.plot([100, 200, 300], values, marker="o", ms=4, lw=1.3, color=COLORS[method], label=METHOD_LABELS[method])
        ax.set_xticks([100, 200, 300])
        ax.set_xlabel("Released run label")
        ax.set_ylabel("Label-2 ASR (%)")
        ax.set_ylim(-3, upper)
        ax.set_title(title)
        ax.legend(frameon=False, ncol=2, loc="best", fontsize=6.5)
        fig.tight_layout()
        save(fig, output_dir, stem)


def figure_response_composition(data: dict, output_dir: Path) -> None:
    cells = [r for r in data["cells"] if r["phase"] == "main"]
    counts = {m: defaultdict(int) for m in METHODS}
    for row in cells:
        for label in [0, 1, 2]:
            counts[row["method"]][label] += int(row[f"label{label}_count"])
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(METHODS))
    left = np.zeros(len(METHODS))
    for label, name in [(0, "Label 0"), (1, "Label 1"), (2, "Label 2")]:
        values = []
        for method in METHODS:
            total = sum(counts[method].values())
            values.append(100 * counts[method][label] / total)
        ax.barh(y, values, left=left, color=LABEL_COLORS[label], edgecolor="white", linewidth=0.5, label=name, height=0.65)
        for yi, (start, value) in enumerate(zip(left, values)):
            if value >= 7:
                ax.text(start + value / 2, yi, f"{value:.1f}", ha="center", va="center", fontsize=7)
        left += np.array(values)
    ax.set_yticks(y, [METHOD_LABELS[m] for m in METHODS])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of generated responses (%)")
    ax.set_title("Main-setting response-label composition pooled across three runs", pad=30)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    save(fig, output_dir, "fig10_response_composition")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    figure_framework(args.output_dir)
    figure_main(data, args.output_dir)
    figure_cross_dataset(data, args.output_dir)
    figure_cross_model(data, args.output_dir)
    figure_selection_ablation(data, args.output_dir)
    figure_feedback_ablation(data, args.output_dir)
    figure_efficiency(data, args.output_dir)
    figure_evaluator_alignment(data, args.output_dir)
    figure_seed_stability(data, args.output_dir)
    figure_response_composition(data, args.output_dir)
    print(json.dumps({"status": "FIGURE_GENERATION_PASS", "figures": 18, "formats": ["pdf", "svg", "png"]}))


if __name__ == "__main__":
    main()
