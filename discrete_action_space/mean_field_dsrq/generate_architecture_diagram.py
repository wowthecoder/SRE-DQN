"""Generate the robust mean-field DSRQ architecture diagram.

The output is intentionally static and dependency-light so it can be
regenerated for reports without opening a notebook.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mfdsrq_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "report_graphs"
OUT_STEM = "robust_mean_field_dsrq_architecture"
NN_OUT_STEM = "robust_mean_field_dsrq_neural_network_architecture"


COLORS = {
    "env": "#dbeafe",
    "env_edge": "#2563eb",
    "net": "#dcfce7",
    "net_edge": "#16a34a",
    "robust": "#ffedd5",
    "robust_edge": "#ea580c",
    "train": "#f3e8ff",
    "train_edge": "#9333ea",
    "store": "#fee2e2",
    "store_edge": "#dc2626",
    "neutral": "#f8fafc",
    "neutral_edge": "#475569",
    "text": "#0f172a",
}


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 9.5,
    weight: str = "normal",
    radius: float = 0.11,
    wrap: bool = True,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
        linewidth=1.45,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    if wrap:
        wrap_width = max(14, int(width * 13.5))
        text = "\n".join(
            "\n".join(
                textwrap.wrap(
                    line,
                    width=wrap_width,
                    break_long_words=False,
                    replace_whitespace=False,
                )
            )
            for line in text.splitlines()
        )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        fontweight=weight,
        linespacing=1.18,
    )


def add_panel(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    *,
    edge: str,
) -> None:
    x, y = xy
    panel = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1,
        linestyle="--",
        facecolor="#ffffff",
        edgecolor=edge,
        alpha=0.84,
    )
    ax.add_patch(panel)
    ax.text(
        x + 0.25,
        y + height - 0.28,
        title,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=edge,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["neutral_edge"],
    rad: float = 0.0,
    lw: float = 1.45,
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        shrinkA=4,
        shrinkB=4,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if label:
        lx = (start[0] + end[0]) / 2 + label_offset[0]
        ly = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=8.3,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )


def elbow_arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str = COLORS["neutral_edge"],
    lw: float = 1.45,
) -> None:
    if len(points) < 2:
        return
    if len(points) > 2:
        xs, ys = zip(*points[:-1])
        ax.plot(xs, ys, color=color, linewidth=lw)
    patch = FancyArrowPatch(
        points[-2],
        points[-1],
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=4,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(patch)


def build_diagram() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 10.5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    ax.text(
        0.4,
        10.08,
        "Robust Mean-Field DSRQ Architecture",
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="center",
        color=COLORS["text"],
    )
    ax.text(
        0.42,
        9.72,
        "A high-level view of self-play, mean-field approximation, robust action selection, and Q-learning updates",
        fontsize=10.5,
        ha="left",
        va="center",
        color="#334155",
    )

    add_panel(ax, (0.35, 5.6), 5.4, 3.85, "Collection and Mean-Field State", edge=COLORS["env_edge"])
    add_panel(ax, (6.05, 5.6), 5.25, 3.85, "Robust Action Path", edge=COLORS["robust_edge"])
    add_panel(ax, (11.65, 5.6), 5.95, 3.85, "Learning Update", edge=COLORS["train_edge"])
    add_panel(ax, (0.35, 0.55), 17.25, 4.55, "Networks, Replay, Self-Play, and Artifacts", edge=COLORS["neutral_edge"])

    add_box(
        ax,
        (0.75, 8.05),
        2.0,
        0.78,
        "Battle self-play\nmain team vs\nopponent team",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
        weight="bold",
    )
    add_box(
        ax,
        (0.78, 6.82),
        1.95,
        0.74,
        "Each alive agent sees a\nlocal map view and a\nfeature vector",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
        fontsize=8.6,
    )
    add_box(
        ax,
        (3.35, 8.04),
        1.95,
        0.78,
        "Previous action mix\none histogram per team\n21 action bins",
        face=COLORS["neutral"],
        edge=COLORS["neutral_edge"],
        fontsize=8.7,
    )
    add_box(
        ax,
        (3.25, 6.78),
        2.18,
        0.82,
        "Choose which team's\naction mix represents\nthe mean field",
        face=COLORS["neutral"],
        edge=COLORS["neutral_edge"],
        fontsize=8.4,
    )

    add_box(
        ax,
        (6.38, 8.0),
        2.1,
        0.9,
        "Collect a batch\nagent observations\nplus mean-field input",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
        fontsize=8.8,
    )
    add_box(
        ax,
        (8.95, 8.0),
        1.95,
        0.9,
        "Neural Q model predicts\npayoffs for each own\naction vs mean action",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.8,
        weight="bold",
    )
    add_box(
        ax,
        (6.4, 6.55),
        2.2,
        0.98,
        "Robust value step\nmoves limited probability\nmass toward worse actions",
        face=COLORS["robust"],
        edge=COLORS["robust_edge"],
        fontsize=8.4,
        weight="bold",
    )
    add_box(
        ax,
        (9.02, 6.55),
        1.86,
        0.98,
        "Turn robust values\ninto a policy and pick\nthe best action",
        face=COLORS["robust"],
        edge=COLORS["robust_edge"],
        fontsize=8.4,
    )
    add_box(
        ax,
        (7.62, 5.82),
        1.98,
        0.45,
        "Exploration can still choose random actions",
        face="#fff7ed",
        edge=COLORS["robust_edge"],
        fontsize=7.9,
    )

    add_box(
        ax,
        (11.95, 8.02),
        1.88,
        0.82,
        "Step the battle\ncollect rewards, deaths,\nand next observations",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
        fontsize=8.4,
    )
    add_box(
        ax,
        (14.2, 8.02),
        1.65,
        0.82,
        "Current actions become\nthe next team action mix",
        face=COLORS["neutral"],
        edge=COLORS["neutral_edge"],
        fontsize=8.5,
    )
    add_box(
        ax,
        (12.05, 6.62),
        2.05,
        0.9,
        "Store training examples\nfor the main team only",
        face=COLORS["store"],
        edge=COLORS["store_edge"],
        fontsize=8.2,
    )
    add_box(
        ax,
        (14.48, 6.62),
        1.98,
        0.9,
        "Replay buffer\nsamples past experience\nafter warm-up",
        face=COLORS["store"],
        edge=COLORS["store_edge"],
        fontsize=8.2,
        weight="bold",
    )
    add_box(
        ax,
        (13.25, 4.72),
        1.92,
        0.48,
        "sampled minibatch",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
        fontsize=8.2,
    )

    add_box(
        ax,
        (0.75, 3.52),
        2.72,
        1.08,
        "Shared visual encoder\nextracts tactical features\nfrom each local view",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.4,
        weight="bold",
    )
    add_box(
        ax,
        (0.9, 2.25),
        2.42,
        0.84,
        "Side information encoder\nadds agent-level features",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.5,
    )
    add_box(
        ax,
        (3.98, 3.12),
        2.18,
        1.0,
        "Payoff head outputs a\n21 x 21 table:\nown action by mean action",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.5,
    )
    add_box(
        ax,
        (6.82, 3.18),
        2.05,
        0.9,
        "Average over the\nobserved mean-field mix\nto get nominal Q-values",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.4,
    )
    add_box(
        ax,
        (9.52, 3.15),
        2.18,
        0.96,
        "Next-state action choice\nuses the online network\nand robust values",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
        fontsize=8.4,
    )
    add_box(
        ax,
        (12.28, 3.15),
        2.12,
        0.96,
        "Target network estimates\nthe next robust return",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
        fontsize=8.4,
    )
    add_box(
        ax,
        (15.02, 3.15),
        1.9,
        0.96,
        "Train toward\nreward plus discounted\nrobust future value",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
        fontsize=8.1,
        weight="bold",
    )

    add_box(
        ax,
        (5.45, 1.2),
        2.4,
        0.82,
        "Slowly update the\ntarget network",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
        fontsize=8.5,
    )
    add_box(
        ax,
        (9.05, 1.2),
        2.5,
        0.82,
        "If the main team improves,\nsoftly copy it into the\nopponent sparring policy",
        face=COLORS["neutral"],
        edge=COLORS["neutral_edge"],
        fontsize=8.1,
        weight="bold",
    )
    add_box(
        ax,
        (14.2, 1.2),
        2.28,
        0.82,
        "Update the online\nQ model with\ngradient descent",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
        fontsize=8.4,
        weight="bold",
    )
    arrow(ax, (2.75, 8.44), (3.35, 8.44), color=COLORS["env_edge"], label="actions feed histograms", label_offset=(0, 0.34))
    arrow(ax, (1.75, 8.05), (1.75, 7.56), color=COLORS["env_edge"])
    arrow(ax, (2.73, 7.18), (3.25, 7.18), color=COLORS["neutral_edge"])
    arrow(ax, (4.3, 8.04), (4.3, 7.6), color=COLORS["neutral_edge"])
    arrow(ax, (5.43, 7.18), (6.38, 8.44), color=COLORS["env_edge"], rad=0.16, label="tile per alive agent", label_offset=(0.18, 0.24))
    arrow(ax, (8.48, 8.45), (8.95, 8.45), color=COLORS["net_edge"])
    arrow(ax, (9.9, 8.0), (7.5, 7.53), color=COLORS["robust_edge"], rad=0.08)
    arrow(ax, (8.6, 7.04), (9.02, 7.04), color=COLORS["robust_edge"])
    arrow(ax, (10.88, 7.04), (11.95, 8.43), color=COLORS["env_edge"], rad=0.08, label="selected actions", label_offset=(0.04, 0.2))
    arrow(ax, (13.83, 8.43), (14.2, 8.43), color=COLORS["neutral_edge"])
    arrow(
        ax,
        (15.02, 8.02),
        (14.76, 7.52),
        color=COLORS["neutral_edge"],
        label="used next step",
        label_offset=(0.7, -0.05),
    )
    arrow(ax, (12.98, 8.02), (13.05, 7.52), color=COLORS["store_edge"])
    arrow(ax, (14.1, 7.07), (14.48, 7.07), color=COLORS["store_edge"])
    arrow(ax, (15.47, 6.62), (14.21, 5.2), color=COLORS["train_edge"], rad=0.08)
    arrow(ax, (13.25, 4.96), (8.0, 4.08), color=COLORS["train_edge"])
    arrow(ax, (14.21, 4.72), (10.6, 4.11), color=COLORS["train_edge"])

    arrow(ax, (3.47, 4.05), (3.98, 3.62), color=COLORS["net_edge"])
    arrow(ax, (3.32, 2.67), (3.98, 3.3), color=COLORS["net_edge"])
    arrow(ax, (6.16, 3.61), (6.82, 3.61), color=COLORS["net_edge"])
    arrow(ax, (8.87, 3.62), (9.52, 3.62), color=COLORS["train_edge"])
    arrow(ax, (11.7, 3.62), (12.28, 3.62), color=COLORS["train_edge"])
    arrow(ax, (14.4, 3.62), (15.02, 3.62), color=COLORS["train_edge"])
    arrow(ax, (15.95, 3.15), (15.34, 2.02), color=COLORS["train_edge"])
    elbow_arrow(
        ax,
        [(14.2, 1.42), (13.35, 0.82), (7.0, 0.82), (7.0, 1.2)],
        color=COLORS["train_edge"],
    )
    arrow(ax, (6.65, 3.18), (6.65, 2.02), color=COLORS["train_edge"])
    arrow(ax, (7.85, 1.61), (9.05, 1.61), color=COLORS["neutral_edge"])

    # ax.text(
    #     0.78,
    #     0.76,
    #     "Key semantics: robustness is applied to the mean-action histogram, not to the full joint action distribution. "
    #     "The current trainer learns from main-agent replay; the opponent acts as a softly copied sparring policy.",
    #     fontsize=9.2,
    #     color="#334155",
    #     ha="left",
    #     va="center",
    # )

    return fig


def build_neural_network_diagram() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 5.6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(5.25, 10.5)
    ax.axis("off")

    ax.text(
        0.4,
        10.08,
        "Robust Mean-Field DSRQ Neural Network Architecture",
        fontsize=19,
        fontweight="bold",
        ha="left",
        va="center",
        color=COLORS["text"],
    )
    ax.text(
        0.42,
        9.72,
        "The online Q network and target Q network use the same pairwise payoff model; only the online network is optimized directly.",
        fontsize=10.5,
        ha="left",
        va="center",
        color="#334155",
    )

    add_panel(ax, (0.35, 5.68), 17.25, 3.72, "Shared Pairwise Q Network Layer Stack", edge=COLORS["net_edge"])

    add_box(
        ax,
        (0.78, 8.02),
        2.1,
        0.82,
        "Local map view\ninput tensor\n[B, 7, 13, 13]",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
        fontsize=8.5,
        weight="bold",
    )
    add_box(
        ax,
        (0.82, 6.45),
        2.02,
        0.76,
        "Side feature input\nagent feature vector\n[B, 34]",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
        fontsize=8.5,
        weight="bold",
    )
    add_box(
        ax,
        (3.42, 8.0),
        2.2,
        0.88,
        "Visual conv block 1\nConv2d 7 -> 32\n3x3, padding 1\nReLU",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.2,
    )
    add_box(
        ax,
        (6.08, 8.0),
        2.2,
        0.88,
        "Visual conv block 2\nConv2d 32 -> 32\n3x3, padding 1\nReLU",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.2,
    )
    add_box(
        ax,
        (8.75, 8.0),
        2.05,
        0.88,
        "Flatten visual map\n[B, 32, 13, 13]\n-> [B, 5408]",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.3,
    )
    add_box(
        ax,
        (11.25, 8.0),
        2.05,
        0.88,
        "Visual embedding\nLinear 5408 -> 256\nReLU\noutput [B, 256]",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.1,
    )
    add_box(
        ax,
        (3.65, 6.36),
        2.18,
        0.88,
        "Feature embedding\nLinear 34 -> 32\nReLU\noutput [B, 32]",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.1,
    )
    add_box(
        ax,
        (13.75, 7.18),
        1.92,
        0.86,
        "Concatenate\nvisual + feature\n[B, 288]",
        face=COLORS["neutral"],
        edge=COLORS["neutral_edge"],
        fontsize=8.4,
        weight="bold",
    )

    head_y = 6.12
    ax.text(
        7.08,
        7.12,
        "Payoff head MLP",
        ha="left",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["net_edge"],
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.95,
        },
    )
    add_box(
        ax,
        (7.05, head_y),
        1.78,
        0.68,
        "Linear\n288 -> 128\nReLU",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.2,
    )
    add_box(
        ax,
        (9.22, head_y),
        1.76,
        0.68,
        "Linear\n128 -> 64\nReLU",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.2,
    )
    add_box(
        ax,
        (11.34, head_y),
        1.72,
        0.68,
        "Linear\n64 -> 441",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.2,
    )
    add_box(
        ax,
        (13.38, head_y),
        2.0,
        0.68,
        "Reshape output\n[B, 441] -> [B, 21, 21]",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.1,
        weight="bold",
    )
    add_box(
        ax,
        (15.85, head_y),
        1.35,
        0.68,
        "Payoff table\nM(o)[a,b]",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.1,
        weight="bold",
    )

    arrow(ax, (2.88, 8.43), (3.42, 8.43), color=COLORS["net_edge"])
    arrow(ax, (5.62, 8.43), (6.08, 8.43), color=COLORS["net_edge"])
    arrow(ax, (8.28, 8.43), (8.75, 8.43), color=COLORS["net_edge"])
    arrow(ax, (10.8, 8.43), (11.25, 8.43), color=COLORS["net_edge"])
    arrow(ax, (2.84, 6.83), (3.65, 6.83), color=COLORS["net_edge"])
    arrow(ax, (13.3, 8.43), (13.75, 7.62), color=COLORS["neutral_edge"], rad=-0.12)
    elbow_arrow(
        ax,
        [(5.83, 6.96), (8.3, 7.42), (13.75, 7.44)],
        color=COLORS["neutral_edge"],
    )
    elbow_arrow(
        ax,
        [(14.7, 7.18), (14.7, 6.94), (7.95, 6.94), (7.95, 6.8)],
        color=COLORS["net_edge"],
    )
    arrow(ax, (8.83, head_y + 0.34), (9.22, head_y + 0.34), color=COLORS["net_edge"])
    arrow(ax, (10.98, head_y + 0.34), (11.34, head_y + 0.34), color=COLORS["net_edge"])
    arrow(ax, (13.06, head_y + 0.34), (13.38, head_y + 0.34), color=COLORS["net_edge"])
    arrow(ax, (15.38, head_y + 0.34), (15.85, head_y + 0.34), color=COLORS["net_edge"])

    ax.text(
        0.52,
        5.38,
        "Notation: B is the minibatch size. The online Q network and target Q network both instantiate this same architecture.",
        ha="left",
        va="center",
        fontsize=9.2,
        color="#334155",
    )

    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_diagram()
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_path}")

    nn_fig = build_neural_network_diagram()
    nn_png_path = OUT_DIR / f"{NN_OUT_STEM}.png"
    nn_fig.savefig(nn_png_path, bbox_inches="tight", facecolor="white")
    plt.close(nn_fig)
    print(f"Wrote {nn_png_path}")


if __name__ == "__main__":
    main()
