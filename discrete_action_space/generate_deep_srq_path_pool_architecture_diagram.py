"""Generate an environment-agnostic Deep SRQ + PATH pool architecture diagram.

The diagram is static and dependency-light so it can be regenerated for reports
without importing the training stack or opening a notebook.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deep_srq_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "report_graphs"
DEFAULT_STEM = "deep_srq_path_pool_architecture"


COLORS = {
    "env": "#dbeafe",
    "env_edge": "#2563eb",
    "net": "#dcfce7",
    "net_edge": "#16a34a",
    "solver": "#ffedd5",
    "solver_edge": "#ea580c",
    "train": "#f3e8ff",
    "train_edge": "#9333ea",
    "replay": "#fee2e2",
    "replay_edge": "#dc2626",
    "neutral": "#f8fafc",
    "neutral_edge": "#475569",
    "text": "#0f172a",
}


def _wrap_text(text: str, width: int) -> str:
    wrapped_lines = []
    for line in text.splitlines():
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.append(
            "\n".join(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=False,
                    replace_whitespace=False,
                )
            )
        )
    return "\n".join(wrapped_lines)


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 9.0,
    weight: str = "normal",
    radius: float = 0.08,
    wrap_width: int | None = None,
    alpha: float = 1.0,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
        linewidth=1.35,
        facecolor=face,
        edgecolor=edge,
        alpha=alpha,
    )
    ax.add_patch(patch)
    if wrap_width is None:
        wrap_width = max(14, int(width * 13.0))
    ax.text(
        x + width / 2,
        y + height / 2,
        _wrap_text(text, wrap_width),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        fontweight=weight,
        linespacing=1.16,
    )


def add_panel(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    *,
    edge: str,
    face: str = "#ffffff",
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.1,
        linestyle="--",
        facecolor=face,
        edgecolor=edge,
        alpha=0.86,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.22,
        y + height - 0.26,
        title,
        ha="left",
        va="top",
        fontsize=10.5,
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
    lw: float = 1.35,
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
    label_size: float = 7.8,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        shrinkA=5,
        shrinkB=5,
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
            fontsize=label_size,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.16",
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
    lw: float = 1.35,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
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
        shrinkB=5,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(patch)
    if label and label_xy:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=7.8,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )


def build_diagram() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(19, 11), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 11)
    ax.axis("off")

    ax.text(
        0.45,
        10.58,
        "Deep SRQ with PATH Solver Process Pool",
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="center",
        color=COLORS["text"],
    )
    ax.text(
        0.47,
        10.22,
        "Environment-agnostic discrete-action architecture: Q_theta(s) emits a normal-form game, PATH solves SRE policies, Dueling Double DQN learns from replay.",
        fontsize=10.2,
        ha="left",
        va="center",
        color="#334155",
    )

    add_panel(ax, (0.35, 5.75), 4.0, 4.15, "State and Rollout", edge=COLORS["env_edge"])
    add_panel(ax, (4.7, 5.75), 5.05, 4.15, "Neural Q Tensor", edge=COLORS["net_edge"])
    add_panel(ax, (10.1, 5.75), 4.55, 4.15, "SRE Stage Solver", edge=COLORS["solver_edge"])
    add_panel(ax, (15.0, 5.75), 3.65, 4.15, "Joint Action", edge=COLORS["env_edge"])
    add_panel(ax, (0.35, 0.55), 18.3, 4.65, "Replay and Double-DQN Update", edge=COLORS["train_edge"])

    add_box(
        ax,
        (0.72, 8.58),
        3.1,
        0.72,
        "Any discrete-action multi-agent environment",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
        weight="bold",
        wrap_width=28,
    )
    add_box(
        ax,
        (0.72, 7.48),
        3.1,
        0.82,
        "Environment encoder\nflat global state s in R^D\nD = config.obs_dim",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
        wrap_width=28,
    )
    add_box(
        ax,
        (0.72, 6.25),
        3.1,
        0.96,
        "Agent/action schema\nN = config.num_agents\nA_i actions per agent\nhomogeneous default A_i = A",
        face=COLORS["neutral"],
        edge=COLORS["neutral_edge"],
        wrap_width=28,
    )

    add_box(
        ax,
        (5.02, 8.5),
        4.35,
        0.78,
        "DuelingJointQNetwork (default network_type='joint_output')",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        weight="bold",
        wrap_width=42,
    )
    add_box(
        ax,
        (5.02, 7.55),
        4.35,
        0.68,
        "Feature MLP: Linear(D,H1) -> ReLU -> Linear(H1,H2) -> ReLU\nq_hidden_dims default (128, 128)",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.35,
        wrap_width=46,
    )
    add_box(
        ax,
        (5.02, 6.55),
        2.05,
        0.7,
        "Value head\nLinear(H2, N)\nV(s): [B,N]",
        face="#ecfccb",
        edge=COLORS["net_edge"],
        fontsize=8.2,
        wrap_width=22,
    )
    add_box(
        ax,
        (7.32, 6.55),
        2.05,
        0.7,
        "Advantage head\nLinear(H2, prod(A_i)*N)\nAdv: [B, |A_joint|, N]",
        face="#ecfccb",
        edge=COLORS["net_edge"],
        fontsize=8.0,
        wrap_width=24,
    )
    add_box(
        ax,
        (5.02, 5.9),
        4.35,
        0.55,
        "Output Q_tensor: [B, A1, ..., AN, N]\nQ = V + (Adv - mean over joint actions)",
        face=COLORS["net"],
        edge=COLORS["net_edge"],
        fontsize=8.05,
        wrap_width=44,
    )

    add_box(
        ax,
        (10.42, 8.38),
        3.85,
        0.86,
        "Policy cache and duplicate-key coalescing\nrounded Q tensor keys + optional warm starts",
        face=COLORS["neutral"],
        edge=COLORS["neutral_edge"],
        fontsize=8.45,
        wrap_width=42,
    )
    add_box(
        ax,
        (10.42, 7.18),
        3.85,
        0.92,
        "Process-pool solver\nsolve_batch([Q_1,...,Q_B], epsilon_robust)\nmax_workers = config.sre_solver_workers",
        face=COLORS["solver"],
        edge=COLORS["solver_edge"],
        weight="bold",
        wrap_width=42,
    )
    add_box(
        ax,
        (10.42, 6.1),
        1.8,
        0.82,
        "N = 2\npath_c_pool\nPATH LCP\nsolve_lcp(...)",
        face="#fff7ed",
        edge=COLORS["solver_edge"],
        fontsize=8.0,
        wrap_width=19,
    )
    add_box(
        ax,
        (12.47, 6.1),
        1.8,
        0.82,
        "N > 2\npath_mcp_nplayer_pool\nPATH MCP\nsolve_mcp(...)",
        face="#fff7ed",
        edge=COLORS["solver_edge"],
        fontsize=7.75,
        wrap_width=20,
    )

    add_box(
        ax,
        (15.36, 8.35),
        2.92,
        0.82,
        "SRE policy profile\npi_i in Delta(A_i)\nfor each agent i",
        face=COLORS["solver"],
        edge=COLORS["solver_edge"],
        weight="bold",
        wrap_width=28,
    )
    add_box(
        ax,
        (15.36, 7.18),
        2.92,
        0.82,
        "Action selection\nsample from pi_i or epsilon-greedy exploration",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
        wrap_width=31,
    )
    add_box(
        ax,
        (15.36, 6.12),
        2.92,
        0.66,
        "joint action a = (a_1,...,a_N)",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
        weight="bold",
        wrap_width=29,
    )

    add_box(
        ax,
        (1.0, 3.98),
        3.5,
        0.78,
        "Replay buffer D\n(s, a, r, s_next, done)",
        face=COLORS["replay"],
        edge=COLORS["replay_edge"],
        weight="bold",
        wrap_width=32,
    )
    add_box(
        ax,
        (5.0, 3.98),
        3.4,
        0.78,
        "Minibatch\nQ_theta(s_batch)[a_1,...,a_N,:]\ncurrent_q: [B,N]",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
        fontsize=8.15,
        wrap_width=34,
    )
    add_box(
        ax,
        (8.95, 3.98),
        3.75,
        0.78,
        "Online next-state game\nQ_theta(s_next) selects pi_next through the same SRE solver path",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
        fontsize=8.05,
        wrap_width=42,
    )
    add_box(
        ax,
        (13.25, 3.98),
        4.05,
        0.78,
        "Target network evaluates\nv_next = E_{a'~prod(pi_next)} Q_target(s_next,a')\nshape [B,N]",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
        fontsize=7.9,
        wrap_width=43,
    )
    add_box(
        ax,
        (3.1, 2.55),
        4.55,
        0.84,
        "Bellman target\ny = r + gamma * (1 - done) * v_next",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
        weight="bold",
        wrap_width=43,
    )
    add_box(
        ax,
        (8.2, 2.55),
        4.25,
        0.84,
        "Optimize online network\nMSE(current_q, y)\nAdam lr default 3e-4 + grad clip 10",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
        wrap_width=41,
    )
    add_box(
        ax,
        (13.0, 2.55),
        3.8,
        0.84,
        "Target update\nhard sync every target_update_steps\nor Polyak target_tau",
        face=COLORS["train"],
        edge=COLORS["train_edge"],
        wrap_width=38,
    )
    add_box(
        ax,
        (5.0, 1.1),
        8.6,
        0.78,
        "Same output contract for alternate network_type options: per_agent_independent and shared_trunk_separate_heads both return [B, A1, ..., AN, N].",
        face=COLORS["neutral"],
        edge=COLORS["neutral_edge"],
        fontsize=8.25,
        wrap_width=86,
    )

    arrow(ax, (3.82, 7.89), (5.02, 8.73), color=COLORS["env_edge"], label="s in R^D", label_offset=(0.05, 0.18))
    arrow(ax, (7.07, 6.9), (7.32, 6.9), color=COLORS["net_edge"], lw=1.1)
    arrow(ax, (7.2, 6.55), (7.2, 6.45), color=COLORS["net_edge"], lw=1.1)
    arrow(
        ax,
        (9.37, 6.17),
        (10.42, 7.63),
        color=COLORS["net_edge"],
        label="Q tensor batch",
        label_offset=(0.03, -0.18),
        label_size=7.4,
    )
    arrow(ax, (12.35, 7.18), (12.35, 6.92), color=COLORS["solver_edge"], lw=1.1)
    arrow(ax, (14.27, 7.78), (15.36, 8.76), color=COLORS["solver_edge"], label="policies", label_offset=(0.02, 0.14))
    arrow(ax, (16.82, 8.35), (16.82, 8.0), color=COLORS["env_edge"], lw=1.1)
    arrow(ax, (16.82, 7.18), (16.82, 6.78), color=COLORS["env_edge"], lw=1.1)
    elbow_arrow(
        ax,
        [(16.82, 6.12), (16.82, 5.45), (2.75, 5.45), (2.75, 4.76)],
        color=COLORS["env_edge"],
        label="transition",
        label_xy=(10.6, 5.48),
    )

    arrow(ax, (4.5, 4.37), (5.0, 4.37), color=COLORS["train_edge"], lw=1.1)
    arrow(ax, (8.4, 4.37), (8.95, 4.37), color=COLORS["train_edge"], lw=1.1)
    arrow(ax, (12.7, 4.37), (13.25, 4.37), color=COLORS["train_edge"], lw=1.1)
    elbow_arrow(
        ax,
        [(15.3, 3.98), (15.3, 3.53), (5.38, 3.53), (5.38, 3.39)],
        color=COLORS["train_edge"],
        label="target values",
        label_xy=(10.25, 3.56),
    )
    arrow(ax, (7.65, 2.97), (8.2, 2.97), color=COLORS["train_edge"], lw=1.1)
    arrow(ax, (12.45, 2.97), (13.0, 2.97), color=COLORS["train_edge"], lw=1.1)
    elbow_arrow(
        ax,
        [(10.32, 3.98), (10.32, 5.33), (12.35, 5.33), (12.35, 6.23)],
        color=COLORS["solver_edge"],
        label="pi_next solve",
        label_xy=(11.28, 5.36),
    )
    elbow_arrow(
        ax,
        [(10.32, 2.55), (10.32, 2.08), (7.2, 2.08), (7.2, 5.97)],
        color=COLORS["net_edge"],
        label="gradients update Q_theta",
        label_xy=(8.8, 2.11),
    )
    elbow_arrow(
        ax,
        [(14.9, 2.55), (14.9, 1.96), (7.2, 1.96), (7.2, 5.97)],
        color=COLORS["train_edge"],
        label="sync Q_target",
        label_xy=(13.0, 1.99),
    )

    ax.text(
        0.55,
        0.18,
        "Notation: D = flattened state dimension, N = agents, A_i = agent i action count, |A_joint| = product_i A_i. Class anchors: DuelingDoubleDqnSreAgent, DuelingJointQNetwork, ProcessPoolPathCBimatrixSreSolver, ProcessPoolPathMcpNPlayerSreSolver.",
        fontsize=8.2,
        ha="left",
        va="center",
        color="#475569",
    )

    return fig


def save_diagram(output_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_diagram()
    saved = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt.lower()}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        saved.append(path)
    plt.close(fig)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Deep SRQ + PATH process-pool architecture diagram."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for generated figures. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help=f"Output filename stem. Default: {DEFAULT_STEM}",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="One or more output formats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = save_diagram(args.output_dir, args.stem, args.formats)
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
