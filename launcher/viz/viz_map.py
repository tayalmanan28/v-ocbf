"""
Visualize the learned CBF (Vc) for BoatRobot.

Plots the learned Vc contour map over the 2-D (x, y) state space with
obstacle circles and goal overlaid.  Since BoatRobot has a pure 2-D state
(unlike PointRobot which requires fixing v and theta), a single panel
fully captures the barrier function.

Usage:
    python launcher/viz/viz_map.py \
        --model_dir ./results/BoatRobot/vocbf_hj_BoatRobot_...
"""
import os
import sys
sys.path.append(".")

import re
import json
import numpy as np
import jax
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle
from absl import app, flags
from ml_collections import ConfigDict

from env.boat_robot import BoatRobot
from jaxrl5.agents import VOCBF

FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "", "Path to the results directory")

# ─── Styling ──────────────────────────────────────────────────────────────────
label_size = 18
legend_size = 30
ticks_size = 18
width = 0.5

font = {
    "family": "Times New Roman",
    "weight": "normal",
    "size": label_size,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d


def get_latest_pickle(directory):
    """Return the path to the .pickle checkpoint with the highest number."""
    pickle_files = [f for f in os.listdir(directory) if f.endswith(".pickle")]
    if not pickle_files:
        raise FileNotFoundError(f"No .pickle files in {directory}")
    numbers = {}
    for f in pickle_files:
        match = re.search(r"\d+", f)
        if match:
            numbers[int(match.group())] = os.path.join(directory, f)
    return numbers[max(numbers.keys())]


# ─── Core plotting ────────────────────────────────────────────────────────────
def plot_cbf(ax, agent, cb=False):
    """
    Evaluate the learned Vc (safe_value) on a dense (x, y) grid and plot
    the contour map.  The BoatRobot observation is simply [x, y].
    """
    n = 201
    x1 = np.linspace(-3.0, 3.0, n)
    x2 = np.linspace(-3.0, 3.0, n)
    x1_grid, x2_grid = np.meshgrid(x1, x2)

    # BoatRobot obs = [x, y]  (2-D)
    batch_obses = np.stack(
        [x1_grid.ravel(), x2_grid.ravel()], axis=-1
    ).astype(np.float32)  # (n*n, 2)

    # Evaluate learned Vc
    safe_value = agent.safe_value.apply_fn(
        {"params": agent.safe_value.params}, jax.device_put(batch_obses)
    )
    value_square = np.asarray(safe_value).reshape(x1_grid.shape)

    # Contour fill
    vmin = float(value_square.min())
    vmax = float(value_square.max())
    # Symmetrise around 0 so the zero-level set is clearly visible
    abs_max = max(abs(vmin), abs(vmax))
    norm = colors.Normalize(vmin=-abs_max, vmax=abs_max)

    ct = ax.contourf(
        x1_grid, x2_grid, value_square,
        norm=norm,
        levels=30,
        cmap="rainbow",
    )

    # Zero-level contour (Vc = 0  ↔  safe / unsafe boundary)
    ct_line = ax.contour(
        x1_grid, x2_grid, value_square,
        levels=[0],
        colors="#32ABD6",
        linewidths=2.0,
        linestyles="solid",
    )
    ax.clabel(ct_line, inline=True, fontsize=15, fmt=r"0")

    if cb:
        ticks = np.linspace(np.floor(vmin), np.ceil(vmax), 6)
        cbar = plt.colorbar(ct, ax=ax, shrink=0.8, pad=0.02, ticks=ticks)
        cbar.ax.tick_params(labelsize=ticks_size)
        cbarlabels = cbar.ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in cbarlabels]

    return ax


def draw_env_features(ax, env):
    """Draw obstacle circles and goal on the axis."""
    for hp in env.hazard_position_list:
        radius = hp[2] if len(hp) > 2 else 0.4
        circle = Circle(
            (hp[0], hp[1]),
            radius,
            fill=False,
            edgecolor="k",
            linewidth=1.5,
        )
        ax.add_patch(circle)

    # Goal
    gp = env.goal_position
    goal_r = max(getattr(env, "goal_size", 0.1), 0.08)
    circle = Circle(
        (gp[0], gp[1]),
        goal_r,
        fill=True,
        alpha=0.5,
        color=(0.35, 0.66, 0.35),
    )
    ax.add_patch(circle)


# ─── Full figure ──────────────────────────────────────────────────────────────
def plot_pic(env, agent, model_dir):
    """
    Two-panel figure:
      Left  – task map (obstacles + goal)
      Right – learned Vc contour with obstacles overlaid
    """
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(9, 4),
        constrained_layout=True,
    )

    my_x_ticks = np.arange(-3, 3.01, 1.5)
    my_y_ticks = np.arange(-3, 3.01, 1.5)

    # ── Panel 1 : task map ───────────────────────────────────────────────
    ax1.set_title("Task", fontsize=label_size, fontname="Times New Roman")

    # Draw obstacles filled
    for hp in env.hazard_position_list:
        radius = hp[2] if len(hp) > 2 else 0.4
        circle = Circle(
            (hp[0], hp[1]),
            radius,
            fill=True,
            alpha=0.5,
            color=(0.30, 0.52, 0.74),
        )
        ax1.add_patch(circle)

    # Goal
    gp = env.goal_position
    goal_r = max(getattr(env, "goal_size", 0.1), 0.08)
    circle = Circle(
        (gp[0], gp[1]),
        goal_r,
        fill=True,
        alpha=0.5,
        color=(0.35, 0.66, 0.35),
    )
    ax1.add_patch(circle)

    ax1.set_xticks(my_x_ticks)
    ax1.set_yticks(my_y_ticks)
    ax1.set_xlim((-3, 3))
    ax1.set_ylim((-3, 3))
    ax1.set_aspect("equal")
    ax1.tick_params(labelsize=ticks_size)
    ax1.tick_params(
        axis="both", which="both",
        bottom=False, left=False, labelbottom=False, labelleft=False,
    )
    for spine in ax1.spines.values():
        spine.set_color("gray")

    # ── Panel 2 : learned Vc (CBF) ──────────────────────────────────────
    ax2.set_title(
        "Learned $V_c$ (CBF)", fontsize=label_size, fontname="Times New Roman"
    )
    ax2 = plot_cbf(ax2, agent, cb=True)
    draw_env_features(ax2, env)

    ax2.set_xticks(my_x_ticks)
    ax2.set_yticks(my_y_ticks)
    ax2.set_xlim((-3, 3))
    ax2.set_ylim((-3, 3))
    ax2.set_aspect("equal")
    ax2.tick_params(labelsize=ticks_size)
    ax2.tick_params(
        axis="both", which="both",
        bottom=False, left=False, labelbottom=False, labelleft=False,
    )
    for spine in ax2.spines.values():
        spine.set_linewidth(width)
        spine.set_color("white")

    # ── Save ─────────────────────────────────────────────────────────────
    imgs_dir = os.path.join(model_dir, "imgs")
    os.makedirs(imgs_dir, exist_ok=True)
    save_path = os.path.join(imgs_dir, "viz_map.png")
    plt.savefig(save_path, dpi=600)
    print(f"Saved to {save_path}")


# ─── Load model ───────────────────────────────────────────────────────────────
def load_model(model_dir):
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        cfg = to_config_dict(json.load(f))

    env = BoatRobot(id=0, seed=0)

    config_dict = dict(cfg["agent_kwargs"])
    config_dict.pop("model_cls", None)
    config_dict.pop("cost_scale", None)
    # Remove BC / dynamics / CBF params that VOCBF.create doesn't accept
    for k in [
        "bc_hidden_dim", "bc_num_layers", "bc_lr", "bc_epochs",
        "dyn_hidden_dim", "dyn_num_layers", "dyn_lr", "dyn_epochs",
        "cbf_alpha",
    ]:
        config_dict.pop(k, None)

    config_dict["env_max_steps"] = env._max_episode_steps

    agent = VOCBF.create(
        cfg["seed"], env.observation_space, env.action_space, **config_dict
    )
    model_file = get_latest_pickle(model_dir)
    agent = agent.load(model_file)
    print(f"Loaded VOCBF model from {model_file}")

    os.makedirs(os.path.join(model_dir, "imgs"), exist_ok=True)

    return env, agent


# ─── Entry ────────────────────────────────────────────────────────────────────
def main(_):
    env, agent = load_model(FLAGS.model_dir)
    plot_pic(env, agent, FLAGS.model_dir)


if __name__ == "__main__":
    app.run(main)
