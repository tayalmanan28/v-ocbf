"""
Trajectory visualization for CBF-QP evaluation.

Plots agent trajectories on a 2D map overlaid with:
  - Vc (CBF) contour heatmap
  - Obstacle zones
  - Goal position
  - Trajectory coloured by Vc value (safe = green, unsafe = red)
  - Markers for start / end positions
  - Arrows where the CBF filter intervened

Usage:
    python launcher/viz/viz_trajectory.py \
        --model_dir ./results/BoatRobot/vocbf_hj_BoatRobot_... \
        --env_id 30 \
        --num_episodes 5 \
        --save_path trajectory_plot.png
"""
import os
import sys
sys.path.append(".")

import re
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib import cm
from absl import app, flags

from ml_collections import ConfigDict
from env.env_list import env_list
from env.point_robot import PointRobot
from env.boat_robot import BoatRobot
from jaxrl5.agents import VOCBF
from jaxrl5.agents.vocbf.vocbf import (
    BCPolicy,
    AffineDynamics,
    build_vc_network,
    cbf_safe_control,
    torchify,
    DEFAULT_DEVICE,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "", "Path to the results directory containing model files")
flags.DEFINE_integer("env_id", 30, "Environment index from env_list (30 = BoatRobot)")
flags.DEFINE_integer("num_episodes", 500, "Number of evaluation episodes to plot")
flags.DEFINE_float("cbf_alpha", 1.0, "CBF alpha parameter")
flags.DEFINE_string("save_path", "trajectory_plot1.png", "Where to save the figure")
flags.DEFINE_bool("show", True, "Show the plot interactively")
flags.DEFINE_integer("contour_resolution", 150, "Grid resolution for Vc contour map")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d


def get_latest_model_file(model_dir):
    files = os.listdir(model_dir)
    pickle_files = [f for f in files if f.endswith(".pickle")]
    if not pickle_files:
        raise FileNotFoundError(f"No .pickle files found in {model_dir}")
    numbers = {}
    for f in pickle_files:
        match = re.search(r"\d+", f)
        if match:
            numbers[int(match.group())] = os.path.join(model_dir, f)
    return numbers[max(numbers.keys())]


# ---------------------------------------------------------------------------
# Run a single episode and record full trajectory data
# ---------------------------------------------------------------------------

def run_episode_with_recording(env, episode_no, policy, V_net, dynamics_model,
                                max_episode_steps, deterministic=True):
    """
    Returns a dict with:
        positions  : (T, state_dim) array of states visited
        actions    : (T, action_dim) array of safe actions taken
        nominal    : (T, action_dim) array of raw BC actions
        vc_values  : (T,) array of Vc at each state
        rewards    : (T,) array of per-step rewards
        intervened : (T,) bool array – True where CBF modified the action
    """
    reset_ret = env.reset(seed=episode_no + 500)
    obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
    
    with torch.no_grad():
        vc = V_net(torchify(obs).unsqueeze(0)).item()
    i = 1
    while vc < 0:
        reset_ret = env.reset(seed=episode_no + 500 + i)
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        with torch.no_grad():
            vc = V_net(torchify(obs).unsqueeze(0)).item()
        i += 1
    print(f"Episode {episode_no}: Initial state Vc={vc:.3f} after {i} resets")

    positions, actions_safe, actions_nominal = [], [], []
    vc_values, rewards_list, intervened = [], [], []

    for _ in range(max_episode_steps):
        # Record state and Vc
        with torch.no_grad():
            vc = V_net(torchify(obs).unsqueeze(0)).item()
        vc_values.append(vc)
        positions.append(obs.copy())

        # Nominal (BC) action
        with torch.no_grad():
            nominal = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
        actions_nominal.append(nominal.copy())

        # CBF-filtered action
        safe_action = cbf_safe_control(V_net, dynamics_model, obs, nominal)
        actions_safe.append(safe_action.copy())

        # Was the action modified?
        intervened.append(not np.allclose(nominal, safe_action, atol=1e-6))

        # Step
        step_ret = env.step(safe_action)
        if len(step_ret) == 4:
            next_obs, reward, done, info = step_ret
        elif len(step_ret) == 5:
            next_obs, reward, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        elif len(step_ret) == 6:
            next_obs, reward, _, terminated, truncated, info = step_ret
            done = bool(terminated or truncated)
        else:
            raise RuntimeError(f"Unexpected env.step() return length: {len(step_ret)}")

        rewards_list.append(float(reward))

        if done:
            # record the terminal state's Vc too
            with torch.no_grad():
                vc_end = V_net(torchify(next_obs).unsqueeze(0)).item()
            vc_values.append(vc_end)
            positions.append(next_obs.copy())
            break
        obs = next_obs

    return {
        "positions": np.array(positions),
        "actions_safe": np.array(actions_safe),
        "actions_nominal": np.array(actions_nominal),
        "vc_values": np.array(vc_values),
        "rewards": np.array(rewards_list),
        "intervened": np.array(intervened),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def compute_vc_contour(V_net, xlim, ylim, resolution=150):
    """Evaluate Vc on a 2-D grid (works for 2-D state spaces like BoatRobot)."""
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    with torch.no_grad():
        vals = V_net(torch.from_numpy(grid).to(DEFAULT_DEVICE)).cpu().numpy().reshape(xx.shape)
    return xx, yy, vals


def plot_trajectories(episodes, env, V_net, save_path, show=True, contour_res=150):
    """Create the full trajectory visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Determine plot bounds from environment observation space
    low = env.observation_space.low[:2]
    high = env.observation_space.high[:2]
    pad = 0.3
    xlim = (low[0] - pad, high[0] + pad)
    ylim = (low[1] - pad, high[1] + pad)

    # ---- Left panel: Vc contour + trajectories coloured by Vc ----
    ax = axes[0]
    ax.set_title("Trajectories coloured by $V_c$ (CBF value)", fontsize=13)

    # Vc contour heatmap (only for 2-D envs)
    state_dim = env.observation_space.shape[0]
    if state_dim == 2:
        xx, yy, vc_grid = compute_vc_contour(V_net, xlim, ylim, contour_res)
        # Safe region: Vc >= 0, unsafe: Vc < 0
        # Use a diverging colourmap centred at 0
        vabs = max(abs(vc_grid.min()), abs(vc_grid.max()))
        norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        cf = ax.contourf(xx, yy, vc_grid, levels=80, cmap="RdYlGn", norm=norm, alpha=0.45)
        ax.contour(xx, yy, vc_grid, levels=[0.0], colors="k", linewidths=2.0, linestyles="--")
        cbar = fig.colorbar(cf, ax=ax, shrink=0.8)
        cbar.set_label("$V_c$ (CBF)", fontsize=11)

    # Obstacles
    _draw_env_features(ax, env)

    # Trajectories
    for i, ep in enumerate(episodes):
        pos = ep["positions"][:, :2]
        vc = ep["vc_values"]
        # Normalise Vc for colour mapping
        all_vc = np.concatenate([e["vc_values"] for e in episodes])
        vmin, vmax = all_vc.min(), all_vc.max()
        if vmax - vmin < 1e-8:
            vmax = vmin + 1.0
        norm_vc = (vc - vmin) / (vmax - vmin)

        # Build coloured line segments
        points = pos.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="coolwarm", linewidth=1.8, alpha=0.85)
        lc.set_array(norm_vc[:-1])
        ax.add_collection(lc)

        # Start / end markers
        ax.plot(pos[0, 0], pos[0, 1], "o", color="blue", markersize=7, zorder=5)
        ax.plot(pos[-1, 0], pos[-1, 1], "s", color="purple", markersize=7, zorder=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # Legend
    ax.plot([], [], "o", color="blue", label="Start")
    ax.plot([], [], "s", color="purple", label="End")
    ax.plot([], [], "--", color="k", label="$V_c = 0$ boundary")
    ax.legend(loc="upper left", fontsize=9)

    # ---- Right panel: CBF interventions ----
    ax2 = axes[1]
    ax2.set_title("CBF interventions (red = filter active)", fontsize=13)
    _draw_env_features(ax2, env)

    for i, ep in enumerate(episodes):
        pos = ep["positions"][:, :2]
        interv = ep["intervened"]

        # Plot full trajectory in light grey
        ax2.plot(pos[:, 0], pos[:, 1], "-", color="grey", linewidth=1.0, alpha=0.5)

        # Overlay intervened segments in red
        for t in range(len(interv)):
            if interv[t]:
                ax2.plot(pos[t:t + 2, 0], pos[t:t + 2, 1], "-", color="red",
                         linewidth=2.5, alpha=0.8)
                # Small arrow showing the correction direction
                if t < len(ep["actions_safe"]):
                    diff = ep["actions_safe"][t] - ep["actions_nominal"][t]
                    scale = 0.15
                    ax2.annotate(
                        "",
                        xy=(pos[t, 0] + diff[0] * scale, pos[t, 1] + diff[1] * scale),
                        xytext=(pos[t, 0], pos[t, 1]),
                        arrowprops=dict(arrowstyle="->", color="orange", lw=1.5),
                    )

        # Start / end markers
        ax2.plot(pos[0, 0], pos[0, 1], "o", color="blue", markersize=7, zorder=5)
        ax2.plot(pos[-1, 0], pos[-1, 1], "s", color="purple", markersize=7, zorder=5)

    # Vc = 0 boundary overlay
    if state_dim == 2:
        ax2.contour(xx, yy, vc_grid, levels=[0.0], colors="k", linewidths=2.0, linestyles="--")

    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")

    ax2.plot([], [], "-", color="grey", label="Trajectory")
    ax2.plot([], [], "-", color="red", linewidth=2.5, label="CBF intervened")
    ax2.plot([], [], "->", color="orange", label="Correction direction")
    ax2.plot([], [], "--", color="k", label="$V_c = 0$ boundary")
    ax2.legend(loc="upper left", fontsize=9)

    # ---- Summary stats annotation ----
    total_steps = sum(len(e["rewards"]) for e in episodes)
    total_intervened = sum(e["intervened"].sum() for e in episodes)
    mean_return = np.mean([e["rewards"].sum() for e in episodes])
    mean_cbf_viol = np.mean([np.sum(e["vc_values"] < 0) for e in episodes])
    stats_text = (
        f"Episodes: {len(episodes)}  |  "
        f"Mean return: {mean_return:.1f}  |  "
        f"CBF interventions: {int(total_intervened)}/{total_steps} steps  |  "
        f"Mean $V_c < 0$ steps: {mean_cbf_viol:.1f}"
    )
    fig.suptitle(stats_text, fontsize=11, y=0.02)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig


def _draw_env_features(ax, env):
    """Draw obstacles and goal for BoatRobot (or PointRobot with hazards)."""
    # Obstacles
    if hasattr(env, "hazard_position_list"):
        for hp in env.hazard_position_list:
            radius = hp[2] if len(hp) > 2 else 0.4
            circle = Circle(
                (hp[0], hp[1]), radius,
                fill=True, alpha=0.35, facecolor="salmon",
                edgecolor="darkred", linewidth=1.5, label="_nolegend_",
            )
            ax.add_patch(circle)
            ax.text(hp[0], hp[1], "H", ha="center", va="center",
                    fontsize=9, color="darkred", fontweight="bold")

    # Goal
    if hasattr(env, "goal_position"):
        gp = env.goal_position
        goal_size = getattr(env, "goal_size", 0.1)
        circle = Circle(
            (gp[0], gp[1]), max(goal_size, 0.08),
            fill=True, alpha=0.5, facecolor="limegreen",
            edgecolor="darkgreen", linewidth=1.5,
        )
        ax.add_patch(circle)
        ax.text(gp[0], gp[1], "G", ha="center", va="center",
                fontsize=9, color="darkgreen", fontweight="bold")


# ---------------------------------------------------------------------------
# Load models (mirrors evaluate_cbf.py logic)
# ---------------------------------------------------------------------------

def load_models(model_dir, env_id):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = to_config_dict(json.load(f))

    env_name = env_list[env_id]

    if env_name == "PointRobot":
        env = PointRobot(id=0, seed=0)
    elif env_name == "BoatRobot":
        env = BoatRobot(id=0, seed=0)
    else:
        import gymnasium as gym
        env = gym.make(env_name)

    env_max_steps = env._max_episode_steps
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # VOCBF value functions
    config_dict = dict(cfg["agent_kwargs"])
    config_dict.pop("model_cls", None)
    config_dict.pop("cost_scale", None)
    for k in [
        "bc_hidden_dim", "bc_num_layers", "bc_lr", "bc_epochs",
        "dyn_hidden_dim", "dyn_num_layers", "dyn_lr", "dyn_epochs",
        "cbf_alpha",
    ]:
        config_dict.pop(k, None)
    config_dict["env_max_steps"] = env_max_steps

    agent = VOCBF.create(cfg["seed"], env.observation_space, env.action_space, **config_dict)
    model_file = get_latest_model_file(model_dir)
    agent = agent.load(model_file)
    print(f"Loaded VOCBF model from {model_file}")

    V_net = build_vc_network(agent, state_dim)
    print("Built V_net (Vc as CBF)")

    # BC policy
    bc_path = os.path.join(model_dir, "bc_policy.pt")
    bc_hidden_dim = int(cfg["agent_kwargs"].get("bc_hidden_dim", 256))
    bc_num_layers = int(cfg["agent_kwargs"].get("bc_num_layers", 3))
    bc_policy = BCPolicy(state_dim, action_dim, hidden_dim=bc_hidden_dim,
                         num_layers=bc_num_layers).to(DEFAULT_DEVICE)
    bc_policy.load_state_dict(torch.load(bc_path, map_location=DEFAULT_DEVICE))
    bc_policy.eval()
    print(f"Loaded BC policy from {bc_path}")

    # Dynamics
    dyn_path = os.path.join(model_dir, "dynamics_model.pt")
    dyn_hidden_dim = int(cfg["agent_kwargs"].get("dyn_hidden_dim", 64))
    dyn_num_layers = int(cfg["agent_kwargs"].get("dyn_num_layers", 3))
    dt = 1.0 / env_max_steps
    dynamics_model = AffineDynamics(
        num_action=action_dim, state_dim=state_dim,
        hidden_dim=dyn_hidden_dim, num_layers=dyn_num_layers, dt=dt,
    ).to(DEFAULT_DEVICE)
    dynamics_model.load_state_dict(torch.load(dyn_path, map_location=DEFAULT_DEVICE))
    dynamics_model.eval()
    print(f"Loaded dynamics model from {dyn_path}")

    return env, env_max_steps, bc_policy, V_net, dynamics_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    model_dir = FLAGS.model_dir
    assert model_dir, "Please provide --model_dir"

    env, max_steps, bc_policy, V_net, dynamics_model = load_models(model_dir, FLAGS.env_id)

    print(f"\nRunning {FLAGS.num_episodes} episodes for trajectory visualisation …")
    episodes = []
    for ep in range(FLAGS.num_episodes):
        data = run_episode_with_recording(
            env, ep, bc_policy, V_net, dynamics_model, max_steps, deterministic=True,
        )
        steps = len(data["rewards"])
        n_interv = data["intervened"].sum()
        vc_min = data["vc_values"].min()
        print(f"  Episode {ep}: {steps} steps, {n_interv} interventions, "
              f"Vc_min={vc_min:.3f}, return={data['rewards'].sum():.1f}")
        episodes.append(data)

    plot_trajectories(
        episodes, env, V_net,
        save_path=FLAGS.save_path, show=FLAGS.show,
        contour_res=FLAGS.contour_resolution,
    )
    print("Done.")


if __name__ == "__main__":
    app.run(main)
