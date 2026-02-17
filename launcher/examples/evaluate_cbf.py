"""
Standalone CBF-QP evaluation script.

Supports all environments: Safety Gymnasium (DSRL), PointRobot, BoatRobot.
The environment name is auto-detected from config.json in the model directory.

Usage:
    python launcher/examples/evaluate_cbf.py \
        --model_dir results/OfflineHalfCheetahVelocityGymnasium-v1/vocbf_hj_... \
        --num_episodes 20

This loads:
  - VOCBF value functions (Vc used as CBF) from the latest .pickle checkpoint
  - BC policy from bc_policy.pt
  - Dynamics model from dynamics_model.pt
Then runs CBF-QP evaluation episodes.
"""
import os
import sys
sys.path.append('.')
import re
import json
import numpy as np
from absl import app, flags
import torch
import gymnasium as gym
import dsrl  # registers DSRL environments with gymnasium

from ml_collections import ConfigDict
from env.point_robot import PointRobot
from env.boat_robot import BoatRobot
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import VOCBF
from jaxrl5.agents.vocbf.vocbf import (
    BCPolicy, AffineDynamics, evaluate_cbf, build_vc_network, DEFAULT_DEVICE,
)


FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '', 'Path to the results directory containing model files')
flags.DEFINE_integer('num_episodes', 200, 'Number of evaluation episodes')
flags.DEFINE_float('cbf_alpha', None, 'CBF alpha (default: from config, higher = more conservative)')
flags.DEFINE_bool('render', False, 'Render the environment during evaluation')


def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d


def get_latest_model_file(model_dir):
    """Find the latest .pickle checkpoint in the directory."""
    files = os.listdir(model_dir)
    pickle_files = [f for f in files if f.endswith('.pickle')]
    if not pickle_files:
        raise FileNotFoundError(f"No .pickle files found in {model_dir}")

    numbers = {}
    for f in pickle_files:
        match = re.search(r'\d+', f)
        if match:
            numbers[int(match.group())] = os.path.join(model_dir, f)

    return numbers[max(numbers.keys())]


def main(_):
    model_dir = FLAGS.model_dir
    assert model_dir, "Please provide --model_dir"

    # Load config (auto-detect env_name)
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        cfg = to_config_dict(json.load(f))

    env_name = cfg['env_name']
    print(f"Environment: {env_name}")

    # --- Create environment ---
    if env_name == 'PointRobot':
        env = PointRobot(id=0, seed=0)
        env_max_steps = env._max_episode_steps
    elif env_name == 'BoatRobot':
        env = BoatRobot(id=0, seed=0)
        env_max_steps = env._max_episode_steps
    else:
        # DSRL / Safety Gymnasium environments
        env = gym.make(env_name)
        env_max_steps = env._max_episode_steps  # grab before wrapping
        cost_limit = int(cfg['agent_kwargs'].get('cost_limit', 10))
        env = wrap_gym(env, cost_limit=cost_limit)

        # DSRL's gym.make ignores render_mode, so we patch the underlying
        # MuJoCo renderer directly to enable human rendering.
        if FLAGS.render:
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
            unwrapped = env.unwrapped
            unwrapped.mujoco_renderer = MujocoRenderer(
                unwrapped.model, unwrapped.data,
                getattr(unwrapped, 'default_camera_config', None),
            )
            unwrapped.render_mode = 'human'
            print("Rendering enabled (patched MuJoCo renderer)")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"  state_dim={state_dim}, action_dim={action_dim}, max_steps={env_max_steps}")

    # --- Load VOCBF value functions ---
    config_dict = dict(cfg['agent_kwargs'])
    config_dict.pop("model_cls")
    config_dict.pop("cost_scale", None)
    config_dict.pop("cost_limit", None)
    for k in ['bc_hidden_dim', 'bc_num_layers', 'bc_lr', 'bc_epochs',
              'dyn_hidden_dim', 'dyn_num_layers', 'dyn_lr', 'dyn_epochs',
              'cbf_alpha']:
        config_dict.pop(k, None)
    config_dict['env_max_steps'] = env_max_steps

    agent = VOCBF.create(
        cfg['seed'], env.observation_space, env.action_space, **config_dict
    )
    model_file = get_latest_model_file(model_dir)
    agent = agent.load(model_file)
    print(f"Loaded VOCBF model from {model_file}")

    # --- Build Vc network (CBF) from JAX params ---
    V_net = build_vc_network(agent, state_dim)
    print("Built V_net (Vc as CBF) from JAX safe_value params")

    # --- Load BC policy ---
    bc_path = os.path.join(model_dir, 'bc_policy.pt')
    bc_hidden_dim = int(cfg['agent_kwargs'].get('bc_hidden_dim', 256))
    bc_num_layers = int(cfg['agent_kwargs'].get('bc_num_layers', 3))
    bc_policy = BCPolicy(state_dim, action_dim, hidden_dim=bc_hidden_dim,
                         num_layers=bc_num_layers).to(DEFAULT_DEVICE)
    bc_policy.load_state_dict(torch.load(bc_path, map_location=DEFAULT_DEVICE))
    bc_policy.eval()
    print(f"Loaded BC policy from {bc_path}")

    # --- Load dynamics model ---
    dyn_path = os.path.join(model_dir, 'dynamics_model.pt')
    dyn_hidden_dim = int(cfg['agent_kwargs'].get('dyn_hidden_dim', 64))
    dyn_num_layers = int(cfg['agent_kwargs'].get('dyn_num_layers', 3))
    dt = 0.05  # Must match the dt used during dynamics training
    dynamics_model = AffineDynamics(
        num_action=action_dim, state_dim=state_dim,
        hidden_dim=dyn_hidden_dim, num_layers=dyn_num_layers, dt=dt
    ).to(DEFAULT_DEVICE)
    dynamics_model.load_state_dict(torch.load(dyn_path, map_location=DEFAULT_DEVICE))
    dynamics_model.eval()
    print(f"Loaded dynamics model from {dyn_path} (dt={dt})")

    # --- Run evaluation ---
    cbf_alpha = FLAGS.cbf_alpha if FLAGS.cbf_alpha is not None else float(cfg['agent_kwargs'].get('cbf_alpha', 1.0))
    print(f"\nRunning {FLAGS.num_episodes} evaluation episodes on {env_name} (cbf_alpha={cbf_alpha})...")
    eval_info = evaluate_cbf(
        env, FLAGS.num_episodes, bc_policy, V_net, dynamics_model,
        env_max_steps, deterministic=True, cbf_alpha=cbf_alpha
    )

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({FLAGS.num_episodes} episodes):")
    for k, v in eval_info.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    app.run(main)
