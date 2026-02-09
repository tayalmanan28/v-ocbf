"""
Standalone CBF-QP evaluation script.

Usage:
    python launcher/examples/evaluate_cbf.py \
        --model_dir ./results/PointRobot/vocbf_hj_PointRobot_... \
        --env_id 29 \
        --num_episodes 20

This loads:
  - FISOR value functions (Vc used as CBF) from a .pickle checkpoint
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

from ml_collections import ConfigDict
from env.env_list import env_list
from env.point_robot import PointRobot
from env.boat_robot import BoatRobot
from jaxrl5.agents import FISOR
from jaxrl5.agents.fisor.fisor import (
    BCPolicy, AffineDynamics, evaluate_cbf, DEFAULT_DEVICE,
)


FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '', 'Path to the results directory containing model files')
flags.DEFINE_integer('env_id', 29, 'Environment index from env_list')
flags.DEFINE_integer('num_episodes', 20, 'Number of evaluation episodes')
flags.DEFINE_float('cbf_alpha', 1.0, 'CBF alpha parameter')


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

    # Load config
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        cfg = to_config_dict(json.load(f))

    env_name = env_list[FLAGS.env_id]

    # Create environment
    if env_name == 'PointRobot':
        env = PointRobot(id=0, seed=0)
    elif env_name == 'BoatRobot':
        env = BoatRobot(id=0, seed=0)
    else:
        env = gym.make(env_name)
    env_max_steps = env._max_episode_steps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- Load FISOR value functions ---
    config_dict = dict(cfg['agent_kwargs'])
    model_cls = config_dict.pop("model_cls")
    config_dict.pop("cost_scale", None)
    for k in ['bc_hidden_dim', 'bc_num_layers', 'bc_lr', 'bc_epochs',
              'dyn_hidden_dim', 'dyn_num_layers', 'dyn_lr', 'dyn_epochs',
              'cbf_alpha']:
        config_dict.pop(k, None)
    config_dict['env_max_steps'] = env_max_steps

    agent = FISOR.create(
        cfg['seed'], env.observation_space, env.action_space, **config_dict
    )
    model_file = get_latest_model_file(model_dir)
    agent = agent.load(model_file)
    print(f"Loaded FISOR model from {model_file}")

    # --- Build Vc network (CBF) from JAX params ---
    from jaxrl5.agents.fisor.fisor import build_vc_network
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
    dt = 1.0 / env_max_steps
    dynamics_model = AffineDynamics(
        num_action=action_dim, state_dim=state_dim,
        hidden_dim=dyn_hidden_dim, num_layers=dyn_num_layers, dt=dt
    ).to(DEFAULT_DEVICE)
    dynamics_model.load_state_dict(torch.load(dyn_path, map_location=DEFAULT_DEVICE))
    dynamics_model.eval()
    print(f"Loaded dynamics model from {dyn_path}")

    # --- Run evaluation ---
    print(f"\nRunning {FLAGS.num_episodes} evaluation episodes on {env_name}...")
    eval_info = evaluate_cbf(
        env, FLAGS.num_episodes, bc_policy, V_net, dynamics_model,
        env_max_steps, deterministic=True
    )

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({FLAGS.num_episodes} episodes):")
    for k, v in eval_info.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    app.run(main)
