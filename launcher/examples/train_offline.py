import os
import sys
sys.path.append('.')
import random
import numpy as np
from absl import app, flags
import datetime
import yaml
from ml_collections import config_flags, ConfigDict
from tqdm.auto import trange
import gymnasium as gym
import torch
import torch.nn as tnn
import torch.optim as optim

from env.env_list import env_list
from env.point_robot import PointRobot
from env.boat_robot import BoatRobot
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import VOCBF
from jaxrl5.agents.vocbf.vocbf import (
    BCPolicy, AffineDynamics, evaluate_cbf, torchify, DEFAULT_DEVICE,
    build_vc_network,
)
from jaxrl5.data.dsrl_datasets import DSRLDataset
import json


FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 30, 'Choose env')
flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_string('experiment_name', '', 'experiment name')
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config


# ================================================================
# BC policy training (PyTorch)
# ================================================================

def train_bc_policy(dataset, state_dim, action_dim, hidden_dim=256,
                    num_layers=3, lr=3e-4, epochs=200, batch_size=256):
    """Train a Behavior Cloning policy on the offline dataset."""
    policy = BCPolicy(state_dim, action_dim, hidden_dim=hidden_dim,
                      num_layers=num_layers).to(DEFAULT_DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    observations = torch.FloatTensor(dataset.dataset_dict['observations']).to(DEFAULT_DEVICE)
    actions = torch.FloatTensor(dataset.dataset_dict['actions']).to(DEFAULT_DEVICE)
    n = observations.shape[0]

    print(f"\n--- Training BC Policy ({epochs} epochs, {n} samples) ---")
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.
        num_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            obs_batch = observations[idx]
            act_batch = actions[idx]

            pred_actions = policy(obs_batch)
            loss = tnn.functional.mse_loss(pred_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / num_batches
            print(f"  BC Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.6f}")

    print("  BC training complete.\n")
    return policy


# ================================================================
# Dynamics model training (PyTorch)
# ================================================================

def train_dynamics_model(dataset, state_dim, action_dim, hidden_dim=64,
                         num_layers=3, lr=1e-3, epochs=100, batch_size=256,
                         dt=0.05, val_ratio=0.2, patience=30):
    """Train a control-affine dynamics model on the offline dataset."""
    model = AffineDynamics(
        num_action=action_dim, state_dim=state_dim,
        hidden_dim=hidden_dim, num_layers=num_layers, dt=dt
    ).to(DEFAULT_DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    observations = torch.FloatTensor(dataset.dataset_dict['observations']).to(DEFAULT_DEVICE)
    actions = torch.FloatTensor(dataset.dataset_dict['actions']).to(DEFAULT_DEVICE)
    next_observations = torch.FloatTensor(dataset.dataset_dict['next_observations']).to(DEFAULT_DEVICE)
    n = observations.shape[0]

    # Train/validation split
    n_val = int(n * val_ratio)
    n_train = n - n_val
    perm_all = torch.randperm(n)
    train_idx = perm_all[:n_train]
    val_idx = perm_all[n_train:]

    train_obs, train_act, train_next = observations[train_idx], actions[train_idx], next_observations[train_idx]
    val_obs, val_act, val_next = observations[val_idx], actions[val_idx], next_observations[val_idx]

    print(f"\n--- Training Dynamics Model ({epochs} epochs, {n_train} train / {n_val} val samples, dt={dt}) ---")

    best_val_loss = float('inf')
    best_state_dict = None
    patience_counter = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.
        num_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            pred_next = model.forward_next_state(train_obs[idx], train_act[idx])
            loss = tnn.functional.mse_loss(pred_next, train_next[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_pred = model.forward_next_state(val_obs, val_act)
            val_loss = tnn.functional.mse_loss(val_pred, val_next).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Dynamics Epoch {epoch+1}/{epochs}  Train: {avg_train_loss:.6f}  Val: {val_loss:.6f}")

        # Early stopping with best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
                break

    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    print(f"  Dynamics training complete. Best val loss: {best_val_loss:.6f}\n")
    return model


# ================================================================
# Main training loop
# ================================================================

def call_main(details):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']

    # --- Environment and dataset setup ---
    if details['env_name'] == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'],
                         data_location=details['dataset_kwargs']['pr_data'])
    elif details['env_name'] == 'BoatRobot':
        assert details['dataset_kwargs']['boat_data'] is not None, "No data for Boat Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'],
                         data_location=details['dataset_kwargs']['boat_data'])
    else:
        env = gym.make(details['env_name'])
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'],
                         cost_scale=details['dataset_kwargs']['cost_scale'],
                         ratio=details['ratio'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])

    # --- Create VOCBF agent (value functions only) ---
    config_dict = dict(details['agent_kwargs'])
    config_dict['env_max_steps'] = env_max_steps

    model_cls = config_dict.pop("model_cls")
    config_dict.pop("cost_scale")
    # Remove BC/dynamics/CBF params before passing to VOCBF.create
    bc_hidden_dim = config_dict.pop("bc_hidden_dim", 256)
    bc_num_layers = config_dict.pop("bc_num_layers", 3)
    bc_lr = config_dict.pop("bc_lr", 3e-4)
    bc_epochs = config_dict.pop("bc_epochs", 200)
    dyn_hidden_dim = config_dict.pop("dyn_hidden_dim", 64)
    dyn_num_layers = config_dict.pop("dyn_num_layers", 3)
    dyn_lr = config_dict.pop("dyn_lr", 1e-3)
    dyn_epochs = config_dict.pop("dyn_epochs", 300)
    cbf_alpha = config_dict.pop("cbf_alpha", 1.0)

    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- Phase 1: Train value functions (JAX) ---
    print("\n========== Phase 1: Training Value Functions ==========")
    save_time = 1
    for i in trange(details['max_steps'], smoothing=0.1, desc='Value Function Training'):
        sample = ds.sample_jax(details['batch_size'])
        agent, info = agent.update(sample)

        if i % details['log_interval'] == 0:
            for k, v in info.items():
                print(f"  step {i} | train/{k}: {v:.4f}")

        if i % details['eval_interval'] == 0:
            agent.save(f"./results/{details['group']}/{details['experiment_name']}", save_time)
            save_time += 1

    # Final save
    agent.save(f"./results/{details['group']}/{details['experiment_name']}", save_time)
    print("Value function training complete.\n")

    # --- Phase 2: Train BC policy (PyTorch) ---
    # print("========== Phase 2: Training BC Policy ==========")
    # bc_policy = train_bc_policy(
    #     ds, state_dim, action_dim,
    #     hidden_dim=bc_hidden_dim, num_layers=bc_num_layers,
    #     lr=bc_lr, epochs=bc_epochs
    # )
    # # Save BC policy
    # bc_path = f"./results/{details['group']}/{details['experiment_name']}/bc_policy.pt"
    # torch.save(bc_policy.state_dict(), bc_path)
    # print(f"  BC policy saved to {bc_path}")

    # --- Phase 3: Train dynamics model (PyTorch) ---
    print("========== Phase 3: Training Dynamics Model ==========")
    dt = 0.05
    dynamics_model = train_dynamics_model(
        ds, state_dim, action_dim,
        hidden_dim=dyn_hidden_dim, num_layers=dyn_num_layers,
        lr=dyn_lr, epochs=dyn_epochs, dt=dt
    )
    # Save dynamics model
    dyn_path = f"./results/{details['group']}/{details['experiment_name']}/dynamics_model.pt"
    torch.save(dynamics_model.state_dict(), dyn_path)
    print(f"  Dynamics model saved to {dyn_path}")

    # --- Phase 4: Build Vc network (CBF) and evaluate ---
    print("\n========== Phase 4: CBF-QP Evaluation ==========")
    V_net = build_vc_network(agent, state_dim)

    eval_info = evaluate_cbf(
        env, details['eval_episodes'], bc_policy, V_net, dynamics_model,
        env_max_steps, deterministic=True, cbf_alpha=cbf_alpha
    )
    print(f"\nEvaluation results:")
    for k, v in eval_info.items():
        print(f"  {k}: {v:.4f}")

    print("\nDone!")


def main(_):
    parameters = FLAGS.config
    parameters['env_name'] = env_list[FLAGS.env_id]
    parameters['ratio'] = FLAGS.ratio
    parameters['group'] = parameters['env_name']

    parameters['experiment_name'] = 'vocbf_' \
                                + parameters['agent_kwargs']['critic_type'] + '_' \
                                + parameters['env_name'] if FLAGS.experiment_name == '' else FLAGS.experiment_name
    parameters['experiment_name'] += '_' + str(datetime.date.today()) + '_s' + str(parameters['seed']) + '_' + str(random.randint(0, 1000))

    if parameters['env_name'] == 'PointRobot' or parameters['env_name'] == 'BoatRobot':
        parameters['max_steps'] = 100001
        parameters['batch_size'] = 256
        parameters['eval_interval'] = 25000

    print(parameters)

    if not os.path.exists(f"./results/{parameters['group']}/{parameters['experiment_name']}"):
        os.makedirs(f"./results/{parameters['group']}/{parameters['experiment_name']}")
    with open(f"./results/{parameters['group']}/{parameters['experiment_name']}/config.json", "w") as f:
        json.dump(to_dict(parameters), f, indent=4)

    call_main(parameters)


if __name__ == '__main__':
    app.run(main)
