"""Evaluation utilities.

The primary evaluation is now done via CBF-QP safety filtering using:
  - A BC reference policy
  - The learned Vc as a Control Barrier Function
  - A learned control-affine dynamics model

See jaxrl5.agents.fisor.fisor for evaluate_policy() and evaluate_cbf().
"""
from typing import Dict

import gymnasium as gym
import numpy as np
import time
from tqdm.auto import trange


def evaluate_value_functions(agent, dataset, num_samples: int = 1024) -> Dict[str, float]:
    """
    Evaluate the quality of learned value functions on held-out data.
    Useful for monitoring training progress without needing a policy.
    """
    import jax.numpy as jnp

    sample = dataset.sample_jax(num_samples)

    # Reward Q values
    qs = agent.target_critic.apply_fn(
        {"params": agent.target_critic.params},
        sample["observations"],
        sample["actions"],
    )
    q_mean = float(qs.min(axis=0).mean())

    # Reward V values
    v = agent.value.apply_fn(
        {"params": agent.value.params}, sample["observations"]
    )
    v_mean = float(v.mean())

    # Cost Qc values
    qcs = agent.safe_target_critic.apply_fn(
        {"params": agent.safe_target_critic.params},
        sample["observations"],
        sample["actions"],
    )
    qc_mean = float(qcs.max(axis=0).mean())

    # Cost Vc values
    vc = agent.safe_value.apply_fn(
        {"params": agent.safe_value.params}, sample["observations"]
    )
    vc_mean = float(vc.mean())
    vc_positive_frac = float((vc > 0).mean())

    return {
        "q_mean": q_mean,
        "v_mean": v_mean,
        "qc_mean": qc_mean,
        "vc_mean": vc_mean,
        "vc_positive_fraction": vc_positive_frac,
    }
