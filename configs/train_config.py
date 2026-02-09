from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        seed=-1,
        max_steps=1000001,
        eval_episodes=20,
        batch_size=1024, # Critic mini-batch is fixed to 256
        log_interval=25000,
        eval_interval=250000,
        normalize_returns=True,
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=25,
        pr_data='data/point_robot-expert-random-100k.hdf5',  # The location of point_robot data
        boat_data='data/boat-1M_mix.hdf5',  # The location of boat_robot data
    )

    possible_structures = {
        "fisor": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="FISOR",
                    cost_limit=10,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    critic_objective='expectile',
                    critic_hyperparam=0.9,
                    cost_critic_hyperparam=0.9,
                    critic_type="hj",  # [hj, qc]
                    cost_ub=150,
                    value_layer_norm=False,
                    # BC policy training (PyTorch)
                    bc_hidden_dim=256,
                    bc_num_layers=3,
                    bc_lr=3e-4,
                    bc_epochs=200,
                    # Dynamics model training (PyTorch)
                    dyn_hidden_dim=64,
                    dyn_num_layers=3,
                    dyn_lr=1e-3,
                    dyn_epochs=100,
                    # CBF evaluation
                    cbf_alpha=1.0,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]