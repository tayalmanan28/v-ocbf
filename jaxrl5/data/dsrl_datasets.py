import os
import gymnasium as gym
import dsrl
import numpy as np
from jaxrl5.data.dataset import Dataset
import h5py


class DSRLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5, critic_type="qc", data_location=None, cost_scale=1., ratio = 1.0):

        if data_location is not None:
            # Point Robot
            dataset_dict = {}
            print('=========Data loading=========')
            print('Load point robot data from:', data_location)
            f = h5py.File(data_location, 'r')
            dataset_dict["observations"] = np.array(f['state'])
            dataset_dict["actions"] = np.array(f['action'])
            dataset_dict["next_observations"] = np.array(f['next_state'])
            dataset_dict["rewards"] = np.array(f['reward'])
            dataset_dict["dones"] = np.array(f['done'])
            dataset_dict['costs'] = np.array(f['h'])

            violation = np.array(f['cost'])
            print('env_max_episode_steps', env._max_episode_steps)
            print('mean_episode_reward', env._max_episode_steps * np.mean(dataset_dict['rewards']))
            print('mean_episode_cost', env._max_episode_steps * np.mean(violation))

        else:
            # DSRL
            if ratio == 1.0:
                dataset_dict = env.get_dataset()
            else:
                _, dataset_name = os.path.split(env.dataset_url)
                file_list = dataset_name.split('-')
                ratio_num = int(float(file_list[-1].split('.')[0]) * ratio)
                dataset_ratio = '-'.join(file_list[:-1]) + '-' + str(ratio_num) + '-' + str(ratio) + '.hdf5'
                dataset_dict = env.get_dataset(os.path.join('data', dataset_ratio))
            print('max_episode_reward', env.max_episode_reward, 
                'min_episode_reward', env.min_episode_reward,
                'mean_episode_reward', env._max_episode_steps * np.mean(dataset_dict['rewards']))
            print('max_episode_cost', env.max_episode_cost, 
                'min_episode_cost', env.min_episode_cost,
                'mean_episode_cost', env._max_episode_steps * np.mean(dataset_dict['costs']))
            print('data_num', dataset_dict['actions'].shape[0])
            dataset_dict['dones'] = np.logical_or(dataset_dict["terminals"],
                                                dataset_dict["timeouts"]).astype(np.float32)
            del dataset_dict["terminals"]
            del dataset_dict['timeouts']

            if critic_type == "hj":
                # Compute continuous h = -(velocity - velocity_threshold)
                # h > 0 when velocity < threshold (safe)
                # h < 0 when velocity > threshold (unsafe)
                env_id = env.spec.id.lower() if env.spec else ""
                states = dataset_dict['observations']

                # Extract velocity from observations based on environment
                if 'ant' in env_id:
                    vx = states[:, 13]
                    vy = states[:, 14]
                    velocity = np.sqrt(vx**2 + vy**2)
                    velocity_threshold = 2.6222
                elif 'halfcheetah' in env_id:
                    velocity = states[:, 8]
                    velocity_threshold = 3.2096
                elif 'hopper' in env_id:
                    velocity = states[:, 5]
                    velocity_threshold = 0.7402
                elif 'walker2d' in env_id:
                    velocity = states[:, 8]
                    velocity_threshold = 2.3415
                elif 'swimmer' in env_id:
                    velocity = states[:, 3]
                    velocity_threshold = 0.2282
                else:
                    raise ValueError(f"Unknown velocity env: {env_id}. "
                                     "Cannot compute h for CBF.")

                h_values = -(velocity - velocity_threshold)
                print(f"  Velocity env: {env_id}")
                print(f"  velocity_threshold: {velocity_threshold}")
                print(f"  h stats: min={h_values.min():.4f}, max={h_values.max():.4f}, "
                      f"mean={h_values.mean():.4f}, frac_safe={np.mean(h_values > 0):.4f}")
                dataset_dict['costs'] = h_values

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["masks"] = 1.0 - dataset_dict['dones']
        del dataset_dict['dones']

        super().__init__(dataset_dict)
