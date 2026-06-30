# V-OCBF: Learning Safety Filters from Offline Data via Value-Guided Offline Control Barrier Functions

[![OpenReview](https://img.shields.io/badge/OpenReview-PGO9mpIyyb-blue.svg)](https://openreview.net/forum?id=PGO9mpIyyb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official code repository for the paper **V-OCBF: Learning Safety Filters from Offline Data via Value-Guided Offline Control Barrier Functions**.

**Authors**: Mumuksh Tayal, Manan Tayal, Aditya Singh, Shishir Kolathaya, Ravi Prakash.

## Abstract
The paper introduces Value-Guided Offline Control Barrier Functions (V-OCBF), a framework for learning neural Control Barrier Functions (CBFs) from offline demonstrations without requiring access to a dynamics model. It uses a recursive finite-difference barrier update to propagate safety information and an expectile-based objective to handle out-of-distribution actions. The learned barrier is utilized within a Quadratic Program (QP) to synthesize safe, real-time controls.

## Repository Structure

- `configs/`: Contains configuration files for different training runs (e.g., `train_config.py`).
- `data/`: Directory to store offline datasets (HDF5 format).
- `env/`: Contains the environment definitions (e.g., `boat_robot`).
- `jaxrl5/`: The core JAX-based reinforcement learning and V-OCBF algorithm implementation.
- `launcher/`: Scripts for executing training (`train_offline.py`) and visualization.
- `run.sh`: Main entry point script for running the offline training.

## Installation

We recommend using Anaconda/Miniconda to manage your environment. The environment configuration is provided in `environment.yaml`.

```bash
# Clone the repository
git clone https://github.com/tayalmanan28/V-OCBF.git
cd V-OCBF

# Create and activate the conda environment
conda env create -f environment.yaml
conda activate VOCBF
```

*Note: Depending on your specific CUDA drivers, you may need to ensure JAX is correctly configured to utilize your GPU. If you experience GPU fallback issues, you can explicitly update JAX for CUDA 12 via:*
```bash
pip install -U "jax[cuda12]"
```

## Usage

### Training

To start training the V-OCBF model on offline data, you can simply run the provided bash script:

```bash
./run.sh
```

Alternatively, you can manually trigger the training script and customize the arguments (such as the environment ID or dataset ratios):

```bash
python launcher/examples/train_offline.py --env_id 30 --config configs/train_config.py:vocbf
```

Key arguments for `train_offline.py`:
- `--env_id`: Identifier for the specific environment task.
- `--config`: Path to the hyperparameter config file.
- `--ratio`: The ratio/quantity of data to be used.

### Configuration
You can find and modify the hyperparameters for the value function training, actor-critic networks, learning rates, and cost limits inside `configs/train_config.py`.

### Results and Visualization
Model checkpoints and training logs are saved automatically to the `results/` directory. You can use the scripts within `launcher/viz/` to render environment visualizations (e.g., `viz_map.py`) and plot trajectories.

## Citation

If you find this code or our paper useful in your research, please consider citing our work:

```bibtex
@article{
tayal2026vocbf,
title={V-{OCBF}: Learning Safety Filters from Offline Data via Value-Guided Offline Control Barrier Functions},
author={Mumuksh Tayal and Manan Tayal and Aditya Singh and Shishir Kolathaya and Ravi Prakash},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=PGO9mpIyyb},
}
```

## License

This project is open-sourced under the MIT License.
