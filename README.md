# NDKF: A Neural-Enhanced Distributed Kalman Filter for Nonlinear Multi-Sensor Estimation

This repository contains the implementation of the Neural-Enhanced Distributed Kalman Filter (NDKF) as described in our paper:

**[NDKF: A Neural-Enhanced Distributed Kalman Filter for Nonlinear Multi-Sensor Estimation](<https://arxiv.org/abs/2502.02687>)**

NDKF is a data-driven framework for multi-sensor state estimation in nonlinear systems. By leveraging neural networks to learn both system dynamics and measurement functions from data, NDKF performs local prediction and update steps at each sensor node and fuses information via a consensus-based process. This approach reduces communication overhead and avoids a single point of failure, while improving scalability, robustness, and accuracy compared to traditional Kalman filters.

## Overview

The NDKF framework integrates neural network models within a distributed Kalman filtering scheme. Key features include:

- **Learned Dynamics:** A neural network learns the residual dynamics of the system.
- **Local Measurement Models:** Each sensor node uses a neural network to model its own measurement function.
- **Consensus-Based Fusion:** Local estimates are fused using an information-weighted consensus approach.
- **Improved Performance:** Simulation results on a 2D system with four sensor nodes show that NDKF outperforms a distributed Extended Kalman Filter under challenging nonlinear conditions.

## File Structure

- **ndkf.py:** Contains the `NDKF` class, neural network definitions, and training routines.
- **example_ndkf.py:** Provides an example that trains the networks, runs the NDKF simulation on a 2D system with 4 nodes, and produces plots for both the state trajectory and measurement innovation residuals.

## Installation

Ensure you have Python 3.6 or later installed. The required Python packages are:

- [numpy](https://pypi.org/project/numpy/)
- [torch](https://pypi.org/project/torch/)
- [matplotlib](https://pypi.org/project/matplotlib/)

You can install these dependencies using pip:

```bash
pip install numpy torch matplotlib
```
## Usage

To train the NDKF models and run the simulation, execute the example script:

```bash
python example_ndkf.py
```

This script will:

-   Train the neural network for the residual dynamics.
-   Train measurement networks for each sensor node.
-   Run the NDKF simulation on a 2D system with 4 nodes.
-   Generate plots of the system trajectories and measurement innovation residuals.

## Citation

If you use this code in your research, please cite our [paper](https://arxiv.org/abs/2502.02687):

```bibtex
@misc{NDKF2025,
  title={{NDKF}: A Neural-Enhanced Distributed {Kalman} Filter for Nonlinear Multi-Sensor Estimation},
  author={Farzan, Siavash and Parisi, Bennett},
  year={2025},
  eprint={2502.02687},
  archivePrefix={arXiv},
  primaryClass={eess.SY},
  url={https://arxiv.org/abs/2502.02687},
}
```
