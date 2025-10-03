"""
Example Usage of the NDKF Class

Authors: Siavash Farzan
Date: 10-02-2025
License: MIT License

Description:
    This script demonstrates the use of the NDKF class on a simulated 2D system
    monitored by four sensor nodes. In addition to the fused state trajectory,
    it plots the measurement innovation residuals for Node 1.
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# Import the NDKF class and network functions from ndkf.py
from ndkf import (
    NDKF,
    DynamicsResidualNet2D,
    MeasurementNet2D,
    train_residual_dynamics_net,
    train_measurement_net
)

def set_random_seeds(seed: Optional[int] = None) -> int:
    if seed is None:
        # Generate a random seed based on current time
        seed = int(time.time() * 1000) % (2**32)
    
    # Set seeds for all random number generators
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    return seed

# ---------------------------
# Configuration Constants
# ---------------------------
@dataclass
class SystemConfig:
    """Configuration parameters for the 2D system."""
    # Dynamics parameters
    velocity_amplitude: float = 0.05
    velocity_frequency: float = 10.0
    nonlinear_coupling: float = 0.02
    process_noise_std: float = 0.002
    
    # Measurement parameters
    measurement_noise_std: float = 0.1
    measurement_amplitude: float = 0.5
    measurement_freq_multiplier: float = 2.0
    measurement_coef_x: float = 0.4
    
    # Network architecture
    dynamics_hidden_size: int = 128
    dynamics_dropout: float = 0.2
    measurement_hidden_size: int = 32
    
    # Training parameters
    dynamics_epochs: int = 3000
    measurement_epochs: int = 1000
    learning_rate: float = 0.001
    training_steps: int = 1000
    
    # Filter parameters
    initial_covariance_scale: float = 0.5
    process_covariance_diag: float = 0.005
    measurement_covariance: float = 0.01
    
    # Simulation parameters
    num_simulation_steps: int = 100
    num_sensor_nodes: int = 4
    state_dimension: int = 2
    initial_state: np.ndarray = None
    
    def __post_init__(self):
        if self.initial_state is None:
            self.initial_state = np.zeros(self.state_dimension, dtype=np.float32)


# ---------------------------
# System Dynamics and Measurement Models
# ---------------------------
class TrueDynamicsModel:
    """True nonlinear dynamics model for the 2D system."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self._process_cov = np.diag([config.process_noise_std] * config.state_dimension)
    
    def evolve(self, x: np.ndarray, k: int) -> np.ndarray:
        """
        Evolve system state forward by one time step.
        
        Args:
            x: Current state [px, py]
            k: Time step index
            
        Returns:
            Next state with process noise
        """
        cfg = self.config
        time_scaled = k / cfg.velocity_frequency
        
        vx = cfg.velocity_amplitude * np.cos(time_scaled) + \
             cfg.nonlinear_coupling * np.sin(3 * x[1])
        vy = cfg.velocity_amplitude * np.sin(time_scaled) + \
             cfg.nonlinear_coupling * np.cos(2 * x[0])
        
        process_noise = np.random.multivariate_normal(
            mean=np.zeros(cfg.state_dimension),
            cov=self._process_cov
        )
        
        return x + np.array([vx, vy]) + process_noise


class TrueMeasurementModel:
    """True nonlinear measurement model for sensor nodes."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def measure(self, node_id: int, x: np.ndarray) -> float:
        """
        Generate measurement from a sensor node.
        
        Args:
            node_id: Sensor node identifier (1-4)
            x: State vector [px, py]
            
        Returns:
            Measurement value with noise
        """
        cfg = self.config
        px, py = x
        noise = np.random.normal(0, cfg.measurement_noise_std)
        
        freq_px = cfg.measurement_freq_multiplier * px
        freq_py = cfg.measurement_freq_multiplier * py
        
        if node_id == 1:
            return np.sin(freq_px) + cfg.measurement_amplitude * py + noise
        elif node_id == 2:
            return np.cos(freq_py) - cfg.measurement_coef_x * px + noise
        elif node_id == 3:
            return np.sin(freq_px) + np.cos(freq_py) + noise
        else:  # node_id == 4
            return np.sin(freq_px) - np.cos(freq_py) + noise


# ---------------------------
# Training Data Generation
# ---------------------------
class TrainingDataGenerator:
    """Generate training data for dynamics and measurement networks."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.dynamics_model = TrueDynamicsModel(config)
        self.measurement_model = TrueMeasurementModel(config)
    
    def generate_dynamics_residual_data(
        self,
        num_steps: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for dynamics residual network.
        
        Args:
            num_steps: Number of time steps (defaults to config value)
            
        Returns:
            Tuple of (state+time features, residuals)
        """
        if num_steps is None:
            num_steps = self.config.training_steps
        
        cfg = self.config
        x_vals = []
        r_vals = []
        x = self.config.initial_state.copy()
        
        for k in range(num_steps):
            t = k / num_steps
            u = self._compute_control_input(k)
            x_next = self.dynamics_model.evolve(x, k)
            residual = x_next - (x + u)
            
            x_vals.append(np.concatenate((x, [t])))
            r_vals.append(residual)
            x = x_next.astype(np.float32)
        
        return np.array(x_vals, dtype=np.float32), np.array(r_vals, dtype=np.float32)
    
    def generate_measurement_data(
        self,
        node_id: int,
        num_steps: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for measurement network.
        
        Args:
            node_id: Sensor node identifier
            num_steps: Number of time steps (defaults to config value)
            
        Returns:
            Tuple of (states, measurements)
        """
        if num_steps is None:
            num_steps = self.config.training_steps
        
        x_vals = []
        z_vals = []
        x = self.config.initial_state.copy()
        
        for k in range(num_steps):
            x_vals.append(x.copy())
            z = self.measurement_model.measure(node_id, x)
            z_vals.append([z])
            x = self.dynamics_model.evolve(x, k)
        
        return np.array(x_vals, dtype=np.float32), np.array(z_vals, dtype=np.float32)
    
    def _compute_control_input(self, k: int) -> np.ndarray:
        """Compute nominal control input at time step k."""
        cfg = self.config
        time_scaled = k / cfg.velocity_frequency
        return np.array([
            cfg.velocity_amplitude * np.cos(time_scaled),
            cfg.velocity_amplitude * np.sin(time_scaled)
        ])


# ---------------------------
# Network Training
# ---------------------------
class NetworkTrainer:
    """Train dynamics and measurement networks."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.data_generator = TrainingDataGenerator(config)
    
    def train_dynamics_network(self) -> DynamicsResidualNet2D:
        """Train the dynamics residual network."""
        print("Training dynamics residual network...")
        cfg = self.config
        
        x_train, r_train = self.data_generator.generate_dynamics_residual_data()
        
        dyn_net = DynamicsResidualNet2D(
            input_dim=cfg.state_dimension + 1,  # state + time
            hidden_size=cfg.dynamics_hidden_size,
            dropout_prob=cfg.dynamics_dropout
        )
        
        return train_residual_dynamics_net(
            dyn_net,
            x_train,
            r_train,
            epochs=cfg.dynamics_epochs,
            lr=cfg.learning_rate
        )
    
    def train_measurement_networks(self) -> List[MeasurementNet2D]:
        """Train measurement networks for all sensor nodes."""
        meas_nets = []
        cfg = self.config
        
        for node_id in range(1, cfg.num_sensor_nodes + 1):
            print(f"Training measurement network for Node {node_id}...")
            
            x_train, z_train = self.data_generator.generate_measurement_data(node_id)
            
            net_meas = MeasurementNet2D(hidden_size=cfg.measurement_hidden_size)
            net_meas = train_measurement_net(
                net_meas,
                x_train,
                z_train,
                epochs=cfg.measurement_epochs,
                lr=cfg.learning_rate
            )
            meas_nets.append(net_meas)
        
        return meas_nets


# ---------------------------
# NDKF Simulation
# ---------------------------
class NDKFSimulator:
    """Simulate and visualize NDKF performance."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.dynamics_model = TrueDynamicsModel(config)
        self.measurement_model = TrueMeasurementModel(config)
    
    def create_filter(
        self,
        dyn_net: DynamicsResidualNet2D,
        meas_nets: List[MeasurementNet2D]
    ) -> NDKF:
        """Create and initialize the NDKF filter."""
        cfg = self.config
        
        initial_state = cfg.initial_state.copy()
        initial_cov = cfg.initial_covariance_scale * np.eye(
            cfg.state_dimension,
            dtype=np.float32
        )
        
        Qs = [
            np.diag([cfg.process_covariance_diag] * cfg.state_dimension)
            for _ in range(cfg.num_sensor_nodes)
        ]
        Rs = [
            np.array([[cfg.measurement_covariance]])
            for _ in range(cfg.num_sensor_nodes)
        ]
        
        return NDKF(dyn_net, meas_nets, initial_state, initial_cov, Qs, Rs)
    
    def run_simulation(
        self,
        ndkf_filter: NDKF
    ) -> Tuple[np.ndarray, np.ndarray, List[List[np.ndarray]]]:
        """
        Run the NDKF simulation.
        
        Returns:
            Tuple of (true_trajectory, fused_trajectory, node_trajectories)
        """
        cfg = self.config
        x_true = cfg.initial_state.copy()
        true_history = []
        fused_history = []
        node_history = [[] for _ in range(cfg.num_sensor_nodes)]
        
        for k in range(cfg.num_simulation_steps):
            # Evolve true system
            x_true = self.dynamics_model.evolve(x_true, k)
            true_history.append(x_true.copy())
            
            # Compute control input and time parameter
            u = self._compute_control_input(k)
            t = k / cfg.num_simulation_steps
            
            # Generate measurements
            measurements = [
                self.measurement_model.measure(node_id, x_true)
                for node_id in range(1, cfg.num_sensor_nodes + 1)
            ]
            
            # Perform filter step
            x_fused, node_estimates, _ = ndkf_filter.step(k, u, t, measurements)
            fused_history.append(x_fused.copy())
            
            for i in range(cfg.num_sensor_nodes):
                node_history[i].append(node_estimates[i].copy())
        
        return (
            np.array(true_history),
            np.array(fused_history),
            node_history
        )
    
    def _compute_control_input(self, k: int) -> np.ndarray:
        """Compute nominal control input at time step k."""
        cfg = self.config
        time_scaled = k / cfg.velocity_frequency
        return np.array([
            cfg.velocity_amplitude * np.cos(time_scaled),
            cfg.velocity_amplitude * np.sin(time_scaled)
        ])


# ---------------------------
# Visualization
# ---------------------------
class ResultVisualizer:
    """Visualize simulation results."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def plot_trajectories(
        self,
        true_history: np.ndarray,
        fused_history: np.ndarray,
        node_history: List[List[np.ndarray]]
    ) -> None:
        """Plot the true, fused, and node-level trajectories."""
        cfg = self.config
        time = np.arange(cfg.num_simulation_steps)
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # X position plot
        self._plot_position_component(
            axs[0],
            time,
            true_history,
            fused_history,
            node_history,
            component=0,
            label='X Position',
            var_name='p_x'
        )
        
        # Y position plot
        self._plot_position_component(
            axs[1],
            time,
            true_history,
            fused_history,
            node_history,
            component=1,
            label='Y Position',
            var_name='p_y'
        )
        
        axs[1].set_xlabel('Time step')
        plt.suptitle(f"NDKF Results (2D/{cfg.num_sensor_nodes} Nodes)")
        plt.tight_layout()
        plt.show()
    
    def _plot_position_component(
        self,
        ax: plt.Axes,
        time: np.ndarray,
        true_history: np.ndarray,
        fused_history: np.ndarray,
        node_history: List[List[np.ndarray]],
        component: int,
        label: str,
        var_name: str
    ) -> None:
        """Plot a single position component."""
        ax.plot(time, true_history[:, component], 'k-', label=f'True ${var_name}$')
        
        for i in range(self.config.num_sensor_nodes):
            node_arr = np.array(node_history[i])
            ax.plot(
                time,
                node_arr[:, component],
                '--',
                label=f'Node {i+1} ${var_name}$'
            )
        
        ax.plot(
            time,
            fused_history[:, component],
            'r-',
            linewidth=2,
            label=f'Fused ${var_name}$'
        )
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)
    
    def plot_measurement_residuals(
        self,
        meas_net: MeasurementNet2D,
        node_id: int = 1
    ) -> None:
        """Plot measurement innovation residuals for a specific node."""
        cfg = self.config
        dynamics_model = TrueDynamicsModel(cfg)
        measurement_model = TrueMeasurementModel(cfg)
        
        z_true_list = []
        z_pred_list = []
        x_test = cfg.initial_state.copy()
        
        for k in range(cfg.num_simulation_steps):
            x_test = dynamics_model.evolve(x_test, k)
            z_true = measurement_model.measure(node_id, x_test)
            
            with torch.no_grad():
                x_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(0)
                z_pred = meas_net(x_tensor).item()
            
            z_true_list.append(z_true)
            z_pred_list.append(z_pred)
        
        plt.figure(figsize=(10, 4))
        plt.plot(z_true_list, 'k-', label=f'True Measurement (Node {node_id})')
        plt.plot(z_pred_list, 'r--', label=f'Predicted Measurement (Node {node_id})')
        plt.xlabel('Time step')
        plt.ylabel('Measurement value')
        plt.title(f'Measurement Innovation Residuals for Node {node_id} (NDKF)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ---------------------------
# Main Execution
# ---------------------------
def main(seed: Optional[int] = None):
    """
    Main execution function.
    
    Args:
        seed: Random seed for reproducibility. If None, a random seed is generated.
    """
    # Set random seed for this run
    actual_seed = set_random_seeds(seed)
    print()  # Blank line for readability
    
    # Initialize configuration
    config = SystemConfig()
    
    # Train networks
    trainer = NetworkTrainer(config)
    dyn_net = trainer.train_dynamics_network()
    meas_nets = trainer.train_measurement_networks()
    
    # Set networks to evaluation mode
    dyn_net.eval()
    for net in meas_nets:
        net.eval()
    
    # Create and run simulation
    simulator = NDKFSimulator(config)
    ndkf_filter = simulator.create_filter(dyn_net, meas_nets)
    true_history, fused_history, node_history = simulator.run_simulation(ndkf_filter)
    
    # Visualize results
    visualizer = ResultVisualizer(config)
    visualizer.plot_trajectories(true_history, fused_history, node_history)
    visualizer.plot_measurement_residuals(meas_nets[0], node_id=1)


if __name__ == "__main__":
    main(seed=2886738106)