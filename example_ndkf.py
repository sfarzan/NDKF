"""
Example Usage of the NDKF Class

Authors: Siavash Farzan
Date: 02-03-2025
License: MIT License

Description:
    This script demonstrates the use of the NDKF class on a simulated 2D system
    monitored by four sensor nodes. In addition to the fused state trajectory,
    it plots the measurement innovation residuals for Node 1.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import the NDKF class and network functions from ndkf.py
from ndkf import (
    NDKF,
    DynamicsResidualNet2D,
    MeasurementNet2D,
    train_residual_dynamics_net,
    train_measurement_net
)

# ---------------------------
# Define True System and Measurement Models
# ---------------------------
def true_system_dynamics(x, k):
    """
    2D system dynamics.
    """
    vx = 0.05 * np.cos(k / 10.0) + 0.02 * np.sin(3 * x[1])
    vy = 0.05 * np.sin(k / 10.0) + 0.02 * np.cos(2 * x[0])
    process_noise = np.random.multivariate_normal(mean=[0, 0], cov=np.diag([0.002, 0.002]))
    return x + np.array([vx, vy]) + process_noise

def true_measurement_nonlinear(node_id, x):
    """
    Nonlinear measurement model.
    """
    px, py = x
    noise = np.random.normal(0, 0.1)
    if node_id == 1:
        return np.sin(2 * px) + 0.5 * py + noise
    elif node_id == 2:
        return np.cos(2 * py) - 0.4 * px + noise
    elif node_id == 3:
        return np.sin(2 * px) + np.cos(2 * py) + noise
    else:
        return np.sin(2 * px) - np.cos(2 * py) + noise

# ---------------------------
# Data Generation Functions
# ---------------------------
def generate_residual_training_data_2d(num_steps=1000):
    x_vals = []
    r_vals = []
    x = np.array([0.0, 0.0], dtype=np.float32)
    for k in range(num_steps):
        t = k / num_steps
        u = np.array([0.05 * np.cos(k / 10.0), 0.05 * np.sin(k / 10.0)])
        x_next = true_system_dynamics(x, k)
        residual = x_next - (x + u)
        x_vals.append(np.concatenate((x, [t])))
        r_vals.append(residual)
        x = x_next.astype(np.float32)
    return np.array(x_vals, dtype=np.float32), np.array(r_vals, dtype=np.float32)

def generate_measurement_training_data_2d(node_id, num_steps=1000):
    x_vals = []
    z_vals = []
    x = np.array([0.0, 0.0], dtype=np.float32)
    for k in range(num_steps):
        x_vals.append(x.copy())
        z = true_measurement_nonlinear(node_id, x)
        z_vals.append([z])
        x = true_system_dynamics(x, k)
    return np.array(x_vals, dtype=np.float32), np.array(z_vals, dtype=np.float32)

# ---------------------------
# Main Simulation Example
# ---------------------------
def main():   
    # Train the dynamics residual network
    print("Training dynamics residual network...")
    x_train_res, r_train = generate_residual_training_data_2d(num_steps=1000)
    dyn_net = DynamicsResidualNet2D(input_dim=3, hidden_size=128, dropout_prob=0.2)
    dyn_net = train_residual_dynamics_net(dyn_net, x_train_res, r_train, epochs=3000, lr=0.001)
    
    # Train measurement networks for each sensor node
    meas_nets = []
    for node_id in range(1, 5):
        print(f"Training measurement network for Node {node_id}...")
        x_train_meas, z_train_meas = generate_measurement_training_data_2d(node_id, num_steps=1000)
        net_meas = MeasurementNet2D(hidden_size=32)
        net_meas = train_measurement_net(net_meas, x_train_meas, z_train_meas, epochs=1000, lr=0.001)
        meas_nets.append(net_meas)
    
    # Set networks to evaluation mode
    dyn_net.eval()
    for net in meas_nets:
        net.eval()
    
    # Create an NDKF instance
    initial_state = np.array([0.0, 0.0], dtype=np.float32)
    initial_cov = 0.5 * np.eye(2, dtype=np.float32)
    Qs = [np.diag([0.005, 0.005]) for _ in range(4)]
    Rs = [np.array([[0.01]]) for _ in range(4)]
    ndkf_filter = NDKF(dyn_net, meas_nets, initial_state, initial_cov, Qs, Rs)
    
    # Simulation loop for the trajectory
    num_steps = 100
    x_true = np.array([0.0, 0.0])
    true_history = []
    fused_history = []
    node_history = [[] for _ in range(4)]
    
    for k in range(num_steps):
        # Evolve the true system state
        x_true = true_system_dynamics(x_true, k)
        true_history.append(x_true.copy())
        
        # Compute the nominal control input and time parameter
        u = np.array([0.05 * np.cos(k / 10.0), 0.05 * np.sin(k / 10.0)])
        t = k / num_steps
        
        # Generate measurements for each sensor node
        measurements = []
        for node_id in range(1, 5):
            z = true_measurement_nonlinear(node_id, x_true)
            measurements.append(z)
        
        # Perform one filter step
        x_fused, node_estimates, _ = ndkf_filter.step(k, u, t, measurements)
        fused_history.append(x_fused.copy())
        for i in range(4):
            node_history[i].append(node_estimates[i].copy())
    
    true_history = np.array(true_history)
    fused_history = np.array(fused_history)
    
    # ---------------------------
    # Plot the Trajectory Results
    # ---------------------------
    time = np.arange(num_steps)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # X position plot
    axs[0].plot(time, true_history[:, 0], 'k-', label='True $p_x$')
    for i in range(4):
        node_arr = np.array(node_history[i])
        axs[0].plot(time, node_arr[:, 0], '--', label=f'Node {i+1} $p_x$')
    axs[0].plot(time, fused_history[:, 0], 'r-', linewidth=2, label='Fused $p_x$')
    axs[0].set_ylabel('X Position')
    axs[0].legend()
    axs[0].grid(True)
    
    # Y position plot
    axs[1].plot(time, true_history[:, 1], 'k-', label='True $p_y$')
    for i in range(4):
        node_arr = np.array(node_history[i])
        axs[1].plot(time, node_arr[:, 1], '--', label=f'Node {i+1} $p_y$')
    axs[1].plot(time, fused_history[:, 1], 'r-', linewidth=2, label='Fused $p_y$')
    axs[1].set_ylabel('Y Position')
    axs[1].set_xlabel('Time step')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.suptitle("NDKF Results (2D/4 Nodes)")
    plt.show()
    
    # ---------------------------
    # Plot Measurement Innovation Residuals for Node 1
    # ---------------------------
    test_steps = 100
    z_true_list = []
    z_pred_list = []
    x_test = np.array([0.0, 0.0], dtype=np.float32)
    
    for k in range(test_steps):
        x_test = true_system_dynamics(x_test, k)
        z_true = true_measurement_nonlinear(1, x_test)
        with torch.no_grad():
            z_pred = meas_nets[0](torch.tensor(x_test, dtype=torch.float32).unsqueeze(0)).item()
        z_true_list.append(z_true)
        z_pred_list.append(z_pred)
    
    plt.figure(figsize=(10, 4))
    plt.plot(z_true_list, 'k-', label='True Measurement (Node 1)')
    plt.plot(z_pred_list, 'r--', label='Predicted Measurement (Node 1)')
    plt.xlabel('Time step')
    plt.ylabel('Measurement value')
    plt.title('Measurement Innovation Residuals for Node 1 (NDKF)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()