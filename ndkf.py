"""
NDKF: A Neural-Enhanced Distributed Kalman Filter for Nonlinear Multi-Sensor Estimation

Authors: Siavash Farzan
Date: 02-03-2025
License: MIT License

Description:
    This file implements the NDKF algorithm for nonlinear multi-sensor state estimation as described
    in the paper "NDKF: A Neural-Enhanced Distributed Kalman Filter for Nonlinear Multi-Sensor Estimation."
    The algorithm leverages neural networks to learn both the residual dynamics and measurement models 
    directly from data. It uses a consensus-based fusion process to combine local estimates from multiple 
    sensor nodes into a global state estimate.

Usage:
    Run this file to train the neural network models and execute the NDKF simulation:
        python ndkf.py

References:
    - Paper: "NDKF: A Neural-Enhanced Distributed Kalman Filter for Nonlinear Multi-Sensor Estimation"
    - GitHub Repository: https://github.com/sfarzan/NDKF
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import functional as AF
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ============================================================
# 1. True 2D System and Measurement Models
# ============================================================
def true_system_dynamics(x, k):
    """
    Modified true 2D system dynamics with additional nonlinearities:
      p_x[k+1] = p_x[k] + 0.05*cos(k/10) + 0.02*sin(3*p_y[k]) + process_noise_x
      p_y[k+1] = p_y[k] + 0.05*sin(k/10) + 0.02*cos(2*p_x[k]) + process_noise_y
    """
    vx = 0.05 * np.cos(k / 10.0) + 0.02 * np.sin(3 * x[1])
    vy = 0.05 * np.sin(k / 10.0) + 0.02 * np.cos(2 * x[0])
    process_noise = np.random.multivariate_normal(mean=[0, 0], cov=np.diag([0.002, 0.002]))
    return x + np.array([vx, vy]) + process_noise

def true_measurement_nonlinear(node_id, x):
    """
    Modified true nonlinear measurement functions:
      Node 1: z = sin(2*p_x) + 0.5*p_y + noise
      Node 2: z = cos(2*p_y) - 0.4*p_x + noise
      Node 3: z = sin(2*p_x) + cos(2*p_y) + noise
      Node 4: z = sin(2*p_x) - cos(2*p_y) + noise
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

def true_dynamics_jacobian(x, k):
    """
    Returns the Jacobian of the dynamics.
    Here we assume it is the identity matrix.
    """
    return np.eye(2)

# ============================================================
# 2. Training Data Generation for Dynamics Residual (NDKF)
# ============================================================
def generate_residual_training_data_2d(num_steps=1000):
    """
    Generate training pairs for the dynamics residual network.
    Input: [p_x, p_y, t] where t = k/num_steps.
    Target: residual = x_{k+1} - (x_k + u_k), with u_k = [0.05*cos(k/10), 0.05*sin(k/10)].
    """
    x_vals = []
    r_vals = []
    x = np.array([0.0, 0.0], dtype=np.float32)
    for k in range(num_steps):
        t = k / num_steps
        u = np.array([0.05 * np.cos(k/10.0), 0.05 * np.sin(k/10.0)])
        x_next = true_system_dynamics(x, k)
        residual = x_next - (x + u)
        x_vals.append(np.concatenate((x, [t])))
        r_vals.append(residual)
        x = x_next.astype(np.float32)
    return np.array(x_vals, dtype=np.float32), np.array(r_vals, dtype=np.float32)

def generate_measurement_training_data_2d(node_id, num_steps=1000):
    """
    Generate training data (x, z) for each measurement network.
    """
    x_vals = []
    z_vals = []
    x = np.array([0.0, 0.0], dtype=np.float32)
    for k in range(num_steps):
        x_vals.append(x.copy())
        z = true_measurement_nonlinear(node_id, x)
        z_vals.append([z])
        x = true_system_dynamics(x, k)
    return np.array(x_vals, dtype=np.float32), np.array(z_vals, dtype=np.float32)

# ============================================================
# 3. Dynamics Residual Network (NDKF)
# ============================================================
class DynamicsResidualNet2D(nn.Module):
    def __init__(self, input_dim=3, hidden_size=128, dropout_prob=0.2):
        super(DynamicsResidualNet2D, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

def train_residual_dynamics_net(net, x_train, r_train, epochs=3000, lr=0.001):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    loss_fn = nn.MSELoss()
    x_tensor = torch.from_numpy(x_train)
    r_tensor = torch.from_numpy(r_train)
    for epoch in range(epochs):
        optimizer.zero_grad()
        r_pred = net(x_tensor)
        loss = loss_fn(r_pred, r_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch+1) % 500 == 0:
            print(f"Residual Dynamics Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    return net

# ============================================================
# 4. Measurement Network
# ============================================================
class MeasurementNet2D(nn.Module):
    def __init__(self, hidden_size=32):
        super(MeasurementNet2D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def train_measurement_net(net, x_train, z_train, epochs=1000, lr=0.001):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x_tensor = torch.from_numpy(x_train)
    z_tensor = torch.from_numpy(z_train)
    for epoch in range(epochs):
        optimizer.zero_grad()
        z_pred = net(x_tensor)
        loss = loss_fn(z_pred, z_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 250 == 0:
            print(f"Measurement Net Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    return net

def compute_jacobian(net, x_est):
    """
    Compute the Jacobian of net(x_est) with respect to x_est.
    """
    net.eval()  # Ensure evaluation mode for batch norm
    x_torch = torch.tensor(x_est, dtype=torch.float32, requires_grad=True)
    def single_input_forward(x_input):
        return net(x_input.unsqueeze(0)).squeeze(0)
    J = AF.jacobian(single_input_forward, x_torch)
    return J.detach().numpy()

# ============================================================
# 5. NDKF Simulation (using residual dynamics network)
# ============================================================
def simulate_distributed_kf_2d(dyn_net, meas_nets, num_steps=100):
    Qs = [np.diag([0.005, 0.005])]*4
    Rs = [np.array([[0.01]]),
          np.array([[0.01]]),
          np.array([[0.01]]),
          np.array([[0.01]])]
    
    x_true = np.array([0.0, 0.0])
    node_estimates = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(4)]
    node_covariances = [np.eye(2, dtype=np.float32)*0.5 for _ in range(4)]
    
    x_true_history = []
    node_histories = [[] for _ in range(4)]
    fused_history = []
    meas_pred_history = [[] for _ in range(4)]
    
    for k in range(num_steps):
        x_true_next = true_system_dynamics(x_true, k)
        t = k / num_steps
        u = np.array([0.05 * np.cos(k/10.0), 0.05 * np.sin(k/10.0)])
        for i in range(4):
            # Residual dynamics: input = [state, t]
            dyn_input = np.concatenate((node_estimates[i], [t]))
            inp = torch.tensor(dyn_input, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                residual = dyn_net(inp).squeeze(0).numpy()
            x_pred = node_estimates[i] + u + residual
            # Covariance prediction update
            P_pred = node_covariances[i] + Qs[i]
            
            z_i = true_measurement_nonlinear(i+1, x_true_next)
            with torch.no_grad():
                z_pred = meas_nets[i](torch.tensor(x_pred, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            H = compute_jacobian(meas_nets[i], x_pred)
            S = H @ P_pred @ H.T + Rs[i]
            K = P_pred @ H.T @ np.linalg.inv(S)
            y_tilde = z_i - z_pred
            x_upd = x_pred + (K * y_tilde).squeeze()
            P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
            
            node_estimates[i] = x_upd
            node_covariances[i] = P_upd
            meas_pred_history[i].append(z_pred)
        # Information-weighted fusion
        P_inv_sum = np.zeros((2,2), dtype=np.float32)
        weighted_state_sum = np.zeros(2, dtype=np.float32)
        for i in range(4):
            P_inv = np.linalg.inv(node_covariances[i])
            P_inv_sum += P_inv
            weighted_state_sum += P_inv @ node_estimates[i]
        P_fused = np.linalg.inv(P_inv_sum)
        x_fused = P_fused @ weighted_state_sum
        
        x_true_history.append(x_true_next.copy())
        for i in range(4):
            node_histories[i].append(node_estimates[i].copy())
        fused_history.append(x_fused.copy())
        
        x_true = x_true_next.copy()
    return {
        'x_true': np.array(x_true_history),
        'node_estimates': [np.array(h) for h in node_histories],
        'fused_estimates': np.array(fused_history),
        'meas_predictions': [np.array(h) for h in meas_pred_history]
    }

# ============================================================
# 6. Main Script: Training and NDKF Simulation
# ============================================================
if __name__ == "__main__":
    num_steps = 100

    # Train the dynamics residual network
    print("Training residual dynamics network...")
    x_train_res, r_train = generate_residual_training_data_2d(num_steps=1000)
    dyn_net = DynamicsResidualNet2D(input_dim=3, hidden_size=128, dropout_prob=0.2)
    dyn_net = train_residual_dynamics_net(dyn_net, x_train_res, r_train, epochs=3000, lr=0.001)
    
    # Train the measurement networks for each node
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
    
    print("Running NDKF simulation...")
    results = simulate_distributed_kf_2d(dyn_net, meas_nets, num_steps=num_steps)
    method_label = "NDKF"

    # Plot 1: Trajectory Plot
    x_true = results['x_true']
    fused_est = results['fused_estimates']
    time = np.arange(x_true.shape[0])
    fig, axs = plt.subplots(2, 1, figsize=(10,8), sharex=True)
    axs[0].plot(time, x_true[:,0], 'k-', label='True $p_x$')
    for i in range(4):
        axs[0].plot(time, results['node_estimates'][i][:,0], '--', label=f'Node {i+1} $p_x$')
    axs[0].plot(time, fused_est[:,0], 'r-', linewidth=2, label='Fused $p_x$')
    axs[0].set_ylabel('X Position')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time, x_true[:,1], 'k-', label='True $p_y$')
    for i in range(4):
        axs[1].plot(time, results['node_estimates'][i][:,1], '--', label=f'Node {i+1} $p_y$')
    axs[1].plot(time, fused_est[:,1], 'r-', linewidth=2, label='Fused $p_y$')
    axs[1].set_ylabel('Y Position')
    axs[1].set_xlabel('Time step')
    axs[1].legend()
    axs[1].grid(True)
    plt.suptitle(f"{method_label} Results (2D/4 Nodes)")
    plt.savefig('trajectory_plot.eps', format='eps')
    plt.show()

    # Plot 2: Measurement Innovation Residuals for Node 1
    test_steps = 100
    z_true_list = []
    z_pred_list = []
    x = np.array([0.0, 0.0], dtype=np.float32)
    for k in range(test_steps):
        x = true_system_dynamics(x, k)
        z_true = true_measurement_nonlinear(1, x)
        with torch.no_grad():
            z_pred = meas_nets[0](torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
        z_true_list.append(z_true)
        z_pred_list.append(z_pred)
    plt.figure(figsize=(10,4))
    plt.plot(z_true_list, 'k-', label='True Measurement (Node 1)')
    plt.plot(z_pred_list, 'r--', label='Predicted Measurement (Node 1)')
    plt.xlabel('Time step')
    plt.ylabel('Measurement value')
    plt.title('Measurement Innovation Residuals for Node 1 (NDKF)')
    plt.legend()
    plt.grid(True)
    plt.savefig('innovation_residuals_plot.eps', format='eps')
    plt.show()
