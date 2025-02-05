"""
NDKF Class and Supporting Functions

Authors: Siavash Farzan
Date: 02-03-2025
License: MIT License

Description:
    This file implements the NDKF class and supporting neural network models and training
    functions. The class encapsulates the Neural-Enhanced Distributed Kalman Filter (NDKF)
    algorithm for nonlinear multi-sensor estimation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import functional as AF

# ---------------------------
# Dynamics Residual Network
# ---------------------------
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
        if (epoch + 1) % 500 == 0:
            print(f"Residual Dynamics Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    return net

# ---------------------------
# Measurement Network
# ---------------------------
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
        if (epoch + 1) % 250 == 0:
            print(f"Measurement Net Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    return net

# ---------------------------
# NDKF Class
# ---------------------------
class NDKF:
    def __init__(self, dyn_net, meas_nets, initial_state, initial_cov, Qs, Rs):
        """
        Initializes the NDKF filter.
        
        Args:
            dyn_net: Neural network for dynamics residual.
            meas_nets: List of measurement networks (one per node).
            initial_state: Initial state vector (numpy array).
            initial_cov: Initial covariance matrix.
            Qs: List of process noise covariance matrices for each node.
            Rs: List of measurement noise covariance matrices for each node.
        """
        self.dyn_net = dyn_net
        self.meas_nets = meas_nets
        self.num_nodes = len(meas_nets)
        self.node_estimates = [initial_state.copy() for _ in range(self.num_nodes)]
        self.node_covariances = [initial_cov.copy() for _ in range(self.num_nodes)]
        self.Qs = Qs
        self.Rs = Rs

    @staticmethod
    def compute_jacobian(net, x_est):
        """
        Computes the Jacobian of net at x_est.
        """
        net.eval()
        x_torch = torch.tensor(x_est, dtype=torch.float32, requires_grad=True)
        def single_input_forward(x_input):
            return net(x_input.unsqueeze(0)).squeeze(0)
        J = AF.jacobian(single_input_forward, x_torch)
        return J.detach().numpy().reshape(1, -1)

    def predict(self, u, t):
        """
        Prediction step for each node.
        """
        for i in range(self.num_nodes):
            dyn_input = np.concatenate((self.node_estimates[i], [t]))
            inp = torch.tensor(dyn_input, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                residual = self.dyn_net(inp).squeeze(0).numpy()
            x_pred = self.node_estimates[i] + u + residual
            P_pred = self.node_covariances[i] + self.Qs[i]
            self.node_estimates[i] = x_pred
            self.node_covariances[i] = P_pred
        return self.node_estimates, self.node_covariances

    def update(self, measurements):
        """
        Update step for each node using its measurement.
        """
        for i in range(self.num_nodes):
            x_pred = self.node_estimates[i]
            P_pred = self.node_covariances[i]
            with torch.no_grad():
                z_pred = self.meas_nets[i](torch.tensor(x_pred, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            H = self.compute_jacobian(self.meas_nets[i], x_pred)
            S = H @ P_pred @ H.T + self.Rs[i]
            K = P_pred @ H.T @ np.linalg.inv(S)
            y_tilde = measurements[i] - z_pred
            x_upd = x_pred + (K * y_tilde).squeeze()
            P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
            self.node_estimates[i] = x_upd
            self.node_covariances[i] = P_upd
        return self.node_estimates, self.node_covariances

    def fuse(self):
        """
        Fuses the node estimates with an information-weighted strategy.
        """
        P_inv_sum = np.zeros((len(self.node_estimates[0]), len(self.node_estimates[0])))
        weighted_state_sum = np.zeros(len(self.node_estimates[0]))
        for i in range(self.num_nodes):
            P_inv = np.linalg.inv(self.node_covariances[i])
            P_inv_sum += P_inv
            weighted_state_sum += P_inv @ self.node_estimates[i]
        P_fused = np.linalg.inv(P_inv_sum)
        x_fused = P_fused @ weighted_state_sum
        return x_fused, P_fused

    def step(self, k, u, t, measurements):
        """
        Runs one filter step (predict, update, and fuse).
        
        Args:
            k: Current time step.
            u: Nominal control input.
            t: Time parameter.
            measurements: List of measurements from each node.
            
        Returns:
            x_fused: The fused state estimate.
        """
        self.predict(u, t)
        self.update(measurements)
        x_fused, _ = self.fuse()
        return x_fused, self.node_estimates, self.node_covariances