"""
NDKF Class and Supporting Functions

Authors: Siavash Farzan
Date: 09-30-2025
License: MIT License

Description:
    This file implements the NDKF class and supporting neural network models and training
    functions. The class encapsulates the Neural-Enhanced Distributed Kalman Filter (NDKF)
    algorithm for nonlinear multi-sensor estimation.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

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
    """A small MLP that predicts residual dynamics for a 2D state.

    Input features are [x, y, t] -> residual in R^2.
    """

    def __init__(self, input_dim: int = 3, hidden_size: int = 128, dropout_prob: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def train_residual_dynamics_net(
    net: nn.Module,
    x_train: np.ndarray,
    r_train: np.ndarray,
    epochs: int = 3000,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> nn.Module:
    """Train the residual dynamics network with MSE loss.

    Args:
        net: The dynamics residual network.
        x_train: Training inputs of shape (N, in_dim).
        r_train: Training residual targets of shape (N, 2).
        epochs: Number of epochs.
        lr: Learning rate.
        device: Torch device string or object.
        verbose: If True, print progress every 500 epochs.

    Returns:
        The trained network (in eval() mode).
    """
    net = net.to(device)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    loss_fn = nn.MSELoss()

    x_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    r_tensor = torch.as_tensor(r_train, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        r_pred = net(x_tensor)
        loss = loss_fn(r_pred, r_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if verbose and (epoch + 1) % 500 == 0:
            print(f"Residual Dynamics Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    net.eval()
    return net


# ---------------------------
# Measurement Network
# ---------------------------
class MeasurementNet2D(nn.Module):
    """A small MLP that maps a 2D state to a scalar measurement."""
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_measurement_net(
    net: nn.Module,
    x_train: np.ndarray,
    z_train: np.ndarray,
    epochs: int = 1000,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> nn.Module:
    """Train a measurement network with MSE loss.

    Args:
        net: The measurement network.
        x_train: Training inputs of shape (N, 2).
        z_train: Training targets of shape (N, 1) or (N,).
        epochs: Number of epochs.
        lr: Learning rate.
        device: Torch device string or object.
        verbose: If True, print progress every 250 epochs.

    Returns:
        The trained network (in eval() mode).
    """
    net = net.to(device)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    z_tensor = torch.as_tensor(z_train, dtype=torch.float32, device=device).view(-1, 1)

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        z_pred = net(x_tensor)
        loss = loss_fn(z_pred, z_tensor)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 250 == 0:
            print(f"Measurement Net Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    net.eval()
    return net


# ---------------------------
# NDKF Class
# ---------------------------
class NDKF:
    """Neural-Enhanced Distributed Kalman Filter for 2D states and scalar measurements.

    This implementation keeps the core method unchanged and tightens typing, shapes,
    and numerical robustness.
    """

    def __init__(
        self,
        dyn_net: nn.Module,
        meas_nets: Sequence[nn.Module],
        initial_state: np.ndarray,
        initial_cov: np.ndarray,
        Qs: Sequence[np.ndarray],
        Rs: Sequence[np.ndarray],
        eps_reg: float = 1e-9,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the NDKF.

        Args:
            dyn_net: Neural network for residual dynamics (expects [x, y, t] -> R^2).
            meas_nets: List of measurement networks (one per node), each 2D state -> scalar.
            initial_state: Initial state vector (shape (2,)).
            initial_cov: Initial covariance matrix (2x2).
            Qs: List of process noise covariance matrices for each node (2x2).
            Rs: List of measurement noise covariance matrices for each node (1x1).
            eps_reg: Small diagonal regularizer added before matrix inversions.
            device: Torch device for forward passes.
        """
        self.dyn_net = dyn_net.to(device).eval()
        self.meas_nets = [m.to(device).eval() for m in meas_nets]
        self.device = torch.device(device)
        self.eps_reg = float(eps_reg)

        self.num_nodes = len(meas_nets)
        self.state_dim = int(initial_state.size)
        self.node_estimates: List[np.ndarray] = [np.asarray(initial_state, dtype=np.float64).copy() for _ in range(self.num_nodes)]
        self.node_covariances: List[np.ndarray] = [np.asarray(initial_cov, dtype=np.float64).copy() for _ in range(self.num_nodes)]
        self.Qs = [np.asarray(Q, dtype=np.float64) for Q in Qs]
        self.Rs = [np.asarray(R, dtype=np.float64) for R in Rs]

        # Cached identity for the update step
        self._I = np.eye(self.state_dim, dtype=np.float64)

    @staticmethod
    def compute_jacobian(net: nn.Module, x_est: np.ndarray, device: torch.device | str = "cpu") -> np.ndarray:
        """Compute the measurement Jacobian H at x_est (shape 1 x n)."""
        net.eval()
        x_torch = torch.tensor(x_est, dtype=torch.float32, device=device, requires_grad=True)

        def single_input_forward(x_input: torch.Tensor) -> torch.Tensor:
            return net(x_input.unsqueeze(0)).squeeze(0)

        J = AF.jacobian(single_input_forward, x_torch)
        # For a scalar measurement, J has shape (n,); reshape to (1, n)
        return J.detach().cpu().numpy().reshape(1, -1)

    def predict(self, u: np.ndarray, t: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Prediction step for each node.

        Args:
            u: Nominal control input (shape (2,)).
            t: Time parameter fed to the residual dynamics net.
        """
        for i in range(self.num_nodes):
            dyn_input = np.concatenate((self.node_estimates[i], [t])).astype(np.float32, copy=False)
            inp = torch.from_numpy(dyn_input).to(self.device).unsqueeze(0)
            with torch.no_grad():
                residual = self.dyn_net(inp).squeeze(0).detach().cpu().numpy().astype(np.float64)
            x_pred = self.node_estimates[i] + np.asarray(u, dtype=np.float64) + residual
            P_pred = self.node_covariances[i] + self.Qs[i]
            self.node_estimates[i] = x_pred
            self.node_covariances[i] = P_pred
        return self.node_estimates, self.node_covariances

    def update(self, measurements: Sequence[float | np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Update step for each node using its local measurement."""
        for i in range(self.num_nodes):
            x_pred = self.node_estimates[i]
            P_pred = self.node_covariances[i]

            # Predicted measurement
            with torch.no_grad():
                z_pred_t = self.meas_nets[i](torch.as_tensor(x_pred, dtype=torch.float32, device=self.device).unsqueeze(0))
                z_pred = float(z_pred_t.squeeze(0).detach().cpu().numpy())

            # Linearize
            H = self.compute_jacobian(self.meas_nets[i], x_pred, device=self.device)  # (1, n)

            # Innovation
            z_meas = float(np.asarray(measurements[i]).reshape(()))  # scalar
            y_tilde = z_meas - z_pred  # scalar

            # Kalman gain
            S = H @ P_pred @ H.T + self.Rs[i]  # (1, 1)
            # Regularize S to avoid numerical issues
            S = S + self.eps_reg * np.eye(1)
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)

            K = (P_pred @ H.T) @ S_inv  # (n, 1)

            # State and covariance update (standard form; keep method unchanged)
            x_upd = x_pred + (K * y_tilde).reshape(-1)
            P_upd = (self._I - K @ H) @ P_pred

            # Symmetrize to counter round-off
            P_upd = 0.5 * (P_upd + P_upd.T)

            self.node_estimates[i] = x_upd
            self.node_covariances[i] = P_upd

        return self.node_estimates, self.node_covariances

    def fuse(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fuse node estimates with information-weighted strategy."""
        n = self.state_dim
        P_inv_sum = np.zeros((n, n), dtype=np.float64)
        weighted_state_sum = np.zeros(n, dtype=np.float64)

        for i in range(self.num_nodes):
            P_i = self.node_covariances[i] + self.eps_reg * self._I  # slight regularization
            try:
                P_inv = np.linalg.inv(P_i)
            except np.linalg.LinAlgError:
                P_inv = np.linalg.pinv(P_i)
            P_inv_sum += P_inv
            weighted_state_sum += P_inv @ self.node_estimates[i]

        try:
            P_fused = np.linalg.inv(P_inv_sum)
        except np.linalg.LinAlgError:
            P_fused = np.linalg.pinv(P_inv_sum)

        x_fused = P_fused @ weighted_state_sum
        return x_fused, P_fused

    def step(
        self,
        k: int,
        u: np.ndarray,
        t: float,
        measurements: Sequence[float | np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Run one filter step (predict, update, and fuse).

        Args:
            k: Current time step (kept for interface consistency).
            u: Nominal control input (shape (2,)).
            t: Time parameter fed to the residual dynamics net.
            measurements: Iterable of scalar measurements, one per node.

        Returns:
            x_fused: The fused state estimate.
            node_estimates: List of node state estimates after update.
            node_covariances: List of node covariance matrices after update.
        """
        # Note: k is currently unused but preserved to avoid API changes.
        self.predict(u, t)
        self.update(measurements)
        x_fused, _ = self.fuse()
        return x_fused, self.node_estimates, self.node_covariances
