"""
Tier 3: Robust Statistics & Physics-Informed Loss Tests
Tests smooth Cauchy loss, Huber loss, kinematic acceleration penalty,
boundedness of influence functions vs MSE, and outlier robustness.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn


# --- Reference Loss Implementations (Specification Oracle) ---

class CauchyLoss(nn.Module):
    """
    Smooth Cauchy Loss (Lorentzian M-estimator):
    L(e; gamma) = (gamma^2 / 2) * ln(1 + (e / gamma)^2)
    Gradient: psi(e) = e / (1 + (e / gamma)^2)
    Bounded supremum: sup |psi(e)| = gamma / 2 at |e| = gamma.
    """
    def __init__(self, gamma: float = 5.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        e = pred - target
        loss = (self.gamma**2 / 2.0) * torch.log(1.0 + (e / self.gamma)**2)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class PhysicsKinematicLoss(nn.Module):
    """
    Penalizes non-physical vehicle acceleration exceeding max_accel (default 30 (km/h)/s).
    L_phys = mean(max(0, |a| - max_accel)^2)
    """
    def __init__(self, max_accel: float = 30.0, dt: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.max_accel = float(max_accel)
        self.dt = float(dt)
        self.reduction = reduction
        
    def forward(self, pred_speeds: torch.Tensor) -> torch.Tensor:
        # pred_speeds: (B, T) or (B, 1) or sequential speed estimates
        if pred_speeds.ndim < 2 or pred_speeds.shape[-1] <= 1:
            return torch.tensor(0.0, device=pred_speeds.device, dtype=pred_speeds.dtype)
        
        accel = torch.diff(pred_speeds, dim=-1) / self.dt
        excess = torch.clamp(torch.abs(accel) - self.max_accel, min=0.0)
        penalty = excess**2
        if self.reduction == "mean":
            return penalty.mean()
        elif self.reduction == "sum":
            return penalty.sum()
        return penalty


class CombinedPhysicsLoss(nn.Module):
    """Combined robust regression and kinematic acceleration loss."""
    def __init__(self, gamma: float = 5.0, physics_weight: float = 0.1, max_accel: float = 30.0):
        super().__init__()
        self.cauchy = CauchyLoss(gamma=gamma)
        self.physics = PhysicsKinematicLoss(max_accel=max_accel)
        self.physics_weight = float(physics_weight)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, speed_seq: torch.Tensor = None) -> torch.Tensor:
        reg_loss = self.cauchy(pred, target)
        if speed_seq is not None and speed_seq.shape[-1] > 1:
            phys_loss = self.physics(speed_seq)
            return reg_loss + self.physics_weight * phys_loss
        return reg_loss


# --- Unit Tests ---

def test_cauchy_loss_zero_at_origin():
    """Verifies that Cauchy loss is exactly 0 when prediction equals target."""
    criterion = CauchyLoss(gamma=5.0)
    target = torch.tensor([50.0, 75.0, 100.0])
    pred = torch.tensor([50.0, 75.0, 100.0])
    loss = criterion(pred, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_cauchy_loss_symmetry():
    """Verifies that Cauchy loss is symmetric: L(y + e, y) == L(y - e, y)."""
    criterion = CauchyLoss(gamma=5.0)
    y = torch.tensor([60.0])
    e = torch.tensor([12.5])
    
    loss_pos = criterion(y + e, y)
    loss_neg = criterion(y - e, y)
    assert torch.isclose(loss_pos, loss_neg, atol=1e-6)


def test_cauchy_loss_taylor_mse_convergence():
    """
    Verifies that for small errors (|e| << gamma), Cauchy loss converges
    to 0.5 * e^2 (matching MSE behavior for near-zero residuals).
    """
    gamma = 10.0
    criterion = CauchyLoss(gamma=gamma)
    
    small_e = torch.tensor([0.05], dtype=torch.float64)
    y = torch.tensor([50.0], dtype=torch.float64)
    
    cauchy_val = criterion(y + small_e, y)
    expected_mse_half = 0.5 * (small_e**2)
    rel_error = torch.abs(cauchy_val - expected_mse_half) / expected_mse_half
    assert rel_error < 1e-4, f"Cauchy did not match 0.5*e^2 at small e: rel error {rel_error.item()}"


def test_cauchy_gradient_supremum_bound():
    """
    Mathematical Proof Verification (Feature 9):
    Verifies that Cauchy loss gradient is strictly bounded by gamma / 2,
    achieved exactly at |e| = gamma.
    """
    gamma = 6.0
    expected_sup = gamma / 2.0  # 3.0
    
    # Evaluate gradient exactly at e = gamma
    e_at_gamma = torch.tensor([gamma], requires_grad=True)
    loss_at_gamma = (gamma**2 / 2.0) * torch.log(1.0 + (e_at_gamma / gamma)**2)
    loss_at_gamma.backward()
    grad_at_gamma = e_at_gamma.grad.item()
    
    assert np.isclose(grad_at_gamma, expected_sup, atol=1e-5), \
        f"Gradient at e=gamma should equal gamma/2 = {expected_sup}, got {grad_at_gamma}"
    
    # Evaluate at multiple points to confirm it never exceeds gamma/2
    test_errors = torch.linspace(-50.0, 50.0, 200, requires_grad=True)
    loss = (gamma**2 / 2.0) * torch.log(1.0 + (test_errors / gamma)**2)
    loss.sum().backward()
    max_observed_grad = test_errors.grad.abs().max().item()
    assert max_observed_grad <= expected_sup + 1e-5, \
        f"Gradient exceeded supremum {expected_sup}: max was {max_observed_grad}"


def test_cauchy_redescending_outlier_immunity_vs_mse():
    """
    Verifies that under an extreme acoustic outlier (e.g. e = 500 km/h),
    Cauchy gradient approaches zero (redescending M-estimator),
    whereas MSE gradient explodes to 500.
    """
    gamma = 5.0
    outlier_e = torch.tensor([500.0], requires_grad=True)
    
    # Cauchy loss
    loss_cauchy = (gamma**2 / 2.0) * torch.log(1.0 + (outlier_e / gamma)**2)
    loss_cauchy.backward()
    cauchy_grad = outlier_e.grad.item()
    
    # MSE loss: L = 0.5 * e^2 => grad = e = 500
    mse_grad = outlier_e.item()
    
    # Cauchy gradient should be small (< 0.1) and far below MSE gradient (500)
    assert abs(cauchy_grad) < 0.1, f"Cauchy gradient {cauchy_grad} should be near zero for extreme outlier"
    assert abs(cauchy_grad) < 0.001 * mse_grad, "Cauchy gradient should be vastly smaller than MSE"


def test_huber_loss_bounded_gradients():
    """Verifies that PyTorch Smooth L1 / Huber loss has Lipschitz gradient bounded by delta."""
    delta = 2.0
    huber = nn.HuberLoss(delta=delta)
    
    e = torch.tensor([100.0], requires_grad=True)
    target = torch.tensor([0.0])
    loss = huber(e, target)
    loss.backward()
    
    assert np.isclose(e.grad.item(), delta, atol=1e-5), \
        f"Huber gradient for large error should equal delta={delta}, got {e.grad.item()}"


def test_physics_acceleration_penalty_zero_for_valid_kinematics():
    """
    Verifies that when speed variation is within physical bounds (|a| <= 30 (km/h)/s),
    the physics kinematic penalty is exactly 0.0.
    """
    phys_loss = PhysicsKinematicLoss(max_accel=30.0, dt=1.0)
    # Speed trajectory: 50 -> 65 -> 80 -> 70 -> 55 (max step is 15 km/h <= 30)
    valid_speeds = torch.tensor([[50.0, 65.0, 80.0, 70.0, 55.0]])
    penalty = phys_loss(valid_speeds)
    assert penalty.item() == 0.0, f"Expected 0 penalty for physically valid speeds, got {penalty.item()}"


def test_physics_acceleration_penalty_active_for_discontinuities():
    """
    Verifies that when speed jumps unphysically (|a| > 30 (km/h)/s),
    the penalty is strictly positive and proportional to (a - 30)^2.
    """
    phys_loss = PhysicsKinematicLoss(max_accel=30.0, dt=1.0)
    # Speed jump: 50 -> 100 (|a| = 50 > 30)
    unphysical_speeds = torch.tensor([[50.0, 100.0]])
    penalty = phys_loss(unphysical_speeds)
    
    expected_penalty = (50.0 - 30.0)**2  # 20^2 = 400.0
    assert np.isclose(penalty.item(), expected_penalty, atol=1e-5), \
        f"Expected penalty {expected_penalty}, got {penalty.item()}"


def test_combined_physics_loss_robustness():
    """
    Verifies end-to-end combined robust physics loss with mixed normal and outlier inputs.
    """
    criterion = CombinedPhysicsLoss(gamma=5.0, physics_weight=0.1, max_accel=30.0)
    
    pred = torch.tensor([[60.0]], requires_grad=True)
    target = torch.tensor([[62.0]])
    speed_seq = torch.tensor([[50.0, 60.0]])  # delta = 10 <= 30 => phys penalty = 0
    
    loss = criterion(pred, target, speed_seq)
    loss.backward()
    
    assert loss.item() > 0.0
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()
