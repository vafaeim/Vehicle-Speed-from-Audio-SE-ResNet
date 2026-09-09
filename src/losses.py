# robust statistical and physics informed loss formulations

import torch
import torch.nn as nn
from .config import Config

# smooth cauchy loss lorentzian m estimator
class CauchyLoss(nn.Module):
    def __init__(self, gamma=5.0, reduction='mean'):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        
    def forward(self, pred, target):
        e = pred - target
        loss = (self.gamma ** 2 / 2.0) * torch.log(1.0 + (e / self.gamma) ** 2)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# alias for smooth cauchy loss
SmoothCauchyLoss = CauchyLoss

# huber loss with bounded lipschitz gradient
class HuberLoss(nn.Module):
    def __init__(self, delta=5.0, reduction='mean'):
        super().__init__()
        self.delta = float(delta)
        self.reduction = reduction
        
    def forward(self, pred, target):
        e = pred - target
        abs_e = torch.abs(e)
        linear_mask = abs_e > self.delta
        loss = torch.where(linear_mask, self.delta * (abs_e - 0.5 * self.delta), 0.5 * (e ** 2))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# kinematic acceleration regularizer penalizing physical discontinuities
class KinematicAccelerationLoss(nn.Module):
    def __init__(self, max_accel=30.0, dt=1.0, reduction='mean'):
        super().__init__()
        self.max_accel = float(max_accel)
        self.dt = float(dt)
        self.reduction = reduction
        
    def forward(self, pred_speeds):
        # return zero penalty if single frame prediction
        if pred_speeds.ndim < 2 or pred_speeds.shape[-1] <= 1:
            return torch.tensor(0.0, device=pred_speeds.device, dtype=pred_speeds.dtype)
        
        # calculate frame to frame acceleration
        accel = torch.diff(pred_speeds, dim=-1) / self.dt
        excess = torch.clamp(torch.abs(accel) - self.max_accel, min=0.0)
        penalty = excess ** 2
        if self.reduction == 'mean':
            return penalty.mean()
        elif self.reduction == 'sum':
            return penalty.sum()
        return penalty

# alias for kinematic acceleration loss
PhysicsKinematicLoss = KinematicAccelerationLoss

# physical speed domain boundary constraint
class DomainBoundaryLoss(nn.Module):
    def __init__(self, speed_min=10.0, speed_max=140.0, reduction='mean'):
        super().__init__()
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.reduction = reduction
        
    def forward(self, pred):
        low_penalty = torch.clamp(self.speed_min - pred, min=0.0) ** 2
        high_penalty = torch.clamp(pred - self.speed_max, min=0.0) ** 2
        penalty = low_penalty + high_penalty
        if self.reduction == 'mean':
            return penalty.mean()
        elif self.reduction == 'sum':
            return penalty.sum()
        return penalty

# composite physics informed multi objective loss
class PhysicsInformedLoss(nn.Module):
    def __init__(self, gamma=5.0, physics_weight=0.1, bound_weight=0.05, max_accel=30.0, dt=1.0, loss_type='cauchy', huber_delta=10.0):
        super().__init__()
        self.loss_type = loss_type
        self.gamma = float(gamma)
        self.huber_delta = float(huber_delta)
        
        self.cauchy_loss = CauchyLoss(gamma=gamma)
        self.huber_loss = HuberLoss(delta=huber_delta)
        
        if loss_type == 'huber':
            self.regression_loss = self.huber_loss
        else:
            self.regression_loss = self.cauchy_loss
            
        self.cauchy = self.regression_loss
        self.physics = KinematicAccelerationLoss(max_accel=max_accel, dt=dt)
        self.bound = DomainBoundaryLoss(speed_min=Config.SPEED_MIN, speed_max=Config.SPEED_MAX)
        
        self.physics_weight = float(physics_weight)
        self.bound_weight = float(bound_weight)
        
    def forward(self, pred, target, speed_seq=None):
        total_loss = self.regression_loss(pred, target)
        
        # When using Cauchy loss on raw km/h targets, add an auxiliary Huber regularization term (0.1x)
        # to ensure non-vanishing linear gradient pull (|e| > 10 km/h) preventing mean-collapse
        if self.loss_type == 'cauchy':
            total_loss = total_loss + 0.1 * self.huber_loss(pred, target)
            
        # add kinematic acceleration penalty if trajectory provided
        if speed_seq is not None and speed_seq.shape[-1] > 1:
            total_loss = total_loss + self.physics_weight * self.physics(speed_seq)
            
        # add domain boundary penalty
        if self.bound_weight > 0.0:
            total_loss = total_loss + self.bound_weight * self.bound(pred)
            
        return total_loss

# alias for combined physics loss
CombinedPhysicsLoss = PhysicsInformedLoss
