# models/network.py
import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL).
    Passes features unchanged during forward propagation.
    Flips the sign of the gradient and scales it by alpha during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


class DualObjectiveAdversarialNet(nn.Module):
    """
    Modular Dual-Objective Adversarial Neural Network.
    
    1. Shared Feature Encoder E(x): Extracts latent representation z from vectorized resume features.
    2. Primary Predictor Head P(z): Predicts job qualification score (Task Objective L_task).
    3. Adversarial Discriminator Head A(z): Attempts to detect sensitive demographic attribute S from z (Adversarial Objective L_adv).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 16):
        super(DualObjectiveAdversarialNet, self).__init__()
        
        # Shared Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Primary Job Qualification Predictor Head
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # Adversarial Sensitive Attribute Discriminator Head
        self.adversary = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        latent = self.encoder(x)
        
        # Primary task prediction
        y_pred = self.predictor(latent)
        
        # Adversarial sensitivity prediction through Gradient Reversal Layer
        reversed_latent = grad_reverse(latent, alpha)
        s_pred = self.adversary(reversed_latent)
        
        return y_pred, s_pred