# models/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.network import DualObjectiveAdversarialNet
from models.dataset import ResumeCandidateDataset

class DualObjectiveTrainer:
    """
    Trainer executing dual-objective adversarial minimization/maximization:
    min_{Encoder, Predictor} max_{Adversary} [ L_task(y_pred, y) - lambda * L_adv(s_pred, s) ]
    Implemented via Gradient Reversal Layer backpropagation.
    """
    def __init__(self, input_dim: int, lr: float = 0.005, adv_weight: float = 1.0):
        self.model = DualObjectiveAdversarialNet(input_dim=input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion_task = nn.BCELoss()
        self.criterion_adv = nn.BCELoss()
        self.adv_weight = adv_weight

    def train_epoch(self, dataloader: DataLoader, alpha: float = 1.0):
        self.model.train()
        running_task_loss = 0.0
        running_adv_loss = 0.0
        running_total_loss = 0.0

        for X_batch, y_batch, s_batch in dataloader:
            self.optimizer.zero_grad()
            
            y_pred, s_pred = self.model(X_batch, alpha=alpha)
            
            # Primary Job-Fit Loss
            loss_task = self.criterion_task(y_pred, y_batch)
            
            # Sensitive Attribute Classifier Loss
            loss_adv = self.criterion_adv(s_pred, s_batch)
            
            # Total Dual-Objective Loss (GRL automatically flips sign of loss_adv gradient during backprop)
            total_loss = loss_task + (self.adv_weight * loss_adv)
            
            total_loss.backward()
            self.optimizer.step()
            
            running_task_loss += loss_task.item() * len(X_batch)
            running_adv_loss += loss_adv.item() * len(X_batch)
            running_total_loss += total_loss.item() * len(X_batch)

        n = len(dataloader.dataset)
        return {
            'task_loss': running_task_loss / n,
            'adv_loss': running_adv_loss / n,
            'total_loss': running_total_loss / n
        }

    def predict(self, X: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            y_pred, s_pred = self.model(X, alpha=0.0)
        return y_pred.numpy(), s_pred.numpy()