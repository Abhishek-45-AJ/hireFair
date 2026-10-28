# models/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class ResumeCandidateDataset(Dataset):
    """
    PyTorch Dataset wrapper for vectorised candidate resumes,
    ground-truth qualifications, and demographic attribute labels.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, s: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.s = torch.tensor(s, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.s[idx]