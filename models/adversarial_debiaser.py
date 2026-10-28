import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient scaled by alpha
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)

class AdversarialDebiasingNet(nn.Module):
    """
    Compact PyTorch Adversarial Debiasing Model with GRL.
    - Shared Encoder: Extracts 8-dim latent representation from candidate features.
    - Predictor Head: Predicts job qualification probability.
    - Adversary Head: Tries to predict sensitive gender label from latent features.
    - GRL: Flips adversary gradient to purge gender information from encoder.
    """
    def __init__(self, input_dim=4, hidden_dim=16):
        super(AdversarialDebiasingNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.adversary = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x, alpha=1.0):
        latent = self.encoder(x)
        y_pred = self.predictor(latent)
        reversed_latent = grad_reverse(latent, alpha)
        s_pred = self.adversary(reversed_latent)
        return y_pred, s_pred

# Global model instance
global_adversarial_model = None

def train_pytorch_adversarial_model(epochs=40, lr=0.01):
    global global_adversarial_model
    np.random.seed(42)
    torch.manual_seed(42)

    n = 150
    genders = np.random.choice([0, 1], size=n) # 0: Male, 1: Female
    skills = np.random.uniform(0.3, 1.0, size=n)
    sims = np.random.uniform(0.2, 0.95, size=n)
    exps = np.random.uniform(0.1, 1.0, size=n)
    
    # Ground truth qualification: purely skill + sim + exp
    qual = (skills * 0.5 + sims * 0.3 + exps * 0.2)
    ground_truth = (qual >= 0.52).astype(np.float32)
    
    # Gender cue signal: male group gets positive cue bias
    gender_cue = np.where(genders == 0, 0.75 + np.random.normal(0, 0.08, n), 0.25 + np.random.normal(0, 0.08, n))
    gender_cue = np.clip(gender_cue, 0, 1)

    X = np.column_stack((skills, sims, exps, gender_cue)).astype(np.float32)
    Y = ground_truth.reshape(-1, 1)
    S = genders.astype(np.float32).reshape(-1, 1)

    X_t = torch.tensor(X)
    Y_t = torch.tensor(Y)
    S_t = torch.tensor(S)

    model = AdversarialDebiasingNet(input_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    crit_task = nn.BCELoss()
    crit_adv = nn.BCELoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        y_pred, s_pred = model(X_t, alpha=1.2)
        
        loss_task = crit_task(y_pred, Y_t)
        loss_adv = crit_adv(s_pred, S_t)
        total_loss = loss_task + loss_adv
        
        total_loss.backward()
        optimizer.step()
        
        history.append({
            'epoch': epoch,
            'task_loss': round(float(loss_task.item()), 4),
            'adversary_loss': round(float(loss_adv.item()), 4),
            'total_loss': round(float(total_loss.item()), 4)
        })

    global_adversarial_model = model
    return model, history