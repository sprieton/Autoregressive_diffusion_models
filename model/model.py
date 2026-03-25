import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import jax.numpy as jnp
from flax import linen as nn
from absl import logging
import utils.config as cfg

# ---------------------------
# Distribución categórica
# ---------------------------
class SoftmaxCategorical(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

    def log_prob(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()
        logits = logits.view(*x.shape, self.n_classes)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.view(*logits.shape[:-1], self.n_classes)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()
    



# ---------------------------
# Model definitions
# ---------------------------
# 1- simple MLP
class SimpleMLP(nn.Module):
    def __init__(self, D=784, hidden=512, K=256):
        super().__init__()
        self.D = D
        self.K = K
        self.fc1 = nn.Linear(D, hidden)
        self.fc2 = nn.Linear(hidden, D * K)
    
    def forward(self, x_masked):
        # x_masked: (batch, D), valores en [0,1] normalizados
        h = F.relu(self.fc1(x_masked))
        out = self.fc2(h)
        out = out.view(-1, self.D, self.K)  # (batch, D, K)
        return out
class LightMNISTModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.relu = nn.ReLU()
        self.head = nn.Conv2d(hidden_dim, 2, kernel_size=1)  # 2 clases para binario

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        logits = self.head(x)
        return logits.permute(0, 2, 3, 1)  # [B, H, W, n_classes]


# ---------------------------
# Clase de entrenamiento
# ---------------------------
class Trainer:
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dist = SoftmaxCategorical(n_channels=1, n_classes=2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for imgs, _ in self.train_loader:
            imgs = imgs.to(self.device).long()  # MNIST binarizado 0/1
            self.optimizer.zero_grad()
            logits = self.model(imgs.float())
            loss = -self.dist.log_prob(imgs, logits).mean()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        return total_loss / len(self.train_loader.dataset)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for imgs, _ in self.test_loader:
                imgs = imgs.to(self.device).long()
                logits = self.model(imgs.float())
                loss = -self.dist.log_prob(imgs, logits).mean()
                total_loss += loss.item() * imgs.size(0)
        return total_loss / len(self.test_loader.dataset)

    @torch.no_grad()
    def sample_images(self, n=16):
        self.model.eval()
        z = torch.zeros((n, 1, 28, 28), device=self.device)  # dummy input
        logits = self.model(z.float())
        samples = self.dist.sample(logits)  # [B, H, W]
        
        # Convertir a numpy con shape (B, H, W)
        samples = samples.squeeze(1).cpu().numpy() if samples.shape[1]==1 else samples.cpu().numpy()
        return samples

