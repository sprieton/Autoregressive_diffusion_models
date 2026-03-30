import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --------------------------------------------------
# Time Embedding (sinusoidal)
# --------------------------------------------------
class TimeEmbedding(nn.Module):
    """
    Embeds the timestep t into a continuous vector representation.

    In Algorithm 2:
    - t ~ Uniform(1, ..., D)
    - This embedding allows the model to condition on which step Lt is optimized

    Args:
        t: Tensor of shape (B,) containing timesteps

    Returns:
        Tensor of shape (B, dim) representing time embeddings
    """
    def __init__(self, dim, max_time=1000):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,)
        half_dim = self.dim // 2

        # Standard sinusoidal embedding (as in diffusion models)
        emb = torch.exp(
            torch.arange(half_dim, device=t.device) *
            -(math.log(10000.0) / (half_dim - 1))
        )

        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return emb  # (B, dim)
    

# ---------------------------
# Categoric Distribution
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
    