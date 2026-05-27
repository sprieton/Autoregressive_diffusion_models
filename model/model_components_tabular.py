"""Input parametrization f(i, m, t) for tabular categorical data.

Tabular sibling of InputProcessingImage. Operates on (B, D) integer vectors
instead of (B, 1, H, W) images. Each column is turned into a token by summing
three embeddings:
    - value embedding   : the (possibly absorbed) category code
    - column embedding  : which heterogeneous column this token belongs to
    - mask embedding    : observed (1) vs masked/unobserved (0)
The sinusoidal time embedding is returned separately for conditioning.
"""

import torch
import torch.nn as nn

from utils.utils import TimeEmbedding


class InputProcessingTabular(nn.Module):
    def __init__(self, num_classes, D, emb_dim=64, max_time=1000):
        super().__init__()
        assert emb_dim % 2 == 0, "emb_dim must be even for sinusoidal time embedding"
        self.value_emb = nn.Embedding(num_classes, emb_dim)
        self.col_emb = nn.Embedding(D, emb_dim)
        self.mask_emb = nn.Embedding(2, emb_dim)
        self.time_embedding = TimeEmbedding(emb_dim, max_time)
        self.register_buffer("col_idx", torch.arange(D))

    def forward(self, x, t, mask):
        # x: (B, D) long ; t: (B,) ; mask: (B, D) in {0,1}
        assert x.dtype in (torch.int32, torch.int64)
        h = self.value_emb(x)                       # (B, D, E)
        h = h + self.col_emb(self.col_idx)[None]    # (1, D, E) broadcast
        h = h + self.mask_emb(mask.long())          # (B, D, E)
        temb = self.time_embedding(t.float())       # (B, E)
        return h, temb
