"""Tabular networks for the OA-ARDM.

Both take the token features (B, D, E) and time embedding (B, E) produced by
InputProcessingTabular and return per-column logits (B, D, num_classes). They
mirror the MNIST LeNet/ViT pair: a plain MLP and a small Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPWithTime(nn.Module):
    """Flatten the column tokens, condition on time, predict all columns at once."""

    def __init__(self, D, emb_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.D = D
        self.num_classes = num_classes
        self.time_proj = nn.Linear(emb_dim, emb_dim)
        self.fc1 = nn.Linear(D * emb_dim + emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, D * num_classes)

    def forward(self, feats, temb):
        B = feats.size(0)
        h = feats.reshape(B, -1)                          # (B, D*E)
        h = torch.cat([h, self.time_proj(temb)], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.out(h).view(B, self.D, self.num_classes)


class TabTransformer(nn.Module):
    """Transformer encoder over column tokens — order-agnostic by construction."""

    def __init__(self, D, emb_dim, num_classes, num_heads=4, num_layers=2):
        super().__init__()
        self.num_classes = num_classes
        self.time_proj = nn.Linear(emb_dim, emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=4 * emb_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(emb_dim, num_classes)

    def forward(self, feats, temb):
        h = feats + self.time_proj(temb).unsqueeze(1)     # broadcast over columns
        h = self.encoder(h)                                # (B, D, E)
        return self.out(h)                                 # (B, D, num_classes)
