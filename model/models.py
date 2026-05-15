import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.config as cfg

# 1- simple LeNet
class LeNetWithTime(nn.Module):
    def __init__(self, num_classes=cfg.num_clases, 
                 time_emb_dim=cfg.time_emb_dim, hidden_dim=cfg.hidden_dim):
        super().__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(64, 32, kernel_size=5, padding=2)    # (B,64,28,28) -> (B,32,28,28)
        self.pool = nn.AvgPool2d(2)                                 # (B,32,28,28) -> (B,32,14,14)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)    # (B,32,14,14) -> (B,64,14,14)
        # Pool: (B,64,14,14) -> (B,64,7,7)
        # MLP for logits
        self.fc1 = nn.Linear(64*7*7 + time_emb_dim, hidden_dim)     # t_emb (B,64) + pool -> (B, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, cfg.img_h*cfg.img_w*num_classes)
        self.num_classes = num_classes

    def forward(self, feat_masked, t_emb):
        B = feat_masked.size(0)
        h = F.relu(self.conv1(feat_masked))
        h = self.pool(h)
        h = F.relu(self.conv2(h))
        h = self.pool(h)
        
        h = h.view(B, -1)  # Flatten (B, 64*7*7)
        h = torch.cat([h, t_emb], dim=1)  # (B, 64*7*7 + time_emb_dim)
        h = F.relu(self.fc1(h))
        out = self.fc2(h)  # (B, 28*28*num_classes)
        out = out.view(B, cfg.img_h, cfg.img_w, self.num_classes).permute(0, 3, 1, 2)  # (B, num_classes, 28, 28)
        return out

# 2- Tiny Transformer (ViT-style)
class TinyTimeViT(nn.Module):
    def __init__(self, img_size=cfg.img_h, patch_size=7, in_channels=64, 
                 emb_dim=cfg.time_emb_dim, num_heads=cfg.num_heads, 
                 num_layers=cfg.num_trans_layers, time_emb_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.emb_dim = emb_dim
        self.img_size = img_size
        self.num_classes = cfg.num_clases

        # --- patch embedding ---
        self.patch_embed = nn.Conv2d(in_channels, emb_dim, patch_size, stride=patch_size)
        # --- positional encoding ---
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))
        # --- transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # --- time embedding projection ---
        self.time_proj = nn.Linear(time_emb_dim, emb_dim)
        # --- project back to pixel logits per patch ---
        self.patch_out = nn.Linear(emb_dim, patch_size * patch_size * self.num_classes)

    def forward(self, feat_masked, t_emb):
        B = feat_masked.size(0)
        p = self.patch_size

        # --- patch embedding ---
        x = self.patch_embed(feat_masked)          # (B, emb_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)          # (B, num_patches, emb_dim)
        x = x + self.pos_embed                     # add positional encoding

        # --- add time embedding ---
        t_emb_proj = self.time_proj(t_emb).unsqueeze(1)  # (B,1,emb_dim)
        x = x + t_emb_proj                           # broadcast to all patches

        # --- transformer ---
        x = self.transformer(x)                     # (B, num_patches, emb_dim)

        # --- predict logits per patch ---
        x = self.patch_out(x)                       # (B, num_patches, p*p*num_classes)
        x = x.view(B, self.num_patches, p, p, self.num_classes)  # (B, N, p, p, C)

        # --- reconstruct image ---
        # reshape patches to H, W
        h_patches = self.img_size // p
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, N, p, p)
        x = x.reshape(B, self.num_classes, h_patches, h_patches, p, p)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, C, H_p, p, W_p, p)
        x = x.reshape(B, self.num_classes, self.img_size, self.img_size)

        return x  # (B, num_classes, H, W)