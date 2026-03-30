import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import TimeEmbedding

# --------------------------------------------------
# Input Processing Module (PyTorch version)
# --------------------------------------------------
class InputProcessingImage(nn.Module):
    """
    Implements the input parametrization f(i, m, t) described in Algorithm 2.

    IMPORTANT:
    This module does NOT implement Algorithm 2 itself.
    It only prepares the input to the neural network given:
        - x (data)
        - m (mask derived from σ and t)
        - t (timestep)

    Algorithm 2 (outside this module):
        1. Sample t ~ Uniform(1, ..., D)
        2. Sample permutation σ
        3. Compute mask m = (σ < t)

    This module implements:
        f(i, m, t), where i = m * x + (1 - m) * a

    Inputs:
        x    : (B, 1, H, W) integer tensor with values in {0,...,K-1}
               → original data point
        mask : (B, 1, H, W) binary tensor {0,1}
               → m = (σ < t), indicates observed variables
        t    : (B,) timestep tensor
               → indicates which Lt term is optimized

    Outputs:
        h_first : (B, C, H, W)
                  → feature representation used by the main network
        temb    : (B, C)
                  → time embedding for conditioning deeper layers
    """

    def __init__(self, num_classes=256, num_channels=64, max_time=1000):
        super().__init__()

        self.num_classes = num_classes
        self.num_channels = num_channels

        assert num_channels % 4 == 0

        # --------------------------------------------------
        # Continuous pathway (3/4 of channels)
        # Processes normalized image + mask
        # --------------------------------------------------
        self.conv_in = nn.Conv2d(
            in_channels=2,  # image + mask
            out_channels=num_channels * 3 // 4,
            kernel_size=3,
            padding=1
        )

        # --------------------------------------------------
        # Discrete embedding pathway (1/4 of channels)
        # Models x as categorical variables
        # --------------------------------------------------
        self.emb_ch = num_channels // 4
        self.embedding = nn.Embedding(num_classes, self.emb_ch)
        self.emb_proj = nn.Linear(self.emb_ch, self.emb_ch)

        # --------------------------------------------------
        # Time embedding (conditioning on t)
        # --------------------------------------------------
        self.time_embedding = TimeEmbedding(num_channels, max_time)

    def forward(self, x, t, mask):

        assert self.num_classes >= 1
        assert x.dtype in [torch.int32, torch.int64]

        # --------------------------------------------------
        # Step 0 (Algorithm context):
        # x = original datapoint
        # mask = m = (σ < t)  → provided externally
        # --------------------------------------------------

        # Keep discrete version for categorical embeddings
        x_int = x  # (B,1,H,W)

        # --------------------------------------------------
        # Step 1: Handle timestep t
        # Corresponds to conditioning on Lt in Algorithm 2
        # --------------------------------------------------
        if t is None:
            print("Warning: model not conditioned on t")
            t = torch.zeros(x.shape[0], device=x.device)

        # --------------------------------------------------
        # Step 2: Construct continuous input representation
        # Normalize x to [-1, 1] for stable training
        # --------------------------------------------------
        x = x.float() * 2.0 / float(self.num_classes) - 1.0

        # --------------------------------------------------
        # Step 3: Provide mask m explicitly to the model
        # This allows the model to distinguish:
        #   - observed variables (m = 1)
        #   - masked variables (m = 0)
        #
        # This is crucial to model:
        #   p(x_k | x_{σ(<t)})
        # --------------------------------------------------
        x = torch.cat([x, mask], dim=1)  # (B, 2, H, W)

        # --------------------------------------------------
        # Step 4: Time embedding
        # Gives information about how many variables are masked
        # (i.e., which Lt term is optimized)
        # --------------------------------------------------
        temb = self.time_embedding(t.float())  # (B, C)

        # --------------------------------------------------
        # Step 5: Continuous pathway
        # CNN processes normalized image + mask
        # --------------------------------------------------
        h_first = self.conv_in(x)  # (B, 3/4*C, H, W)

        # --------------------------------------------------
        # Step 6: Discrete (categorical) embedding pathway
        # Important because likelihood is:
        #   log C(x_k | θ_k)
        # --------------------------------------------------

        # Remove channel dimension → (B,H,W)
        x_flat = x_int.squeeze(1)

        # Embed each pixel as categorical variable
        emb_x = self.embedding(x_flat)  # (B,H,W,emb_ch)

        # Linear projection
        emb_x = self.emb_proj(emb_x)

        # Convert to convolutional format
        emb_x = emb_x.permute(0, 3, 1, 2)  # (B, emb_ch, H, W)

        # --------------------------------------------------
        # Step 7: Combine both representations
        # Final input to network f
        # --------------------------------------------------
        h_first = torch.cat([h_first, emb_x], dim=1)  # (B, C, H, W)

        # --------------------------------------------------
        # Output:
        # h_first → input features for predicting logits θ
        # temb    → conditioning for deeper layers
        # --------------------------------------------------
        return h_first, temb