from flax import linen as nn
import jax.numpy as jnp
from absl import logging
import utils.time_embedding as time_embedding

class InputProcessingImage(nn.Module):
  """Input embedding module for ARDMs (image case).

  This module implements the parametrization described in Algorithm 2:
  given x, mask m, and timestep t, it constructs the input representation
  i = m * x + (1 - m) * a and feeds it to the network f(i, m, t).
  """

  num_classes: int
  num_channels: int
  max_time: float

  @nn.compact
  def __call__(self, x, t, mask, train):
    # x: discrete image (B, H, W, C) with values in {0, ..., K-1}
    # mask: binary mask m indicating observed variables (σ < t)
    # t: timestep sampled from Uniform(1, ..., D)

    assert self.num_classes >= 1
    assert x.dtype == jnp.int32

    # ------------------------------------------------------------------
    # (Algorithm 2 context)
    # x is the original datapoint
    # mask m = (σ < t) is given externally (NOT computed here)
    # ------------------------------------------------------------------

    # Keep discrete version for categorical embeddings (used later)
    x_int = x

    # ------------------------------------------------------------------
    # Step: handle timestep input (conditioning on t)
    # Corresponds to f(i, m, t) in the paper
    # ------------------------------------------------------------------
    if t is None:
      logging.info('Warning: model is not conditioned on timestep t.')
      t = jnp.zeros(x.shape[0], dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Step: construct continuous representation of input
    # (part of i = m*x + (1-m)*a, assuming masked values already applied)
    # Normalize x to [-1, 1] for stable training
    # ------------------------------------------------------------------
    x = x * 2. / float(self.num_classes) - 1.

    # ------------------------------------------------------------------
    # Step: provide mask explicitly to the model
    # This allows the network to distinguish:
    # - observed pixels (m = 1)
    # - masked pixels (m = 0)
    # ------------------------------------------------------------------
    x = jnp.concatenate([x, mask], axis=3)

    # ------------------------------------------------------------------
    # Step: embed timestep t
    # Gives the model information about how many variables are masked
    # (i.e., which Lt term is being optimized)
    # ------------------------------------------------------------------
    temb = time_embedding.TimeEmbedding(
        self.num_channels, self.max_time)(t)

    # ------------------------------------------------------------------
    # Step: continuous pathway (standard CNN processing)
    # Processes normalized pixel values + mask
    # ------------------------------------------------------------------
    assert self.num_channels % 4 == 0
    h_first = nn.Conv(
        features=self.num_channels * 3 // 4,
        kernel_size=(3, 3),
        strides=(1, 1),
        name='conv_in')(x)

    # ------------------------------------------------------------------
    # Step: discrete (categorical) embedding pathway
    # Models x as categorical variables (important for log C(x_k | θ_k))
    # ------------------------------------------------------------------
    emb_ch = self.num_channels // 4

    # Embed discrete pixel values
    emb_x = nn.Embed(self.num_classes, emb_ch)(x_int)

    # Flatten channel-wise embeddings
    emb_x = emb_x.reshape(*x_int.shape[:-1], emb_ch * x_int.shape[-1])

    # Project embeddings back to feature space
    h_emb_x = nn.Dense(features=emb_ch, name='emb_x_proj')(emb_x)

    # ------------------------------------------------------------------
    # Step: combine continuous + discrete representations
    # This forms the final input to the main network f
    # ------------------------------------------------------------------
    h_first = jnp.concatenate([h_first, h_emb_x], axis=3)

    # ------------------------------------------------------------------
    # Output:
    # h_first → features representing f(i, m, t)
    # temb → timestep embedding (used in deeper layers)
    # ------------------------------------------------------------------
    return h_first, temb