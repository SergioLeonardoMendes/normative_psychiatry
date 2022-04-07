""" Definition of the Vector Quantized Variational Autoencoder.

Based on:
https://github.com/rosinality/vq-vae-2-pytorch
https://github.com/ritheshkumar95/pytorch-vqvae
https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=kgobZjvvGs3q
https://github.com/rll/deepul/blob/master/demos/lecture4_latent_variable_models_demos.ipynb
https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
https://github.com/kamenbliznashki/generative_models/blob/master/vqvae.py
"""
import torch
from torch import nn
from torch.nn import functional as F


class Quantizer(nn.Module):
    """ Module representing the vector quantizer layer.

    Implements a slightly modified version of the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937

    The difference is that this module uses exponential moving averages to update the embedding vectors
    instead of an auxiliary loss. This has the advantage that the embedding
    updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
    ...) used for the encoder, decoder and other parts of the architecture. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.
    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.

    Args:
      n_embeddings: integer, the number of vectors in the quantized space.
      embedding_dim: integer representing the dimensionality of the tensors in the
        quantized space. Inputs to the modules must be in this format as well.
      commitment_cost: scalar which controls the weighting of the loss terms (see
        equation 4 in the paper).
      decay: float, decay for the moving averages.
      eps: small float constant to avoid numerical instability.
    """

    def __init__(self, n_embed, embed_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.requires_grad = False

        self.embed_dim = embed_dim
        self.n_emb = n_embed
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.register_buffer('N', torch.zeros(n_embed))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())

    def forward(self, x):
        b, c, h, w, d = x.shape
        weight = self.embedding.weight

        # convert inputs from BCHW -> BHWC and flatten input
        flat_inputs = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embed_dim)

        # Calculate distances
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * torch.mm(flat_inputs, weight.t()) \
                    + (weight.t() ** 2).sum(dim=0, keepdim=True)

        # Encoding
        embed_idx = torch.max(-distances, dim=1)[1]
        embed_onehot = F.one_hot(embed_idx, self.n_emb).type(flat_inputs.dtype)

        # Quantize and unflatten
        embed_idx = embed_idx.view(b, h, w, d)
        quantized = self.embedding(embed_idx).permute(0, 4, 1, 2, 3).contiguous()

        # Use EMA to update the embedding vectors
        if self.training:
            self.N.data.mul_(self.decay).add_(1 - self.decay, embed_onehot.sum(0))

            # Laplace smoothing of the cluster size
            embed_sum = torch.mm(flat_inputs.t(), embed_onehot)
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum.t())

            n = self.N.sum()
            weights = (self.N + self.eps) / (n + self.n_emb * self.eps) * n
            embed_normalized = self.embed_avg / weights.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        latent_loss = self.commitment_cost * F.mse_loss(quantized.detach(), x)

        # Stop optimization from accessing the embedding
        quantized_st = (quantized - x).detach() + x

        return quantized_st, embed_idx, latent_loss
