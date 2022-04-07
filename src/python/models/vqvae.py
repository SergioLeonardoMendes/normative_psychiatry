""" VQ-VAE model [1]

References:
    [1] - Neural Discrete Representation Learning (https://arxiv.org/pdf/1711.00937.pdf)
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast

from layers.quantizer import Quantizer


class ResidualLayer(nn.Sequential):
    def __init__(self, n_channels, n_res_channels, p_dropout):
        super().__init__(nn.Conv3d(n_channels, n_res_channels, kernel_size=3, padding=1),
                         nn.ReLU(True),
                         nn.Dropout3d(p_dropout),
                         nn.Conv3d(n_res_channels, n_channels, kernel_size=1))

    def forward(self, x):
        return F.relu(x + super().forward(x), True)


class VQVAE(nn.Module):
    def __init__(self,
                 n_embed=8,
                 embed_dim=64,
                 n_alpha_channels=1,
                 n_channels=64,
                 n_res_channels=64,
                 n_res_layers=2,
                 p_dropout=0.1,
                 commitment_cost=0.25,
                 vq_decay=0.99,
                 latent_resolution="low"):
        super().__init__()
        self.n_embed = n_embed
        self.latent_resolution = latent_resolution

        if latent_resolution == "low":
            self.encoder = nn.Sequential(
                nn.Conv3d(n_alpha_channels, n_channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(
                    *[ResidualLayer(n_channels // 2, n_res_channels // 2, p_dropout) for _ in range(n_res_layers)]),
                nn.Conv3d(n_channels // 2, n_channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(
                    *[ResidualLayer(n_channels // 2, n_res_channels // 2, p_dropout) for _ in range(n_res_layers)]),
                nn.Conv3d(n_channels // 2, n_channels, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(*[ResidualLayer(n_channels, n_res_channels, p_dropout) for _ in range(n_res_layers)]),
                nn.Conv3d(n_channels, embed_dim, 3, stride=1, padding=1),
            )

            self.decoder = nn.Sequential(
                nn.Conv3d(embed_dim, n_channels, 3, stride=1, padding=1),
                nn.Sequential(*[ResidualLayer(n_channels, n_res_channels, p_dropout) for _ in range(n_res_layers)]),
                nn.ConvTranspose3d(n_channels, n_channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(
                    *[ResidualLayer(n_channels // 2, n_res_channels // 2, p_dropout) for _ in range(n_res_layers)]),
                nn.ConvTranspose3d(n_channels // 2, n_channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(
                    *[ResidualLayer(n_channels // 2, n_res_channels // 2, p_dropout) for _ in range(n_res_layers)]),
                nn.ConvTranspose3d(n_channels // 2, n_alpha_channels, 4, stride=2, padding=1),
            )

        elif latent_resolution == "mid":
            self.encoder = nn.Sequential(
                nn.Conv3d(n_alpha_channels, n_channels // 2 , 6, stride=3, padding=2),
                nn.ReLU(),
                nn.Sequential(*[ResidualLayer(n_channels  // 2, n_res_channels // 2, p_dropout) for _ in range(n_res_layers)]),
                nn.Conv3d(n_channels // 2, n_channels, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(*[ResidualLayer(n_channels, n_res_channels, p_dropout) for _ in range(n_res_layers)]),
                nn.Conv3d(n_channels, embed_dim, 3, stride=1, padding=1),
            )

            self.decoder = nn.Sequential(
                nn.Conv3d(embed_dim, n_channels, 3, stride=1, padding=1),
                nn.Sequential(*[ResidualLayer(n_channels, n_res_channels, p_dropout) for _ in range(n_res_layers)]),
                nn.ConvTranspose3d(n_channels, n_channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(*[ResidualLayer(n_channels // 2, n_res_channels // 2, p_dropout) for _ in range(n_res_layers)]),
                nn.ConstantPad3d((1, 0, 1, 0, 1, 0), 0),
                nn.ConvTranspose3d(n_channels // 2, n_alpha_channels, 6, stride=3, padding=3),
            )

        self.codebook = Quantizer(n_embed, embed_dim, commitment_cost=commitment_cost, decay=vq_decay)

    def encode_code(self, x):
        z = self.encoder(x)
        indices = self.codebook(z)[1]
        return indices

    def decode_code(self, latents):
        latents = self.codebook.embedding(latents).permute(0, 4, 1, 2, 3).contiguous()
        return self.decoder(latents)

    def forward(self, x):
        with autocast(enabled=True):
            z = self.encoder(x)
            with autocast(enabled=False):
                e, embed_idx, latent_loss = self.codebook(z.float())
            x_tilde = self.decoder(e.half())

        avg_probs = lambda e: torch.histc(e.float(), bins=self.n_embed, max=self.n_embed).float().div(e.numel())
        perplexity = lambda avg_probs: torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        perplexity_code = perplexity(avg_probs(embed_idx))

        return x_tilde, latent_loss, perplexity_code, embed_idx

    def reconstruct(self, x):
        z = self.encode_code(x)
        x_recon = self.decode_code(z)
        return x_recon
