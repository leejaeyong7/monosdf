import math
import torch
import torch.nn as nn

from .grid_sample import grid_sample

class FreqHashO(nn.Module):
    def __init__(self, log2_res=8, num_freqs=6, num_channels=48, std=0.001):
        super().__init__()
        min_log2_freq = 0
        max_log2_freq = num_freqs - 1
        freqs = 2.0 ** torch.linspace(min_log2_freq, max_log2_freq, num_freqs)
        self.freqs = nn.Parameter(freqs, False)
        self.num_channels = num_channels

    def pos_encode(self, points):
        N = points.shape[0]
        freq_points = points.view(N, 1, -1) * self.freqs.to(points).view(1, -1, 1)

        return torch.stack((freq_points.sin(), freq_points.cos()), -2).view(N, -1).repeat(1, self.num_channels)

    def forward(self, points):
        """
        must return features given points (and optional dirs)
        """
        return self.pos_encode(points)

class FreqHash(nn.Module):
    def __init__(self, log2_res=8, num_freqs=6, num_channels=48, std=0.001):
        super().__init__()
        min_log2_freq = 0
        max_log2_freq = num_freqs - 1
        freqs = 2.0 ** torch.linspace(min_log2_freq, max_log2_freq, num_freqs)
        self.freqs = nn.Parameter(freqs, False)
        res = 2 ** log2_res
        f = torch.randn((num_freqs * 2 * 3, num_channels, res, 1)) * std
        self.cv = nn.Parameter(f, True)
        self.num_freqs = num_freqs

    def pos_encode(self, points):
        N = points.shape[0]
        freq_points = points.view(N, 1, -1) * self.freqs.to(points).view(1, -1, 1)

        # NxFx2x3 => Nx(Fx2x3)
        return torch.stack((freq_points.sin(), freq_points.cos()), -2).view(N, -1).T.contiguous()

    def encoder(self, encs, oencs):
        # (Fx2x3)xN
        F = self.num_freqs
        N = encs.shape[-1]

        # (Fx2x3)xCxRx1
        cv = self.cv
        C = cv.shape[1]
        # Fx2x3 x N => F23x1xNx2
        encs = encs.view(-1, 1, N, 1)
        w = torch.zeros_like(encs)
        grid = torch.cat((w, encs), -1)

        # (Fx2x3)xCx1xN
        fs = grid_sample(cv, grid, True).view(F, 2, 3, C, N)
        fs = fs + oencs.view(F, 2, 3, 1, N)
        # fs = encs.view(F, 2, 3, 1, N).repeat(1, 1, 1, C, 1)
        return fs.permute(4, 3, 0, 1, 2).reshape(N, -1)

    def forward(self, points, scale):
        """
        must return features given points (and optional dirs)
        """
        enc = self.pos_encode(points / scale)
        oenc = self.pos_encode(points / scale)
        return self.encoder(enc, oenc)


class MultiFreqEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""
    def __init__(self, res_scale=4, num_freqs=6, num_channels=8, std=0.001, x_dim=3):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = 0
        self.max_deg = num_freqs - 1
        self.num_feats = num_channels
        self.num_freqs = num_freqs
        scales = 2.0 ** torch.linspace(self.min_deg, self.max_deg, self.num_freqs)
        # scales = torch.tensor([2**i for i in range(self.min_deg, self.max_deg)])
        int_scales = scales.long()
        features = []
        for i, scale in enumerate(int_scales):
            feature = torch.randn((2 * x_dim, num_channels, scale * res_scale, 1)) * std
            features.append(nn.Parameter(feature, True))
        self.register_buffer(
             "scales", scales
        )
        self.features = nn.ParameterList(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        if any([s == 0 for s in x.shape]):
            return torch.zeros((0, self.latent_dim)).to(x)

        # Nx1x3 * Fx1 => NxFx3 => NxF3
        num_scales = self.num_freqs

        xb = torch.reshape((x[Ellipsis, None, :] * self.scales[:, None]), list(x.shape[:-1]) + [num_scales, self.x_dim])
        latent = torch.sin(torch.stack([xb, xb + 0.5 * math.pi], dim=-2))
        # latent = torch.sin(torch.stack([xb, xb + 0.5 * math.pi], dim=-2)).unsqueeze(1).repeat(1, self.num_feats, 1, 1, 1).view(*x.shape[:-1], -1)
        # Fx23xN
        grid_latent = latent.view(-1, num_scales, 2 * self.x_dim).permute(1, 2, 0)
        zs = torch.zeros_like(grid_latent)
        # Fx23xNx2
        grid = torch.stack((zs, grid_latent), -1).unsqueeze(2)

        latents = []
        for i, scale in enumerate(self.scales):
            # 23xCx1xN + 23x1x1xN
            num_channels = self.features[i].shape[1]
            fs = grid_sample(self.features[i], grid[i], True).view(2, self.x_dim, num_channels, -1)
            latents.append(fs.permute(3, 2, 0, 1))

        # Fx2x3xCxN
        latent = (torch.stack(latents, 2) + latent.view(-1, 1, num_scales, 2, self.x_dim)).reshape(*x.shape[:-1], -1)
        return latent