import math
import torch
import torch.nn as nn

from .grid_encoder import grid_sample


class FreqHash(nn.Module):
    def __init__(self, res=128, num_freqs=6, num_channels=48, std=0.001):
        super().__init__()
        min_log2_freq = 0
        max_log2_freq = num_freqs - 1
        freqs = 2.0 ** torch.linspace(min_log2_freq, max_log2_freq, num_freqs)
        self.freqs = nn.Parameter(freqs, False)
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
        fs = grid_sample(cv, grid).view(F, 2, 3, C, N)
        fs = fs + oencs.view(F, 2, 3, 1, N)
        # fs = encs.view(F, 2, 3, 1, N).repeat(1, 1, 1, C, 1)
        return fs.permute(4, 3, 0, 1, 2).reshape(N, -1)

    def forward(self, points, scale):
        """
        must return features given points (and optional dirs)
        """
        enc = self.pos_encode(points / scale * math.pi)
        oenc = self.pos_encode(points)
        return self.encoder(enc, oenc)

