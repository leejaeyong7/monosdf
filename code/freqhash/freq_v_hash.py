import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
from .grid_encoder import grid_sample

class FreqVEncoder(nn.Module):
    def __init__(self, log2_res=4, num_freqs=6, num_channels=16, std=0.2):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.freqs = nn.Parameter(freqs, False)

        res = 2 ** log2_res
        cv = torch.randn((num_freqs * 2 * 8, num_channels, res, res, res)) * std

        self.cv = nn.Parameter(cv, True)

    def pos_encode(self, points):
        N = points.shape[0]
        # NF3
        freq_points = points.view(N, 1, -1) * self.freqs.to(points).view(1, -1, 1)

        # NxFx2x3 => Nx(Fx2x3) => F23xN
        return torch.stack((freq_points.sin(), freq_points.cos()), -2).view(N, -1, 2, 3)

    def vol_inds(self, encs):
        # NxF
        N = encs.shape[0]
        F = self.num_freqs
        xs = encs[..., 0, 0]
        ys = encs[..., 0, 1]
        zs = encs[..., 0, 2]
        xc = encs[..., 1, 0]
        yc = encs[..., 1, 1]
        zc = encs[..., 1, 2]

        # NxFxFxFx3 => NxFFFx3
        sss = torch.stack((xs, ys, zs), -1).view(N, F, 3)
        ssc = torch.stack((xs, ys, zc), -1).view(N, F, 3)
        scs = torch.stack((xs, yc, zs), -1).view(N, F, 3)
        scc = torch.stack((xs, yc, zc), -1).view(N, F, 3)
        css = torch.stack((xc, ys, zs), -1).view(N, F, 3)
        csc = torch.stack((xc, ys, zc), -1).view(N, F, 3)
        ccs = torch.stack((xc, yc, zs), -1).view(N, F, 3)
        ccc = torch.stack((xc, yc, zc), -1).view(N, F, 3)

        # Nx8xFx3 => F8x1x1xNx3
        # NxFFFx8x3
        return torch.stack((sss, ssc, scs, scc, css, csc, ccs, ccc), 2).view(N, F, 8, 3).permute(1, 2, 0, 3).reshape(-1, 1, 1, N, 3)

    def encoder(self, encs, grid, add, requires_hess=False):
        cv = self.cv
        C = cv.shape[1]
        vol_f = grid_sample(cv, grid).view(-1, C, N)

        # (Fx2x3)xN

        if add:
            fs = fs + encs.view(-1, 2, 2, 1, N)

        return fs.permute(4, 3, 0, 1, 2).reshape(N, -1)


    def forward(self, points):
        """
        must return features given points (and optional dirs)
        """
        enc = self.pos_encode(points)
        grid = self.vol_inds(encs)
        af = self.encoder(enc, grid, True, True)

        return af
