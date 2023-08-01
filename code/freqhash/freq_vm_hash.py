import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
from .grid_encoder import grid_sample

class FreqVMEncoder(nn.Module):
    def __init__(self, res=128, num_freqs=6, num_channels=16, std=0.00001):
        super().__init__()
        min_log2_freq = 0
        max_log2_freq = num_freqs - 1
        freqs = 2.0 ** torch.linspace(min_log2_freq, max_log2_freq, num_freqs)
        self.freqs = nn.Parameter(freqs, False)

        cv = torch.randn((num_freqs * 2 * 3, num_channels, res, 1)) * std
        cm = torch.randn((num_freqs * 2 * 3, num_channels, res, res)) * std

        cv = nn.Parameter(cv, True)
        cm = nn.Parameter(cm, True)

        self.params = nn.ParameterDict({
            'cv': cv,
            'cm': cm,
        })

    def pos_encode(self, points):
        N = points.shape[0]
        # NF3
        freq_points = points.view(N, 1, -1) * self.freqs.to(points).view(1, -1, 1)

        # NxFx2x3 => Nx(Fx2x3) => F23xN
        return torch.stack((freq_points.sin(), freq_points.cos()), -2).view(N, -1).T.contiguous()

    def mat_inds(self, encs):
        N = encs.shape[-1]
        # 0:xs 1:ys 2:zs, 3:xc 4:yc 5:zc
        encs = encs.view(-1, 6, N)

        # Fx2xN
        yszs = encs[:, [1, 2]]
        zsxs = encs[:, [2, 0]]
        xsys = encs[:, [0, 1]]
        yczc = encs[:, [4, 5]]
        zcxc = encs[:, [5, 3]]
        xcyc = encs[:, [3, 4]]


        # Fx6x2xN
        return torch.stack([yszs, zsxs, xsys, yczc, zcxc, xcyc], 1).permute(0, 1, 3, 2).view(-1, 1, N, 2).contiguous()

    def encoder(self, encs, oenc, mat_grid):
        # (Fx2x3)xN
        N = encs.shape[-1]
        # (Fx2x3)xCxRx1

        # sampling vector
        cv = self.params['cv']
        C = cv.shape[1]
        encs = encs.view(-1, 1, N, 1)
        w = torch.zeros_like(encs)
        grid = torch.cat((w, encs), -1)

        # (Fx2x3)xCx1xN
        vec_f = grid_sample(cv, grid).view(-1, 2, 3, C, N)
        cm = self.params['cm']

        # Fx2x3xCxN
        mat_f = grid_sample(cm, mat_grid).view(-1, 2, 3, C, N)

        # Fx2x3xCxN
        # basis = self.params['basis']
        fs = (vec_f * mat_f)
        fs = fs + oenc.view(-1, 2, 3, 1, N)

        return fs.permute(4, 3, 0, 1, 2).reshape(N, -1)


    def forward(self, points, scale):
        """
        must return features given points (and optional dirs)
        """
        enc = self.pos_encode(points / scale * math.pi)
        oenc = self.pos_encode(points)
        grid = self.mat_inds(enc)
        return self.encoder(enc, oenc, grid)


