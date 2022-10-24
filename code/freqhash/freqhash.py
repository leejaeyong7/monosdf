import torch
import torch.nn as nn
from .grid_sample import grid_sample

class FreqHash(nn.Module):
    def __init__(self, log2_res=5, num_encodings=6, num_feats=16, std=0.2):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0, num_encodings - 1, num_encodings)
        res = 2 ** log2_res
        features = torch.randn((num_encodings * 2 * 3, num_feats, res, 1)) * std
        self.freqs = nn.Parameter(freqs, False)
        self.features = nn.Parameter(features, True)
        self.num_freqs = num_encodings

    def forward(self, points):
        '''
        enc_pos: BxNx2x3 features
        '''
        enc_pos = self.pos_encode(points)
        return self.encoder(enc_pos)

    def encoder(self, encs):
        # (Fx2x3)xN
        F = self.num_freqs
        N = encs.shape[-1]
        g_f = self.features
        C = g_f.shape[1]
        encs = encs.view(-1, 1, N, 1)
        w = torch.zeros_like(encs)
        grid = torch.cat((w, encs), -1)

        # (Fx2x3)xCx1xN
        fs = grid_sample(g_f, grid, True).view(F, -1, 2, C, N)
        fs = fs + encs.view(F, -1, 2, 1, N)
        return fs.permute(4, 3, 0, 1, 2).reshape(N, -1)

    def pos_encode(self, points):
        N = points.shape[0]
        freq_points = points.view(N, 1, -1) * self.freqs.to(points).view(1, -1, 1)

        # NxFx2x3 => Nx(Fx2x3)
        return torch.stack((freq_points.sin(), freq_points.cos()), -2).view(N, -1).T.contiguous()
