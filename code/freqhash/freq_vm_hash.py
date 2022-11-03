import torch
import torch.nn as nn
import torch.nn.functional as NF
import math
from .grid_encoder import grid_sample

class FreqVMEncoder(nn.Module):
    def __init__(self, log2_res=4, num_freqs=6, num_channels=16, std=0.001):
        super().__init__()
        min_log2_freq = 0
        max_log2_freq = num_freqs - 1
        freqs = 2.0 ** torch.linspace(min_log2_freq, max_log2_freq, num_freqs)
        self.freqs = nn.Parameter(freqs, False)

        res = 2 ** log2_res
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
        vec_f = grid_sample(cv, grid, True).view(-1, 2, 3, C, N)
        cm = self.params['cm']

        # Fx2x3xCxN
        mat_f = grid_sample(cm, mat_grid, True).view(-1, 2, 3, C, N)

        # Fx2x3xCxN
        # basis = self.params['basis']
        fs = (vec_f * mat_f)
        fs = fs + oenc.view(-1, 2, 3, 1, N)

        return fs.permute(4, 3, 0, 1, 2).reshape(N, -1)


    def forward(self, points, scale):
        """
        must return features given points (and optional dirs)
        """
        enc = self.pos_encode(points / scale)
        oenc = self.pos_encode(points)
        grid = self.mat_inds(enc)
        return self.encoder(enc, oenc, grid)



class MultiFreqVMEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, res_scale=4, num_freqs=6, num_channels=8, std=0.001, x_dim=3):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = 0
        self.max_deg = num_freqs - 1
        self.num_feats = num_channels
        self.num_freqs = num_freqs
        scales = 2.0 ** torch.linspace(self.min_deg, self.max_deg, self.num_freqs)
        int_scales = scales.long()
        features_2d = []
        features = []
        for i, scale in enumerate(int_scales):
            scale = 1
            feature = torch.randn((2 * x_dim, num_channels, scale * res_scale, scale * res_scale)) * std
            feature_2d = torch.randn((2 * x_dim, num_channels, scale * res_scale, scale * res_scale)) * std
            features.append(nn.Parameter(feature, True))
            features_2d.append(nn.Parameter(feature_2d, True))
        self.register_buffer(
             "scales", scales
        )
        self.features = nn.ParameterList(features)
        self.features_2d = nn.ParameterList(features_2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        # Nx1x3 * Fx1 => NxFx3 => NxF3
        num_scales = self.num_freqs

        # NxFx3
        xb = torch.reshape((x[Ellipsis, None, :] * self.scales[:, None]), list(x.shape[:-1]) + [num_scales, self.x_dim])

        # NxFx2x3
        latent = torch.sin(torch.stack([xb, xb + 0.5 * math.pi], dim=-2)).view(-1, num_scales, 2, self.x_dim)

        # create grid for 1d

        # NxFx6= > Fx6xN
        grid_latent = latent.view(-1, num_scales, 2 * self.x_dim).permute(1, 2, 0)
        zs = torch.zeros_like(grid_latent)
        grid = torch.stack((zs, grid_latent), -1).unsqueeze(2)

        # create grid for 2d
        # NxFx2x3 => 6[NxFx1x2] => NxFx6x2 => Fx6xNx2
        grid_2d = torch.stack([
            latent[..., 0, [1, 2]],
            latent[..., 0, [2, 0]],
            latent[..., 0, [0, 1]],
            latent[..., 1, [1, 2]],
            latent[..., 1, [2, 0]],
            latent[..., 1, [0, 1]],
        ], 2).permute(1, 2, 0, 3).unsqueeze(2).contiguous()

        latents = []
        for i, scale in enumerate(self.scales):
            num_channels = self.features[i].shape[1]
            fs = grid_sample(self.features[i], grid[i], True).view(2, self.x_dim, num_channels, -1)
            fs_2d = grid_sample(self.features_2d[i], grid_2d[i], True).view(2, self.x_dim, num_channels, -1)
            # 2x3xCxN => NxCx2x3
            latents.append((fs * fs_2d).permute(3, 2, 0, 1))

        # NxCxFx2x3 + Nx1xFx2x3 => Nx(CF23)
        latent = (torch.stack(latents, 2) + latent.view(-1, 1, num_scales, 2, self.x_dim)).reshape(*x.shape[:-1], -1)

        return latent
