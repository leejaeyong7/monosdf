import torch
import torch.nn.functional as NF

def grid_sample_3d(vol, sample):
    FD, C, RD, RH, RW = vol.shape
    # FDx1x1xNx3
    _, _,_, N, _ = sample.shape
    sample = (sample.view(FD, 1, -1, 3).clamp(-1, 1) + 1) / 2 * (RW -1)

    tln = sample.floor().long()
    brf = sample.ceil().long()
    l = tln[..., 0]
    t = tln[..., 1]
    n = tln[..., 2]
    r = brf[..., 0]
    b = brf[..., 1]
    f = brf[..., 2]

    # compute weights
    w = sample - tln
    wx = w[..., 0]
    wy = w[..., 1]
    wz = w[..., 2]

    # left, top, near
    ltn = n * (RW * RH) + t * RW + l
    ltf = f * (RW * RH) + t * RW + l
    lbn = n * (RW * RH) + b * RW + l
    lbf = f * (RW * RH) + b * RW + l
    rtn = n * (RW * RH) + t * RW + r
    rtf = f * (RW * RH) + t * RW + r
    rbn = n * (RW * RH) + b * RW + r
    rbf = f * (RW * RH) + b * RW + r

    flat_vol = vol.view(FD, C, -1)
    fltn = flat_vol.gather(2, ltn.expand(-1, C, -1))
    fltf = flat_vol.gather(2, ltf.expand(-1, C, -1))
    flbn = flat_vol.gather(2, lbn.expand(-1, C, -1))
    flbf = flat_vol.gather(2, lbf.expand(-1, C, -1))
    frtn = flat_vol.gather(2, rtn.expand(-1, C, -1))
    frtf = flat_vol.gather(2, rtf.expand(-1, C, -1))
    frbn = flat_vol.gather(2, rbn.expand(-1, C, -1))
    frbf = flat_vol.gather(2, rbf.expand(-1, C, -1))

    # compute corners
    ftn = fltn * (1 - wx) + frtn * wx
    ftf = fltf * (1 - wx) + frtf * wx
    fbn = flbn * (1 - wx) + frbn * wx
    fbf = flbf * (1 - wx) + frbf * wx

    fn = ftn * (1 - wy) + fbn * wy
    ff = ftf * (1 - wy) + fbf * wy

    # FD x C
    return (fn * (1 - wz) + ff * wz).view(FD, -1, 1, N)
def grid_sample_2d(mat, sample):
    # Nx2
    FD, C, RH, RW = mat.shape
    _, _, N, _ = sample.shape
    sample = (sample.view(FD, 1, -1, 2).clamp(-1, 1) + 1) / 2 * (RW -1)
    tl = sample.floor().long()
    l = tl[..., 0]
    t = tl[..., 1]
    w = sample - tl
    br = sample.ceil().long()
    r = br[..., 0]
    b = br[..., 1]
    wx = w[..., 0]
    wy = w[..., 1]
    tl = t * RW + l
    tr = t * RW + r
    bl = b * RW + l
    br = b * RW + r
    ftl = mat.view(FD, C, -1).gather(2, tl.expand(-1, C, -1))
    ftr = mat.view(FD, C, -1).gather(2, tr.expand(-1, C, -1))
    fbl = mat.view(FD, C, -1).gather(2, bl.expand(-1, C, -1))
    fbr = mat.view(FD, C, -1).gather(2, br.expand(-1, C, -1))

    # compute corners
    # ftl = mat[:, :, t, l]
    # ftr = mat[:, :, t, r]
    # fbl = mat[:, :, b, l]
    # fbr = mat[:, :, b, r]
    ft = ftl * (1 - wx) + ftr * wx
    fb = fbl * (1 - wx) + fbr * wx

    # FD x C
    return (ft * (1 - wy) + fb * wy).view(FD, -1, 1, N)
def grid_sample_1d(vec, sample):
    FD, C, H, _ = vec.shape
    _, _, N, _ = sample.shape
    # FDxCxN => 
    sample = ((sample.view(FD, 1, -1, 2)[..., 1].clamp(-1, 1) + 1) / 2) * (H - 1)
    # vec: 1xFxRx1
    # sample: N
    s = sample.floor().long()
    e = sample.ceil().long()
    w = sample - s
    # 1xFxN
    fs = vec[..., 0].gather(2, s.expand(-1, C,-1))
    fe = vec[..., 0].gather(2, e.expand(-1, C,-1))
    # fs = vec[:, :, s, 0]
    # fe = vec[:, :, e, 0]
    return (fs * (1 - w) + fe * w).view(FD, -1, 1, N)

def grid_sample(features, grid, requires_hess=False):
    if not requires_hess:
        return NF.grid_sample(features, grid, align_corners=True, mode='bilinear')
    if features.shape[-1] == 1:
        return grid_sample_1d(features, grid)
    elif grid.shape[-1] == 2:
        return grid_sample_2d(features, grid)
    else:
        return grid_sample_3d(features, grid)