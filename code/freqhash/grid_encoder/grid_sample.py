import torch
from .grid_backward import grid_backward

# def grid_sample(input, grid):
#     return grid_sample_cuda.apply(input, grid)

def grid_sample_1d(vec, sample):
    FD, C, H, _ = vec.shape
    _, _, N, _ = sample.shape
    sample = ((sample.view(FD, 1, -1, 2)[..., 1].clamp(-1, 1) + 1) / 2) * (H - 1)
    s = sample.floor().long()
    e = sample.ceil().long()
    w = sample - s

    # 1xFxN
    fs = vec[..., 0].gather(2, s.expand(-1, C,-1))
    fe = vec[..., 0].gather(2, e.expand(-1, C,-1))
    return (fs * (1 - w) + fe * w).view(FD, -1, 1, N)

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
    ft = ftl * (1 - wx) + ftr * wx
    fb = fbl * (1 - wx) + fbr * wx

    # FD x C
    return (ft * (1 - wy) + fb * wy).view(FD, -1, 1, N)


def grid_sample(input, grid):
    if input.shape[-1] == 1:
        return grid_sample_1d(input, grid)
    # else:
    #     return grid_sample_2d(input, grid)
    return grid_sample_cuda.apply(input, grid)

class grid_sample_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        assert input.ndim == 4
        assert grid.ndim == 4
        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', align_corners=True)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = grid_backward.apply(grad_output, input, grid)
        return grad_input, grad_grid

# import torch
# from .grid_backward import grid_backward

# def grid_sample(input, grid):
#     return grid_sample_cuda.apply(input, grid)

# class grid_sample_cuda(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, grid):
#         assert input.ndim == 4
#         assert grid.ndim == 4
#         output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', align_corners=True)
#         ctx.save_for_backward(input, grid)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, grid = ctx.saved_tensors
#         grad_input, grad_grid = grid_backward.apply(grad_output, input, grid)
#         return grad_input, grad_grid
