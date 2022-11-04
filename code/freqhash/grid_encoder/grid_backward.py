import torch
from torch.autograd import Function
from .backend import _backend

class grid_backward(Function):
    @staticmethod
    def forward(ctx, grad_output, features, grids):
        N, C, IH, IW = features.shape
        N, OH, OW, _ = grids.shape

        grad_features = torch.zeros_like(features)
        grad_grids = torch.zeros_like(grids)

        dy_dx = torch.zeros((N, C, OH, OW, 2), device=grids.device, dtype=grids.dtype)

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        if not features.is_contiguous():
            features = features.contiguous()
        if not grids.is_contiguous():
            grids = grids.contiguous()

        ctx.save_for_backward(grad_output, grids, dy_dx)
        ctx.dims = [N, C, IH, IW, OH, OW]

        _backend.grid_backward(grad_output, features, grids, dy_dx, grad_features, grad_grids, N, C, IH, IW, OH, OW)
        
        return grad_features, grad_grids

    @staticmethod
    def backward(ctx, grad_grad_features, grad_grad_grids):
        grad_output, grids, dy_dx = ctx.saved_tensors
        N, C, IH, IW, OH, OW = ctx.dims
        grad_grad = (dy_dx * grad_grad_grids.unsqueeze(1)).sum(-1)

        if not grad_grad_grids.is_contiguous():
            grad_grad_grids = grad_grad_grids.contiguous()

        # NxCxIHxIW
        grad2_features = torch.zeros_like(grad_grad_features)
        _backend.grid_backward_backward(grad_output, grad_grad_grids, grids, grad2_features, N, C, IH, IW, OH, OW)

        return grad_grad, grad2_features, None


