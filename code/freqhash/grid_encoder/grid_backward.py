import enum
from math import ceil
from cachetools import cached
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 

from .backend import _backend

class grid_backward(Function):
    @staticmethod
    def forward(ctx, grad_output, features, grids):
        N, C, IH, IW = features.shape
        N, OH, OW, _ = grids.shape

        grad_features = torch.zeros_like(features)
        grad_grids = torch.zeros_like(grids)
        # NxOHxOWx4x2 => 4 coords, 2 directions
        dy_dx__df = torch.zeros((N, C, IH, IW), device=grids.device, dtype=grids.dtype)

        # NxOHxOWx4x2 => 4 coords
        dy_dx = torch.zeros((N, C, OH, OW, 2), device=grids.device, dtype=grids.dtype)

        ctx.save_for_backward(grad_output, features, grids, dy_dx, dy_dx__df, grad_features, grad_grids)
        ctx.dims = [N, C, IH, IW, OH, OW]

        _backend.grid_backward(grad_output, features, grids, dy_dx, dy_dx__df, grad_features, grad_grids, N, C, IH, IW, OH, OW)
        
        return grad_features / 2, grad_grids

    @staticmethod
    def backward(ctx, grad_grad_inputs, grad_grad_embeddings):
        grad_output, features, grids, dy_dx, dy_dx__df, grad_features, grad_grids = ctx.saved_tensors
        N, C, IH, IW, OH, OW = ctx.dims
        grad_grad = dy_dx.sum(-1)

        # NxCxIHxIW
        grad2_features = dy_dx__df

        return grad_grad, grad2_features, None

