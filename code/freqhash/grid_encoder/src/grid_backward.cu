#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <algorithm>
#include <stdexcept>

#include <stdint.h>
#include <cstdio>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")


template <typename T>
static inline __host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


// backward backward

__global__ void kernel_grid_backward_backward(
    const float * __restrict__ grad_grad_grid,   // NxOHxOWx2
    const float * __restrict__ grad_outputs,     // NxCxOHxOW
    const float * __restrict__ features,         // NxCxIHxIW
    const float * __restrict__ grid,             // NxOHxOWx2
    float * __restrict__ grad2_grad,             // NxCxOHxOW
    float * __restrict__ grad2_features,         // NxCxIHxIW
    const uint32_t N, const uint32_t C, 
    const uint32_t IH, const uint32_t IW, 
    const uint32_t OH, const uint32_t OW) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b>= N*OH*OW) return;

    // obtain index of the thread
    const uint32_t n = b / (OH * OW);
    const uint32_t oh = (b / OW) % OH;
    const uint32_t ow = b % OW;

    // skip to the corresponding grids
    grad_grad_grid = grad_grad_grid + (n * OH * OW + oh* OW  + ow) * 2;
    grid = grid + (n * OH * OW + oh* OW  + ow) * 2;

    grad_outputs = grad_outputs + (n * C * OH * OW) + oh * OW + ow;
    grad2_grad = grad2_grad + (n * C * OH * OW) + oh * OW + ow;

    features = features + n * C * IH * IW;
    grad2_features = grad2_features + n * C * IH * IW;

    const float ggx = grad_grad_grid[0];
    const float ggy = grad_grad_grid[1];
    const float gx = (grid[0] + 1) / 2.0 * (IW - 1);
    const float gy = (grid[1] + 1) / 2.0 * (IH - 1);
    if((gx < 0) || (gx > (IW - 1)) || (gy < 0) || (gy > (IH - 1)) || ((ggx == 0) && (ggy == 0))){
        return;
    }

    const uint32_t x0 = (uint32_t)floor(gx);
    const uint32_t y0 = (uint32_t)floor(gy);
    const uint32_t x1 = (uint32_t)ceil(gx);
    const uint32_t y1 = (uint32_t)ceil(gy);
    const float wx = (gx - (float)x0);
    const float wy = (gy - (float)y0);

    const float sx = (float)(IW - 1) / 2.0 * ggx;
    const float sy = (float)(IH - 1) / 2.0 * ggy;

    // pre-compute values
    const uint32_t c00 = y0 * IW + x0;
    const uint32_t c01 = y0 * IW + x1;
    const uint32_t c10 = y1 * IW + x0;
    const uint32_t c11 = y1 * IW + x1;

    for(uint32_t c = 0; c < C; c++){
        const uint32_t o00  = c * IH * IW + c00;
        const uint32_t o01  = c * IH * IW + c01;
        const uint32_t o10  = c * IH * IW + c10;
        const uint32_t o11  = c * IH * IW + c11;
        const float go = (grad_outputs + c * OH * OW)[0];

        atomicAdd(grad2_features + o00, go * (-(1 - wy) * sx - (1 - wx) * sy));
        atomicAdd(grad2_features + o01, go * ((1 - wy) * sx - wx * sy));
        atomicAdd(grad2_features + o10, go * (-wy * sx + (1 - wx) * sy));
        atomicAdd(grad2_features + o11, go * (wy * sx + wx * sy));

        // compute dy_dx
        const float f00 = features[o00];
        const float f01 = features[o01];
        const float f10 = features[o10];
        const float f11 = features[o11];

        const float g0x = -f00 * (1 - wy) + f01 * (1 - wy) - f10 * (wy)     + f11 * wy;
        const float g0y = -f00 * (1 - wx) - f01 * wx       + f10 * (1 - wx) + f11 * wx;

        const float dgx = g0x * sx;
        const float dgy = g0y * sy;

        grad2_grad[c * OH * OW] = dgx + dgy;
    }
}

void grid_backward_backward_cuda(
    const float *grad_grad_grids, 
    const float *grad_outputs, 
    const float *features, 
    const float *grid, 
    float * grad2_grad, 
    float * grad2_feats, 
    const uint32_t N, const uint32_t C, 
    const uint32_t IH, const uint32_t IW, 
    const uint32_t OH, const uint32_t OW) {

    static constexpr uint32_t N_THREAD = 256;
	const dim3 blocks_hashgrid = { div_round_up(N * OH * OW, N_THREAD), 1, 1 };
    kernel_grid_backward_backward<<<blocks_hashgrid, N_THREAD>>>(
        grad_grad_grids,
        grad_outputs,
        features,
        grid,
        grad2_grad,
        grad2_feats,
        N, C, IH, IW, OH, OW
    ); 
}

void grid_backward_backward(
    const at::Tensor grad_grad_grid, 
    const at::Tensor grad_outputs, 
    const at::Tensor features, 
    const at::Tensor grid, 
    at::Tensor grad2_grad, 
    at::Tensor grad2_features, 
    const uint32_t N, const uint32_t C, 
    const uint32_t IH, const uint32_t IW, 
    const uint32_t OH, const uint32_t OW
){

    CHECK_CUDA(grad_grad_grid);
    CHECK_CUDA(grad_outputs);
    CHECK_CUDA(features);
    CHECK_CUDA(grid);
    CHECK_CUDA(grad2_grad);
    CHECK_CUDA(grad2_features);

    CHECK_CONTIGUOUS(grad_grad_grid);
    CHECK_CONTIGUOUS(grad_outputs);
    CHECK_CONTIGUOUS(features);
    CHECK_CONTIGUOUS(grid);
    CHECK_CONTIGUOUS(grad2_grad);
    CHECK_CONTIGUOUS(grad2_features);

    grid_backward_backward_cuda(
        grad_grad_grid.data_ptr<float>(), 
        grad_outputs.data_ptr<float>(), 
        features.data_ptr<float>(), 
        grid.data_ptr<float>(), 
        grad2_grad.data_ptr<float>(), 
        grad2_features.data_ptr<float>(), 
        N, C, IH, IW, OH, OW
    );
}
