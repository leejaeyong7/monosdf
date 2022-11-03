#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

void grid_backward(
    const at::Tensor grad_outputs, 
    const at::Tensor features, 
    const at::Tensor grid, 
    at::Tensor dy_dx, 
    at::Tensor dy_dx__df, 
    at::Tensor grad_features, 
    at::Tensor grad_grid, 
    const uint32_t N, const uint32_t C, 
    const uint32_t IH, const uint32_t IW, 
    const uint32_t OH, const uint32_t OW);

#endif