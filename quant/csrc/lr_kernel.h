#pragma once
#include <torch/extension.h>

torch::Tensor lr_kernel_cuda(
    torch::Tensor _A,
    torch::Tensor _B,
    torch::Tensor _C
);

