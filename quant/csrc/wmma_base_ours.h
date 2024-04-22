#pragma once
#include <torch/extension.h>

torch::Tensor wmma_base_ours_cuda(
    torch::Tensor _A,
    torch::Tensor _B
);

