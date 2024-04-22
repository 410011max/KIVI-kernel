#pragma once
#include <torch/extension.h>

torch::Tensor gemv_4low_rank(
    torch::Tensor _A,
    torch::Tensor _B
);