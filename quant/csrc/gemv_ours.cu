// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:49:00 on Mon, Oct 09, 2023
//
// Description: warp16 smem hgemv

#include "common/common.h"
#include "gemv_ours.h"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

#define COLS_PER_WARP 16
#define COLS_PER_BLOCK 64    // COLS_PER_WARP * WARPS_PER_BLOCK
#define THREADS_PER_GROUP 2  // WARP_SIZE / COLS_PER_WARP

__global__ void warp16SmemKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t N,
                                 size_t K) {
    extern __shared__ half A_smem[];
    size_t A_smem_iters = div_ceil(K, THREADS_PER_BLOCK);
#pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        size_t idx = i * THREADS_PER_BLOCK + threadIdx.x;
        A_smem[idx] = A[idx];
    }

    __syncthreads();

    const size_t group_id = threadIdx.x / THREADS_PER_GROUP;
    const size_t group_col = blockIdx.x * COLS_PER_BLOCK + group_id;
    if (group_col >= N) {
        return;
    }

    const size_t K_iters = div_ceil(K, THREADS_PER_GROUP);
    const size_t group_lane_id = threadIdx.x % THREADS_PER_GROUP;

    float tmp = 0.0;
#pragma unroll
    for (size_t i = 0; i < K_iters; ++i) {
        size_t A_idx = i * THREADS_PER_GROUP + group_lane_id;
        size_t B_idx = i * THREADS_PER_GROUP + group_lane_id + group_col * K;
        tmp += __half2float(A_smem[A_idx]) * __half2float(B[B_idx]);
    }

    constexpr unsigned int mask = 0xffffffff;
#pragma unroll
    for (size_t i = THREADS_PER_GROUP / 2; i >= 1; i /= 2) {
        tmp += __shfl_xor_sync(mask, tmp, i);
    }

    if (group_lane_id == 0) {
        C[group_col] = __float2half(tmp);
    }
}

size_t initWarp16Smem(size_t K) {
    int dev_id = 0;
    HGEMV_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMV_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size = K * sizeof(half);
    //HLOG("smem_max_size: %.0f KBytes (%zu bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMV_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMV_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(warp16SmemKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void warp16Smem(half *A, half *B, half *C, size_t N, size_t K) {
    static size_t smem_max_size = initWarp16Smem(K);

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, COLS_PER_BLOCK));

    warp16SmemKernel<<<grid, block, smem_max_size>>>(A, B, C, N, K);
}


torch::Tensor gemv_4low_rank(
    torch::Tensor _A,
    torch::Tensor _B) 
{
    // Input
    auto A = reinterpret_cast<half*>(_A.data_ptr<at::Half>());
    auto B = reinterpret_cast<half*>(_B.data_ptr<at::Half>());

    int N = _B.size(0);
    int K = _B.size(1);

    //printf("Processing gemv_ours with N=%d, K=%d\n", N, K);
    // Output
    auto options = torch::TensorOptions().dtype(_A.dtype()).device(_A.device());
    at::Tensor _C = torch::empty({1, N}, options);
    auto C = reinterpret_cast<half*>(_C.data_ptr<at::Half>());

    // Kernel execution
    static size_t smem_max_size = initWarp16Smem(K);
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, COLS_PER_BLOCK));

    warp16SmemKernel<<<grid, block, smem_max_size>>>(A, B, C, N, K);
    return _C;
}
