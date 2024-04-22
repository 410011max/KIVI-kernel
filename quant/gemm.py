import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import numpy as np
import torch
# import ipdb
import random
import triton
import triton.language as tl
from new_pack import pack_tensor
from timeit_v2 import py_benchmark
import kivi_gemv



def test_gemm_speed():
	M, N, K = 512, 128, 64
	A = torch.randn((M, K), device='cuda', dtype=torch.float16)
	B = torch.randn((K, N), device='cuda', dtype=torch.float16)	

	# Pytorch
	stmt = "A @ B"
	t_ref = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

	# CUDA
	stmt = "kivi_gemv.wmma_base_ours_cuda(A, B, M, N, K)"
	t_our = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
	print(f'vanilla pytorch gemm: {t_ref * 1000:.3f} ms')
	print(f'our {M}_{N}_{K} gemm: {t_our * 1000:.3f} ms')


if __name__ == "__main__":
	torch.manual_seed(0)
	np.random.seed(0)
	random.seed(0)
 
	test_gemm_speed()