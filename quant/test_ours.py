import torch
import random
import numpy as np
from timeit_v2 import py_benchmark
import kivi_gemv



def test_gemm_correct():
	M, N, K = 128, 128, 64
	A = torch.randn((M, K), device='cuda', dtype=torch.float16)
	B = torch.randn((N, K), device='cuda', dtype=torch.float16)

	output_ref = A @ B.T
	output = kivi_gemv.wmma_base_ours_cuda(A, B)
	
	error = output_ref - output
	rel_out_error = torch.abs(error.float() / (output_ref + 1e-5).float()).mean()
	print(f'GEMM ({M}, {N}, {K}) avg out error: {rel_out_error:.5f}\n')


def test_gemm_speed():
	M, N, K = 128, 128, 64
	A = torch.randn((M, K), device='cuda', dtype=torch.float16)
	B = torch.randn((N, K), device='cuda', dtype=torch.float16)	

	# Pytorch
	stmt = "A @ B.T"
	t_ref = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

	# CUDA
	stmt = "kivi_gemv.wmma_base_ours_cuda(A, B)"
	t_our = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
	
	print(f'GEMM ({M}, {N}, {K}) Speed Test:')
	print(f'pytorch gemm: {t_ref * 1000:.3f} ms')
	print(f'our gemm: {t_our * 1000:.3f} ms\n')


def test_gemv_correct():
	M, N, K = 1, 4096, 128
	A = torch.randn((M, K), device='cuda', dtype=torch.float16)
	B = torch.randn((N, K), device='cuda', dtype=torch.float16)	

	output_ref = A @ B.T
	output = kivi_gemv.gemv_4low_rank(A, B)
	
	error = output_ref - output
	rel_out_error = torch.abs(error.float() / (output_ref + 1e-5).float()).mean()
	print(f'GEMV ({M}, {N}, {K}) avg out error: {rel_out_error:.5f}\n')


def test_gemv_speed():
	M, N, K = 1, 4096, 128
	A = torch.randn((M, K), device='cuda', dtype=torch.float16)
	B = torch.randn((N, K), device='cuda', dtype=torch.float16)	

	# Pytorch
	stmt = "A @ B.T"
	t_ref = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

	# CUDA
	stmt = "kivi_gemv.gemv_4low_rank(A, B)"
	t_our = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1, 
                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
	
	print(f'GEMV ({M}, {N}, {K}) Speed Test:')
	print(f'pytorch gemm: {t_ref * 1000:.3f} ms')
	print(f'our gemm: {t_our * 1000:.3f} ms\n')


def test_lr_kernel_correct():
	M, N, K = 128, 128, 64
	A = torch.randn((M, K), device='cuda', dtype=torch.float16)
	B = torch.randn((N, K), device='cuda', dtype=torch.float16)
	Q = torch.randn((1, N), device='cuda', dtype=torch.float16)

	output_ref = Q @ (A @ B.T).T
	C = kivi_gemv.wmma_base_ours_cuda(A, B)
	output = kivi_gemv.gemv_4low_rank(Q, C)
	# output = kivi_gemv.lr_kernel_ours_cuda(A, B, Q)
 
	error = output_ref - output
	rel_out_error = torch.abs(error / (output_ref + 1e-5)).mean()
	print(f'avg out error: {rel_out_error:.5f}\n')


def test_lr_kernel_speed():
	M, N, K = 128, 128, 64
	A = torch.randn((M, K), device='cuda', dtype=torch.float16)
	B = torch.randn((N, K), device='cuda', dtype=torch.float16)
	Q = torch.randn((1, N), device='cuda', dtype=torch.float16)

	# Pytorch
	stmt = "Q @ (A @ B.T).T"
	t_ref = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")

	# CUDA
	stmt = "kivi_gemv.lr_kernel_ours_cuda(A, B, Q)"
	t_our = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=1,
                      setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
	print(f'vanilla pytorch gemm: {t_ref * 1000:.3f} ms')
	print(f'our {M}_{N}_{K} gemm: {t_our * 1000:.3f} ms\n')


if __name__ == "__main__":
	torch.manual_seed(0)
	np.random.seed(0)
	random.seed(0)

	test_gemm_correct()
	test_gemm_speed()
 
	test_gemv_correct()
	test_gemv_speed()
	
	test_lr_kernel_correct()
	# test_lr_kernel_speed()
	