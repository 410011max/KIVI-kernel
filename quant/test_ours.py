import torch
import kivi_gemv 
from timeit_v2 import py_benchmark


def main():
    A = torch.randn(1, 128).cuda()
    B = torch.randn(4096, 128).cuda()
    
    #C = kivi_gemv.gemv_4low_rank(A.half(), B.half())
    stmt = "kivi_gemv.gemv_4low_rank(A.half(), B.half())"
    t_kivi = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=3,
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
    print(f'ours: {t_kivi * 1000} ms')
    
    stmt = "torch.matmul(A.half(), B.half().t())"
    t_torch = py_benchmark(stmt, {**globals(), **locals()}, min_repeat_second=3,
                                     setup="torch.cuda.synchronize()", finish="torch.cuda.synchronize()")
    print(f'Torch matmul: {t_torch * 1000} ms')

    #print((C_ours-C).mean())
if __name__ == '__main__':
    main()