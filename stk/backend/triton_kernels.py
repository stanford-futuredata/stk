import torch

import triton
import triton.language as tl

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            row_indices, column_indices,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, BLOCK_SIZE: tl.constexpr, ACC_TYPE: tl.constexpr, 
            ):
    # matrix multiplication
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # do matrix multiplication
    pid_m = row_indices.load(pid)
    pid_n = column_indices.load(pid)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE
    C = C + pid * BLOCK_ELEMENTS + tl.arange(BLOCK_ELEMENTS)
    # mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc)


class _matmul(torch.autograd.Function):
    kernel = _kernel

    _locks = {}

    @staticmethod
    def _call(a, b):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # allocates output
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        row_indices = topo.row_indices
        column_indices = topo.column_indices
        # accumulator types
        ACC_TYPE = tl.float32 if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']))
        _kernel[grid](a, b, c, M, N, K,
                      a.stride(0), a.stride(1),
                      b.stride(0), b.stride(1),
                      c.stride(0), c.stride(1),
                      row_indices, column_indices,
                      GROUP_M=8, ACC_TYPE=ACC_TYPE)
        return c

    @staticmethod
    def forward(ctx, a, b, topo):
        return _matmul._call(a, b, topo)

matmul = _matmul.apply