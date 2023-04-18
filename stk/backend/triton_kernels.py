import torch
import triton
import triton.language as tl
from stk import Matrix
from triton.language import int16

@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _sdd_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            row_indices, column_indices, 
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
            ):
    # matrix multiplication
    pid = tl.program_id(0)   
    pid_m = tl.load(row_indices + pid)
    pid_n = tl.load(column_indices + pid)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    #Store to sparse matrix
    acc = acc.to(C.dtype.element_ty)
    cm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = tl.arange(0, BLOCK_N)
    C = C + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    # mask = (cm < M)[:, None] & (cn < N)[None, :]
    tl.store(C, acc, mask=True)

@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _dsd_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            row_indices, column_indices, offsets,
            block_offsets_t, trans_A: tl.constexpr, trans_B: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
            ):

    # matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_M)
    
    start_inx = tl.load(offsets + pid_m)
    end_inx = tl.load(offsets + pid_m + 1)

    pid = pid_m * 2 + pid_n

    # pointers to sparse matrix
    rm =  tl.arange(0, BLOCK_M)
    rak = tl.arange(0, BLOCK_K)
    # rm = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)

    A += (rm[:, None] * stride_am + rak[None, :] * stride_ak)

    # pointers to dense matrix
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rbk = tl.arange(0, BLOCK_K)
    # rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    B += (rbk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    nsub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_K)

    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE
    ak_sub_incr = BLOCK_K * stride_ak
    bk_sub_incr = BLOCK_K * stride_bk
    bk_block_incr = BLOCK_SIZE * stride_bk

    # ak_sub_incr = tl.multiple_of(ak_sub_incr, BLOCK_K)
    # bk_sub_incr = tl.multiple_of(bk_sub_incr, BLOCK_K)
    # bk_block_incr = tl.multiple_of(bk_block_incr, BLOCK_SIZE)

    for k in range(nsub_blocks * (end_inx - start_inx)):
        sub_block_inx = k % nsub_blocks
        block_inx = k // nsub_blocks
        # mask = sub_block_inx == 0

        if trans_A:
            ptr_A = A + tl.load(block_offsets_t + start_inx + block_inx) * BLOCK_ELEMENTS + sub_block_inx * ak_sub_incr
        else:
            ptr_A = A + (start_inx + block_inx) * BLOCK_ELEMENTS + sub_block_inx * ak_sub_incr

        ptr_B = B + tl.load(column_indices + start_inx + block_inx) * bk_block_incr + sub_block_inx * bk_sub_incr

        # tl.debug_barrier()
        # prev_offset_b = offset_b
        # tl.store(debug + pid * 8 + k, offset_b)

        a = tl.load(ptr_A)
        b = tl.load(ptr_B)
        acc += tl.dot(a, b) 

    acc = acc.to(C.dtype.element_ty)

    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # if trans_C:
    #     C = C + (cn[:, None] * stride_cm + cm[None, :] * stride_cn)
    # else:
    C = C + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    tl.store(C, acc, mask=True)

@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _dds_kernel(A, B, C, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            row_indices, column_indices, offsets,
            block_offsets_t, trans_A: tl.constexpr, trans_B: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
            ):

    # matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_M)
    
    start_inx = tl.load(offsets + pid_n)
    end_inx = tl.load(offsets + pid_n + 1)

    # pointers to dense matrix
    rm =  pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rak = tl.arange(0, BLOCK_K)

    A += (rm[:, None] * stride_am + rak[None, :] * stride_ak)

    # pointers to sparse matrix
    rn = tl.arange(0, BLOCK_N)
    rbk = tl.arange(0, BLOCK_K)
    B += (rbk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    nsub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_K)

    BLOCK_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE
    
    ak_sub_incr = BLOCK_K * stride_ak
    ak_block_incr = BLOCK_SIZE * stride_ak
    bk_sub_incr = BLOCK_K * stride_bk

    for k in range(nsub_blocks * (end_inx - start_inx)):
        sub_block_inx = k % nsub_blocks
        block_inx = k // nsub_blocks

        if trans_B:
            ptr_B = B + (start_inx + block_inx) * BLOCK_ELEMENTS + sub_block_inx * bk_sub_incr
        else:
            ptr_B = B + tl.load(block_offsets_t + start_inx + block_inx) * BLOCK_ELEMENTS + sub_block_inx * bk_sub_incr

        ptr_A = A + tl.load(column_indices + start_inx + block_inx) * ak_block_incr + sub_block_inx * ak_sub_incr
        a = tl.load(ptr_A)
        b = tl.load(ptr_B)
        acc += tl.dot(a, b) 

    acc = acc.to(C.dtype.element_ty)
    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    tl.store(C, acc, mask=True)

class _sdd(torch.autograd.Function):

    @staticmethod
    def _call(a, b, topo):
        device = a.device
        trans_A = False
        trans_B = False
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            trans_A = True
        if b.stride(0) > 1 and b.stride(1) > 1:
            trans_B = True
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # allocates output
        nnz_blocks = len(topo.row_indices)
        c = torch.empty((nnz_blocks * topo.blocking, topo.blocking), device=device, dtype=a.dtype, requires_grad=True)

        # accumulator types
        ACC_TYPE = tl.float32 if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # launch kernel
        grid = lambda META: (nnz_blocks,)

        stride_am, stride_ak = a.stride(0), a.stride(1)
        stride_bk, stride_bn = b.stride(0), b.stride(1)

        if trans_A:
            stride_am, stride_ak = a.stride(1), a.stride(0)
        if trans_B:
            stride_bk, stride_bn = b.stride(1), b.stride(0) 

        _sdd_kernel[grid](
            a, b, c, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            c.stride(0), c.stride(1),
            topo.row_indices, topo.column_indices,
            GROUP_M=8, ACC_TYPE=ACC_TYPE
            )
                
        return Matrix(
            size = topo.size(), 
            data = c.reshape((-1, topo.blocking, topo.blocking)),
            row_indices = topo.row_indices,
            column_indices = topo.column_indices,
            offsets = topo.offsets,
            column_indices_t = topo.column_indices_t,
            offsets_t = topo.offsets_t,
            block_offsets_t = topo.block_offsets_t
            )

    @staticmethod
    def forward(ctx, a, b, topo):
        return _sdd._call(a, b, topo)

class _dsd(torch.autograd.Function):

    @staticmethod
    def _call(a, b):
        device = a.device
        trans_A = False
        trans_B = False

        # handle non-contiguous inputs if necessary
        if not a.is_contiguous():
            trans_A = True
        if b.stride(0) > 1 and b.stride(1) > 1:
            trans_B = True

        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        
        # allocates output
        c = torch.empty((M, N), device=device, dtype=a.dtype, requires_grad=True)
        
        # accumulator types
        ACC_TYPE = tl.float32 if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32

        stride_am, stride_ak = a.data.stride(1), a.data.stride(2)
        stride_bk, stride_bn = b.stride(0), b.stride(1)
        row_indices, column_indices  = a.row_indices, a.column_indices
        offsets, block_offsets_t = a.offsets, a.block_offsets_t

        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

        debug = torch.empty((4 * 8), device=device)
        if trans_A:
            stride_am, stride_ak = a.data.stride(2), a.data.stride(1)
            column_indices, offsets = a.column_indices_t, a.offsets_t

        if trans_B:
            stride_bk, stride_bn = b.stride(1), b.stride(0) 

        binary = _dsd_kernel[grid](
            a.data, b, c, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            c.stride(0), c.stride(1),
            row_indices, column_indices, offsets,
            block_offsets_t, trans_A, trans_B,
            GROUP_M=128, ACC_TYPE=ACC_TYPE
        )
        # print(debug)
        return c

    @staticmethod
    def forward(ctx, a, b, topo=None):
        return _dsd._call(a, b)

class _dds(torch.autograd.Function):
    
    @staticmethod
    def _call(a, b):

        device = a.device
        trans_A = False
        trans_B = False

        # handle non-contiguous inputs if necessary
        if not b.is_contiguous():
            trans_B = True
        if a.stride(0) > 1 and a.stride(1) > 1:
            trans_A = True

        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        
        # allocates output
        c = torch.empty((M, N), device=device, dtype=a.dtype, requires_grad=True)
        
        # accumulator types
        ACC_TYPE = tl.float32 if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32

        stride_am, stride_ak = a.stride(0), a.data.stride(1)
        stride_bk, stride_bn = b.data.stride(1), b.data.stride(2)
        row_indices, column_indices  = b.row_indices, b.column_indices_t
        offsets, block_offsets_t = b.offsets_t, b.block_offsets_t

        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

        debug = torch.empty((4 * 8), device=device)
        if trans_B:
            stride_bk, stride_bn = b.data.stride(2), b.data.stride(1)
            column_indices, offsets = b.column_indices, b.offsets

        if trans_A:
            stride_am, stride_ak = a.stride(1), a.stride(0) 

        binary = _dds_kernel[grid](
            a, b.data, c, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            c.stride(0), c.stride(1),
            row_indices, column_indices, offsets,
            block_offsets_t, trans_A, trans_B,
            GROUP_M=128, ACC_TYPE=ACC_TYPE
        )
        # print(debug)
        return c

    @staticmethod
    def forward(ctx, a, b, topo=None):
        return _dds._call(a, b)

sdd = _sdd.apply
dsd = _dsd.apply
dds = _dds.apply

def matmul(mode, a, b, topo=None):
    if mode not in ['sdd', 'dsd', 'dds']:
        raise NotImplementedError(f'Mode {mode} is not implemented. Supported modes are: sdd, dsd, dds')
    fn = {'sdd': _sdd.apply, 'dsd':_dsd.apply, 'dds':_dds.apply}
    if mode == 'sdd':
        assert topo is not None, 'Mode "sdd" requires topo parameter'    
    fn[mode](a, b, topo)


