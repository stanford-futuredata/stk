from stk.matrix import Matrix
import torch
import numpy as np


@torch.no_grad()
def _row_indices(nnz, offsets):
    # TODO(tgale): Update to use 1D COO helper.    
    out = np.digitize(np.arange(nnz), bins=offsets.cpu().numpy()) - 1
    return torch.from_numpy(out.astype(np.int32)).to(offsets.device)


@torch.no_grad()
def _unblock(indices, blocking):
    indices = torch.reshape(indices, [1, indices.numel(), 1])
    out = torch.tile(indices, (blocking, 1, blocking))
    offsets = torch.reshape(torch.arange(blocking), [1, 1, blocking])
    out += offsets
    return out.flatten()


# TODO(tgale): Add input type checking.
@torch.no_grad()
def to_dense(x):
    assert isinstance(x, Matrix)
    # TODO(tgale): Handle blocking.
    row_idxs = _row_indices(x.nnz, x.offsets)
    col_idxs = _unblock(x.indices, x.blocking)
    indices = (row_idxs * x.size()[1] + col_idxs).type(torch.int64)
    out = torch.zeros(x.size()[0] * x.size()[1], dtype=x.dtype, device=x.device)
    out.scatter_(0, indices, x.data.flatten())
    return out.reshape(x.size())


@torch.no_grad()
def mask(x, blocking=1):
    assert x.dim() == 2
    assert x.size()[0] % blocking == 0
    assert x.size()[1] % blocking == 0
    block_rows = x.size()[0] // blocking
    block_cols = x.size()[1] // blocking
    x = torch.reshape(x, [block_rows, blocking, block_cols, blocking])
    x = torch.sum(torch.abs(x), dim=(1, 3))
    return x != 0


# TODO(tgale): Add input type checking.
@torch.no_grad()
def to_sparse(x, blocking=1):
    m = mask(x)

    # TODO(tgale): Set to appropriate type for input matrix.    
    row_nnzs = torch.sum(m, dim=1).type(torch.int32)
    zeros = torch.zeros((1,), dtype=row_nnzs.dtype, device=row_nnzs.device)
    offsets = torch.cat([zeros, torch.cumsum(row_nnzs, dim=0)])
    offsets *= blocking * blocking
    offsets = offsets.type(torch.int32)
    
    indices = torch.nonzero(m)[:, 1].type(torch.int16)
    indices *= blocking
    
    m = torch.reshape(m, [m.size()[0], 1, m.size()[1], 1])
    m = torch.tile(m, (1, blocking, 1, blocking))
    nonzero_indices = torch.nonzero(m.flatten()).flatten()
    data = torch.gather(x.flatten(), dim=0, index=nonzero_indices)
    return Matrix(x.size(), data, indices, offsets)

