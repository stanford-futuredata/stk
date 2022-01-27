from stk.backend import sputnik
from stk.matrix import Matrix
import torch

# TODO(tgale): Handle batched sparse matrices in
# a clean way.
def _make_shape_tensor(x):
    assert x.dim() == 2
    return torch.tensor(
        x.size(),
        dtype=torch.int32,
        device=torch.device("cpu"))

def dsd(a, b):
    assert isinstance(a, Matrix)
    assert isinstance(b, torch.Tensor)
    return sputnik.dsd(
        _make_shape_tensor(a),
        a.data, a.offsets,
        a.row_indices,
        a.column_indices,
        not a.is_contiguous(),
        b)


def dds(a, b):
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, Matrix)
    return sputnik.dds(
        a,
        _make_shape_tensor(b),
        b.data, b.offsets,
        b.row_indices,
        b.column_indices,
        not b.is_contiguous())


def sdd(a, b, topo):
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert isinstance(topo, Matrix)
    assert topo.is_contiguous()
    out = sputnik.sdd(
        a, b,
        _make_shape_tensor(topo),
        topo.data, topo.offsets,
        topo.row_indices,
        topo.column_indices)
    return Matrix(topo.size(),
                  out,
                  topo.row_indices,
                  topo.column_indices,
                  topo.offsets)


def ssd(a, b, topo):
    assert isinstance(a, Matrix)
    assert isinstance(b, torch.Tensor)
    assert isinstance(topo, Matrix)
    assert topo.is_contiguous()
    out = sputnik.ssd(
        _make_shape_tensor(a),
        a.data, a.offsets,
        a.row_indices,
        a.column_indices,
        not a.is_contiguous(),
        b,
        _make_shape_tensor(topo),
        topo.data, topo.offsets,
        topo.row_indices,
        topo.column_indices)
    return Matrix(topo.size(),
                  out,
                  topo.row_indices,
                  topo.column_indices,
                  topo.offsets)


def sds(a, b, topo):
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, Matrix)
    assert isinstance(topo, Matrix)
    assert topo.is_contiguous()
    out = sputnik.sds(
        a,
        _make_shape_tensor(b),
        b.data, b.offsets,
        b.row_indices,
        b.column_indices,
        not b.is_contiguous(),
        _make_shape_tensor(topo),
        topo.data, topo.offsets,
        topo.row_indices,
        topo.column_indices)
    return Matrix(topo.size(),
                  out,
                  topo.row_indices,
                  topo.column_indices,
                  topo.offsets)


def dss(a, b):
    assert isinstance(a, Matrix)
    assert isinstance(b, Matrix)
    return sputnik.dss(
        _make_shape_tensor(a),
        a.data, a.offsets,
        a.row_indices,
        a.column_indices,
        not a.is_contiguous(),
        _make_shape_tensor(b),
        b.data, b.offsets,
        b.row_indices,
        b.column_indices,
        not b.is_contiguous())
