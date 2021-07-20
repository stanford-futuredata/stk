from stk.backend import sputnik
from stk.matrix import Matrix
import torch


def _make_shape_tensor(x):
    return torch.tensor(
        x.size(),
        dtype=torch.int32,
        device=torch.device("cpu"))


def _make_transpose_tensor(x):
    return torch.tensor(
        [not x.is_contiguous],
        dtype=torch.int32,
        device=torch.device("cpu"))


def dsd(a, b):
    assert isinstance(a, Matrix)
    assert isinstance(b, torch.Tensor)
    return sputnik.dsd(
        _make_shape_tensor(a),
        a.data, a.offsets, a.indices,
        _make_transpose_tensor(a),
        b)


def dds(a, b):
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, Matrix)
    return sputnik.dds(
        a,
        _make_shape_tensor(b),
        b.data, b.offsets, b.indices,
        _make_transpose_tensor(b))


def sdd(a, b, topo):
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert isinstance(topo, Matrix)
    assert topo.is_contiguous
    out = sputnik.sdd(
        a, b,
        _make_shape_tensor(topo),
        topo.data, topo.offsets, topo.indices)
    return Matrix(topo.size(), out, topo.indices, topo.offsets)


def ssd(a, b, topo):
    assert isinstance(a, Matrix)
    assert isinstance(b, torch.Tensor)
    assert isinstance(topo, Matrix)
    assert topo.is_contiguous
    out = sputnik.ssd(
        _make_shape_tensor(a),
        a.data, a.offsets, a.indices,
        _make_transpose_tensor(a),
        b,
        _make_shape_tensor(topo),
        topo.data, topo.offsets, topo.indices)
    return Matrix(topo.size(), out, topo.indices, topo.offsets)


def dss(a, b):
    assert isinstance(a, Matrix)
    assert isinstance(b, Matrix)
    return sputnik.dss(
        _make_shape_tensor(a),
        a.data, a.offsets, a.indices,
        _make_transpose_tensor(a),
        _make_shape_tensor(b),
        b.data, b.offsets, b.indices,
        _make_transpose_tensor(b))
