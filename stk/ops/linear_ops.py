from stk.backend import sputnik
from stk.matrix import Matrix
import torch

def dsd(a, b):
    assert isinstance(a, Matrix)
    assert isinstance(b, torch.Tensor)
    return sputnik.dsd(
        a.size(),
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
        b.size(),
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
        topo.size(),
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
        a.size(),
        a.data, a.offsets,
        a.row_indices,
        a.column_indices,
        not a.is_contiguous(),
        b,
        topo.size(),
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
        b.size(),
        b.data, b.offsets,
        b.row_indices,
        b.column_indices,
        not b.is_contiguous(),
        topo.size(),
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
        a.size(),
        a.data, a.offsets,
        a.row_indices,
        a.column_indices,
        not a.is_contiguous(),
        b.size(),
        b.data, b.offsets,
        b.row_indices,
        b.column_indices,
        not b.is_contiguous())
