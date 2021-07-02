from stk.backend import sputnik
from stk.matrix import Matrix
import torch


def _make_shape_tensor(x):
    return torch.tensor(
        x.size(),
        dtype=torch.int32,
        device=torch.device("cpu"))


def _make_transpose_tensor(x):
    # TODO(tgale): Update to handle transposes.
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
