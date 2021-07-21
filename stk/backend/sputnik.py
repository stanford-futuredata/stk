import sputnik_backend as backend
import torch


def _sparse_transpose(x):
    shape = x[0].clone()
    shape[0] = x[0][1]
    shape[1] = x[0][0]
    return (shape,) + x[1:]


def _transpose_helper(x, transpose):
    if isinstance(x, torch.Tensor):
        return x.t() if transpose else x
    if transpose:
        x = _sparse_transpose(x)
    return x + (transpose,)


def _wrap(x):
    if isinstance(x, torch.Tensor):
        return (x,)
    return x

def _is_transposed(x):
    return (not x.is_contiguous() and
            x.stride()[0] == 1 and
            x.stride()[1] == x.size()[0])

def _call_helper(op, out, a, b, trans_a, trans_b):
    args = (_wrap(_transpose_helper(a, trans_a)) +
            _wrap(_transpose_helper(b, trans_b)))
    if isinstance(out, tuple):
        args = args + out
    return op(*args)


def _preprocess_inputs(lhs, rhs, dy):
    if isinstance(lhs, torch.Tensor) and _is_transposed(lhs):
        lhs = lhs.t()
    if isinstance(rhs, torch.Tensor) and _is_transposed(rhs):
        rhs = rhs.t()
    if not dy.is_contiguous() and not _is_transposed(dy):
        dy = dy.contiguous()
    return lhs, rhs, dy


def _postprocess_outputs(x, transpose, grad):
    if isinstance(x, torch.Tensor) and transpose:
        return grad.t()
    return grad


def _lhs_gradient(op, lhs, rhs, dy, trans_lhs, trans_rhs):
    lhs, rhs, dy = _preprocess_inputs(lhs, rhs, dy)

    a, b = (rhs, dy) if trans_lhs else (dy, rhs)
    trans_a = trans_lhs and trans_rhs
    trans_b = trans_lhs or not trans_rhs
    out = _call_helper(op, lhs, a, b, trans_a, trans_b)
    return _postprocess_outputs(lhs, trans_lhs, out)


def _rhs_gradient(op, lhs, rhs, dy, trans_lhs, trans_rhs):
    lhs, rhs, dy = _preprocess_inputs(lhs, rhs, dy)

    a, b = (dy, lhs) if trans_rhs else (lhs, dy)
    trans_a = not trans_lhs or trans_rhs
    trans_b = trans_lhs and trans_rhs
    out = _call_helper(op, rhs, a, b, trans_a, trans_b)
    return _postprocess_outputs(rhs, trans_rhs, out)


class DSD(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                shape,
                data,
                offsets,
                indices,
                transpose_a,
                rhs):
        ctx.save_for_backward(shape, data, offsets, indices, rhs)
        ctx.transpose_a = transpose_a
        out = torch.empty((shape[0], rhs.size()[1]),
                          dtype=rhs.dtype,
                          device=rhs.device)
        backend.dsd(shape,
                    data,
                    offsets,
                    indices,
                    transpose_a,
                    rhs,
                    out)
        return out

    @staticmethod
    def backward(ctx, dy):
        shape, data, offsets, indices, rhs = ctx.saved_tensors
        lhs = (shape, data, offsets, indices)
        trans_a = ctx.transpose_a
        trans_b = _is_transposed(rhs)

        ddata = None
        if ctx.needs_input_grad[1]:
            ddata = _lhs_gradient(sdd,
                                  lhs,
                                  rhs,
                                  dy,
                                  trans_a,
                                  trans_b)
        drhs = None
        if ctx.needs_input_grad[5]:
            op = dds if trans_b else dsd
            drhs = _rhs_gradient(op,
                                 lhs,
                                 rhs,
                                 dy,
                                 trans_a,
                                 trans_b)
        return None, ddata, None, None, None, drhs


dsd = DSD.apply


class DDS(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                lhs,
                shape,
                data,
                offsets,
                indices,
                transpose_b):
        out = torch.empty((lhs.size()[0], shape[1]),
                          dtype=lhs.dtype,
                          device=lhs.device)
        backend.dds(lhs,
                    shape,
                    data,
                    offsets,
                    indices,
                    transpose_b,
                    out)
        return out


dds = DDS.apply


class SDD(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                lhs,
                rhs,
                shape,
                data,
                offsets,
                indices):
        out = torch.empty_like(data)
        backend.sdd(lhs,
                    rhs,
                    shape,
                    out,
                    offsets,
                    indices)
        return out


sdd = SDD.apply


class SSD(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                lhs_shape,
                lhs_data,
                lhs_offsets,
                lhs_indices,
                transpose_a,
                rhs,
                shape,
                data,
                offsets,
                indices):
        out = torch.empty_like(data)
        backend.ssd(lhs_shape,
                    lhs_data,
                    lhs_offsets,
                    lhs_indices,
                    transpose_a,
                    rhs,
                    shape,
                    out,
                    offsets,
                    indices)
        return out


ssd = SSD.apply

class DSS(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                lhs_shape,
                lhs_data,
                lhs_offsets,
                lhs_indices,
                transpose_a,
                rhs_shape,
                rhs_data,
                rhs_offsets,
                rhs_indices,
                transpose_b):
        out = torch.empty((lhs_shape[0], rhs_shape[1]),
                          dtype=lhs_data.dtype,
                          device=lhs_data.device)
        backend.dss(lhs_shape,
                    lhs_data,
                    lhs_offsets,
                    lhs_indices,
                    transpose_a,
                    rhs_shape,
                    rhs_data,
                    rhs_offsets,
                    rhs_indices,
                    transpose_b,
                    out)
        return out


dss = DSS.apply
