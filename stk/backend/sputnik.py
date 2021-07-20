import sputnik_backend as backend
import torch


class DSD(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                shape,
                data,
                offsets,
                indices,
                transpose_a,
                rhs):
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
