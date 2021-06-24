import sputnik_backend as backend
import torch


# TODO(tgale): Make this support transposes.
class DSD(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shape, data, offsets, indices, transpose_a, rhs):
        out = torch.empty((shape[0], rhs.size()[1]), dtype=rhs.dtype, device=rhs.device)
        backend.dsd(shape, data, offsets, indices, transpose_a, rhs, out)
        return out

dsd = DSD.apply

