import torch

# 2. Add heavyweight (data) validation helper.
# 3. Add construction helpers
# 4. Test with custom kernels.
# 5. Make indentation consistent
# 6. Replace asserts with descriptive errors.

##
### Validation helpers.
##

def _validate_shape(shape):
    shape = torch.Size(shape)
    if len(shape) != 2:
        raise ValueError(
            f"shape must have 2 values. Got {len(shape)} values.")
    return shape


def _validate_matrix(shape, data, indices, offsets):
    # Data should be [nnz, block_size, block_size]
    if data.dim() == 1:
        data = torch.reshape(data, [data.numel(), 1, 1])

    if data.dim() != 3:
        raise ValueError(
            "Expected 3D shape for data (nnz, block, block). "
            f"Got shape {data.dim()}D shape.")
    if data.shape[1] != data.shape[2]:
        raise ValueError(
            "Expected square blocking in data. "
            f"Got block shape {[data.shape[1], data.shape[2]]}")

    block_size = data.shape[1]
    if shape[0] % block_size != 0 or shape[1] % block_size != 0:
        raise ValueError(
            "Matrix shape must be dividible by blocking. "
            f"Got shape {[shape[0], shape[1]]} with "
            f"{[block_size, block_size]} blocking.")

    if shape[0] * shape[1] < data.numel():
        raise ValueError(
            "Invalid matrix. Number of nonzeros exceeds matrix capacity "
            f"({data.numel()} v. {shape[0] * shape[1]})")

    if indices.dim() != 1:
        raise ValueError(
            f"Expected 1D indices. Got {indices.dim()}D indices.")

    if offsets.dim() != 1:
        raise ValueError(
            f"Expected 1D offsets. Got {offsets.dim()}D offsets.")

    if indices.numel() != data.shape[0]:
        raise ValueError(
            "Expected 1 index per nonzero block. "
            f"Got {indices.numel()} indices for {data.shape[0]} blocks")

    block_rows = shape[0] / block_size
    if offsets.numel() != block_rows + 1:
        raise ValueError(
            "Expected one offset per block row plus one. "
            "Got {offsets.numel()} offsets with {block_rows} block rows.")

    is_cuda = data.is_cuda and indices.is_cuda and offsets.is_cuda
    is_cpu = not data.is_cuda and not indices.is_cuda and not offsets.is_cuda
    if not (is_cuda or is_cpu):
        raise ValueError(
            "Expected data & meta-data on common device. "
            f"Got data on {data.device}, indices on {indices.device} "
            f"and offsets on {offsets.device}.")

    if data.dtype != torch.float16:
        raise ValueError(
            f"Expected float16 data. Got {data.dtype} data.")
    if indices.dtype != torch.int16:
        raise ValueError(
            f"Expected int16 indices. Got {indices.dtype} indices.")
    if offsets.dtype != torch.int32:
        raise ValueError(
            f"Expected int32 offsets. Got {offsets.dtype} offsets.")
    return data


class Matrix(object):
    """A matrix stored in sparse format.

    Underlying format is block compressed sparse row (BCSR).

    TODO(tgale): Make this mirror torch.Tensor API as much as possible.
    """

    def __init__(self,
                 size,
                 data,
                 indices,
                 offsets):
        self._size = _validate_shape(size)
        self._indices = indices
        self._offsets = offsets

        # Lightweight validation.
        self._data = _validate_matrix(self._size, data, self._indices, self._offsets)
        self._transposed = False


    def validate(self):
        _validate_shape(self._size)
        _validate_matrix(self._size, self._data, self._indices, self._offsets)

        # TODO(tgale): Add heavyweight data validation.

    def to(self, device):
        # TODO(tgale): Handle type conversions here. We
        # need to set the appropriate meta-data type for
        # the given floating-point type.
        self._data = self._data.to(device)
        self._indices = self._indices.to(device)
        self._offsets = self._offsets.to(device)
        return self

    def t(self):
        self._transposed = not self._transposed
        self._size = torch.Size((self._size[1], self._size[0]))
        return self

    def contiguous(self):
        raise ValueError("Not yet implemented.")

    @property
    def is_contiguous(self):
        return not self._transposed

    @property
    def is_cuda(self):
        return self._data.is_cuda

    @property
    def device(self):
        return self._data.device

    def size(self):
        return self._size

    def dim(self):
        return len(self._size)

    @property
    def data(self):
        return self._data

    @property
    def indices(self):
        return self._indices

    @property
    def offsets(self):
        return self._offsets

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def nnz(self):
        return self.data.numel()

    @property
    def blocking(self):
        return self.data.shape[1]

    @property
    def requires_grad(self):
        return self.data.requires_grad

    @property
    def grad(self):
        return Matrix(self.shape,
                      self.data.grad,
                      self.indices,
                      self.offsets)
