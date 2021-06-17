#include <torch/extension.h>
#include <sputnik/sputnik.h>

//
/// Validation helpers.
//

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda())
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda())
#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == torch::ScalarType::Half)
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type() == torch::ScalarType::Int)
#define CHECK_SHORT(x) TORCH_CHECK(x.scalar_type() == torch::ScalarType::Short)
#define CHECK_SCALAR(x) TORCH_CHECK(x.numel() == 1)
#define CHECK_VECTOR(x) TORCH_CHECK(x.ndimension() == 1)
#define CHECK_MATRIX(x) TORCH_CHECK(x.ndimension() == 2)
#define CHECK_SHAPE(x) TORCH_CHECK(x.numel() == 2)

void validate_shape(torch::Tensor shape) {
  CHECK_CPU(shape);
  CHECK_SHAPE(shape);
  CHECK_INT(shape);
}

void validate_block_size(torch::Tensor block_size) {
  CHECK_CPU(block_size);
  CHECK_SCALAR(block_size);
  CHECK_INT(block_size);
}

void validate_transpose(torch::Tensor transpose) {
  CHECK_CPU(transpose);
  CHECK_SCALAR(transpose);
  CHECK_INT(transpose);
}

void validate_sparse(torch::Tensor shape,
		     torch::Tensor block_size,
		     torch::Tensor data,
		     torch::Tensor offsets,
		     torch::Tensor indices,
		     torch::Tensor transpose) {
  validate_shape(shape);
  validate_block_size(block_size);
  validate_transpose(transpose);
  CHECK_CUDA(data);
  CHECK_VECTOR(data);
  CHECK_HALF(data);
  CHECK_CUDA(offsets);
  CHECK_VECTOR(offsets);
  CHECK_INT(offsets);
  CHECK_CUDA(indices);
  CHECK_VECTOR(indices);
  CHECK_SHORT(indices);  
}

void validate_dense(torch::Tensor x) {
  CHECK_CUDA(x);
  CHECK_MATRIX(x);
  CHECK_HALF(x);

  // TODO(tgale): Validate strides.
}

//
/// Conversion helpers.
//

sputnik::block::Matrix as_matrix(torch::Tensor x) {
  validate_dense(x);

  // TODO(tgale): Generalize this and add stride checks
  // to the validation helper.
  TORCH_CHECK(x.is_contiguous());
  return sputnik::block::Matrix(x.sizes()[0],
				x.sizes()[1],
				x.data_ptr());
}

bool is_transposed(torch::Tensor x) {
  // If it passes validation and we're not contiguous,
  // the matrix is transposed.
  validate_dense(x);
  return !x.is_contiguous();
}

// TODO(tgale): Overload this to handle transposes.
sputnik::block::BlockMatrix as_block_matrix(torch::Tensor shape,
					    torch::Tensor block_size,
					    torch::Tensor data,
					    torch::Tensor offsets,
					    torch::Tensor indices,
					    torch::Tensor transpose) {
  validate_sparse(shape, block_size, data, offsets, indices, transpose);
  
  // TODO(tgale): Generalize this.
  TORCH_CHECK(!transpose[0]);

  return sputnik::block::BlockMatrix(shape[0], shape[1],
				     sputnik::block::AsBlockSize(block_size[1]),
				     data.data_ptr(),
				     offsets.data_ptr(),
				     indices.data_ptr());
}

void dsd(torch::Tensor shape,
	 torch::Tensor block_size,
	 torch::Tensor data,
	 torch::Tensor offsets,
	 torch::Tensor indices,
	 torch::Tensor transpose_a,
	 torch::Tensor rhs,
	 torch::Tensor out) {
  auto lhs = as_block_matrix(shape, block_size, data, offsets, indices, transpose_a);
  sputnik::block::Matmul(lhs, transpose_a[0],
			 as_matrix(rhs), is_transposed(rhs),
			 as_matrix(out));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dsd", &dsd, "dense = op(sparse) x op(dense)");
}
