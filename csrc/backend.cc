#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
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
#define CHECK_3D(x) TORCH_CHECK(x.ndimension() == 3)
#define CHECK_SHAPE(x) TORCH_CHECK(x.numel() == 2)

void validate_shape(torch::Tensor shape) {
  CHECK_CPU(shape);
  CHECK_SHAPE(shape);
  CHECK_INT(shape);
}

void validate_transpose(torch::Tensor transpose) {
  CHECK_CPU(transpose);
  CHECK_SCALAR(transpose);
  CHECK_INT(transpose);
}

void validate_sparse(torch::Tensor shape,
		     torch::Tensor data,
		     torch::Tensor offsets,
		     torch::Tensor indices,
		     torch::Tensor transpose) {
  validate_shape(shape);
  validate_transpose(transpose);
  CHECK_CUDA(data);
  CHECK_3D(data);
  CHECK_HALF(data);
  CHECK_CUDA(offsets);
  CHECK_VECTOR(offsets);
  CHECK_INT(offsets);
  CHECK_CUDA(indices);
  CHECK_VECTOR(indices);
  CHECK_SHORT(indices);

  // Blocking must be square.
  TORCH_CHECK(data.size(1) == data.size(2));

  // TODO(tgale): Generalize this.
  TORCH_CHECK(data.size(1) == 128);
}

bool is_transposed(torch::Tensor x) {
  return x.strides()[0] == 1 && x.strides()[1] == x.sizes()[0];
}

void validate_dense(torch::Tensor x) {
  CHECK_CUDA(x);
  CHECK_MATRIX(x);
  CHECK_HALF(x);

  TORCH_CHECK(x.is_contiguous() || is_transposed(x));
}

//
/// Conversion helpers.
//

sputnik::block::Matrix as_matrix(torch::Tensor x) {
  validate_dense(x);
  int rows = is_transposed(x) ? x.sizes()[1] : x.sizes()[0];
  int cols = is_transposed(x) ? x.sizes()[0] : x.sizes()[1];
  return sputnik::block::Matrix(rows, cols, x.data_ptr());
}

int access_metadata(torch::Tensor m, int idx = 0) {
  auto accessor = m.accessor<int, 1>();
  return accessor[idx];
}

// TODO(tgale): Overload this to handle transposes.
sputnik::block::BlockMatrix as_block_matrix(torch::Tensor shape,
					    torch::Tensor data,
					    torch::Tensor offsets,
					    torch::Tensor indices,
					    torch::Tensor transpose) {
  validate_sparse(shape,
		  data,
		  offsets,
		  indices,
		  transpose);

  // TODO(tgale): Generalize this.
  TORCH_CHECK(!access_metadata(transpose));
  return sputnik::block::BlockMatrix(access_metadata(shape, 0),
				     access_metadata(shape, 1),
				     sputnik::block::AsBlockSize(data.size(1)),
				     data.numel(),
				     data.data_ptr(),
				     offsets.data_ptr(),
				     indices.data_ptr());
}

//
/// Custom operations.
//

void dsd(torch::Tensor shape,
	 torch::Tensor data,
	 torch::Tensor offsets,
	 torch::Tensor indices,
	 torch::Tensor transpose_a,
	 torch::Tensor rhs_t,
	 torch::Tensor out_t) {
  // Convert the arguments to sputnik types.
  auto lhs = as_block_matrix(shape,
			     data,
			     offsets,
			     indices,
			     transpose_a);
  bool transpose_lhs = access_metadata(transpose_a);
  auto rhs = as_matrix(rhs_t);
  bool transpose_rhs = is_transposed(rhs_t);
  auto out = as_matrix(out_t);

  // Validate the problem configuration.
  TORCH_CHECK(sputnik::block::ValidMatmul(lhs,
					  transpose_lhs,
					  rhs,
					  transpose_rhs,
					  out));
  sputnik::block::Matmul(lhs,
			 transpose_lhs,
			 rhs,
			 transpose_rhs,
			 out,
			 c10::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dsd", &dsd, "dense = op(sparse) x op(dense)");
}
