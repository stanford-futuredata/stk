#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <sputnik/sputnik.h>

//
/// Validation helpers.
//

#define CALL_CUDA(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda())
#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == torch::ScalarType::Half)
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type() == torch::ScalarType::Int)
#define CHECK_SHORT(x) TORCH_CHECK(x.scalar_type() == torch::ScalarType::Short)
#define CHECK_VECTOR(x) TORCH_CHECK(x.ndimension() == 1)
#define CHECK_MATRIX(x) TORCH_CHECK(x.ndimension() == 2)
#define CHECK_3D(x) TORCH_CHECK(x.ndimension() == 3)

using Shape = std::pair<int, int>;

void validate_sparse(torch::Tensor data,
		     torch::Tensor offsets,
		     torch::Tensor column_indices) {
  CHECK_CUDA(data);
  CHECK_3D(data);
  CHECK_HALF(data);
  CHECK_CUDA(offsets);
  CHECK_VECTOR(offsets);
  CHECK_INT(offsets);
  CHECK_CUDA(column_indices);
  CHECK_VECTOR(column_indices);
  CHECK_SHORT(column_indices);

  // Blocking must be square.
  TORCH_CHECK(data.size(1) == data.size(2));

  // TODO(tgale): Generalize this.
  TORCH_CHECK(data.size(1) == 128);
}

void validate_sparse(torch::Tensor data,
		     torch::Tensor offsets,
		     torch::Tensor row_indices,
		     torch::Tensor column_indices) {
  CHECK_CUDA(row_indices);
  CHECK_VECTOR(row_indices);
  CHECK_SHORT(row_indices);

  // Both match number of blocks.
  TORCH_CHECK(row_indices.size(0) == column_indices.size(0));
  validate_sparse(data, offsets, column_indices);
}

void validate_sparse(torch::Tensor data,
		     torch::Tensor offsets,
		     torch::Tensor row_indices,
		     torch::Tensor column_indices,
		     torch::Tensor offsets_t,
		     torch::Tensor column_indices_t,
		     torch::Tensor block_offsets_t) {
  CHECK_CUDA(offsets_t);
  CHECK_VECTOR(offsets_t);
  CHECK_INT(offsets_t);
  CHECK_CUDA(column_indices_t);
  CHECK_VECTOR(column_indices_t);
  CHECK_SHORT(column_indices_t);
  CHECK_CUDA(block_offsets_t);
  CHECK_VECTOR(block_offsets_t);
  CHECK_INT(block_offsets_t);

  TORCH_CHECK(column_indices_t.size(0) == row_indices.size(0));
  validate_sparse(data, offsets, row_indices, column_indices);
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

// TODO(tgale): Can we delete this?
sputnik::block::BlockMatrix as_block_matrix(Shape shape,
					    torch::Tensor data,
					    torch::Tensor offsets,
					    torch::Tensor column_indices) {
  validate_sparse(data, offsets, column_indices);
  return sputnik::block::BlockMatrix(shape.first, shape.second,
				     sputnik::block::AsBlockSize(data.size(1)),
				     data.numel(),
				     data.data_ptr(),
				     offsets.data_ptr(),
				     column_indices.data_ptr());
}

sputnik::block::BlockMatrix as_block_matrix(Shape shape,
					    torch::Tensor data,
					    torch::Tensor offsets,
					    torch::Tensor row_indices,
					    torch::Tensor column_indices) {
  validate_sparse(data, offsets, row_indices, column_indices);
  return sputnik::block::BlockMatrix(shape.first, shape.second,
				     sputnik::block::AsBlockSize(data.size(1)),
				     data.numel(),
				     data.data_ptr(),
				     offsets.data_ptr(),
				     column_indices.data_ptr(),
				     row_indices.data_ptr());
}

sputnik::block::BlockMatrix as_block_matrix(Shape shape,
					    torch::Tensor data,
					    torch::Tensor offsets,
					    torch::Tensor row_indices,
					    torch::Tensor column_indices,
					    torch::Tensor offsets_t,
					    torch::Tensor column_indices_t,
					    torch::Tensor block_offsets_t) {
  validate_sparse(data, offsets, row_indices, column_indices, offsets_t, column_indices_t, block_offsets_t);
  auto out = sputnik::block::BlockMatrix(shape.first, shape.second,
					 sputnik::block::AsBlockSize(data.size(1)),
					 data.numel(),
					 data.data_ptr(),
					 offsets.data_ptr(),
					 column_indices.data_ptr(),
					 row_indices.data_ptr());
  out.offsets_t = offsets_t.data_ptr();
  out.indices_t = column_indices_t.data_ptr();
  out.block_offsets = block_offsets_t.data_ptr();
  return out;
}

//
/// Row indices helper.
//

void row_indices(Shape shape,
		 torch::Tensor data,
		 torch::Tensor offsets,
		 torch::Tensor column_indices,
		 torch::Tensor row_indices) {
  if (row_indices.numel() == 0) return;
  auto x = as_block_matrix(shape, data, offsets, row_indices, column_indices);
  CALL_CUDA(sputnik::block::RowIndices(x,
				       (short*)x.row_indices,
				       c10::cuda::getCurrentCUDAStream()));
}

//
/// Transpose helper.
//

// TODO(tgale): Replace sort/histc with our faster operations.
// Will need to move those from blockparty into this repo.
//
// TODO(tgale): Consider also returning row indices for the
// transposed matrix if we ever expose the transposed matrix.
std::vector<torch::Tensor> transpose(Shape shape,
				     torch::Tensor data,
				     torch::Tensor offsets,
				     torch::Tensor row_indices,
				     torch::Tensor column_indices) {
  const int kBlockSize = data.size(1);
  const int kCols = shape.second;
  const int kBlockCols = kCols / kBlockSize;

  // Sort row indices by column indices to get the transposed
  // matrix's column indices.
  torch::Tensor gather_indices = column_indices.argsort();
  torch::Tensor column_indices_t = row_indices.gather(0, gather_indices);
  torch::Tensor block_offsets_t = gather_indices.to(torch::kInt32);

  auto options = torch::TensorOptions()
    .dtype(torch::kInt32)
    .device(data.device());
  torch::Tensor zero = torch::zeros(1, options);

  // Calculate the transposed matrix's offsets.
  torch::Tensor nnz_per_column = column_indices.histc(kBlockCols, 0, kBlockCols);
  torch::Tensor offsets_t = at::cat({zero, nnz_per_column.cumsum(0, torch::kInt32)});
  return {column_indices_t, offsets_t, block_offsets_t};
}

void standardize_shape(Shape &shape, bool transpose) {
  int rows = transpose ? shape.second : shape.first;
  int cols = transpose ? shape.first : shape.second;
  shape.first = rows;
  shape.second = cols;
}

//
/// Bitmask helper.
//

size_t size_in_bytes(int rows, int columns) {
  const int kAlignment = 64;
  int column_entries = (columns + kAlignment - 1) / kAlignment;
  return column_entries * rows * sizeof(uint64_t);
}

torch::Tensor allocate_bitmask(sputnik::block::BlockMatrix m, bool trans, torch::Device device) {
  int block_size = AsInt(m.block_size);
  int block_rows = (trans ? m.cols : m.rows) / block_size;
  int block_cols = (trans ? m.rows : m.cols) / block_size;
  size_t bytes = size_in_bytes(block_rows, block_cols);

  auto options = torch::TensorOptions()
    .dtype(torch::kInt8)
    .device(device);
  return torch::empty(bytes, options);
}

//
/// Custom operations.
//

void dsd(Shape shape,
	 torch::Tensor data,
	 torch::Tensor offsets,
	 torch::Tensor row_indices,
	 torch::Tensor column_indices,
	 torch::Tensor offsets_t,
	 torch::Tensor column_indices_t,
	 torch::Tensor block_offsets_t,
	 bool transpose_lhs,
	 torch::Tensor rhs_t,
	 torch::Tensor out_t) {
  if (out_t.numel() == 0) return;

  // If either of the inputs is zero sized the output is zero. This
  // can happen in gradient calculation for a product between a zero
  // sized tensor and a non-zero sized tensor.
  if ((data.numel() == 0) || (rhs_t.numel() == 0)) {
    out_t.zero_();
    return;
  }

  // Convert the arguments to sputnik types.
  standardize_shape(shape, transpose_lhs);
  auto lhs = as_block_matrix(shape,
			     data,
			     offsets,
			     row_indices,
			     column_indices,
			     offsets_t,
			     column_indices_t,
			     block_offsets_t);
  auto rhs = as_matrix(rhs_t);
  bool transpose_rhs = is_transposed(rhs_t);
  auto out = as_matrix(out_t);

  // Validate the problem configuration.
  TORCH_CHECK(sputnik::block::ValidMatmul(lhs,
					  transpose_lhs,
					  rhs,
					  transpose_rhs,
					  out));

  CALL_CUDA(sputnik::block::MatmulEx(lhs,
				     transpose_lhs,
				     rhs,
				     transpose_rhs,
				     out,
				     c10::cuda::getCurrentCUDAStream()));
}

void dds(torch::Tensor lhs_t,
	 Shape shape,
	 torch::Tensor data,
	 torch::Tensor offsets,
	 torch::Tensor row_indices,
	 torch::Tensor column_indices,
	 torch::Tensor offsets_t,
	 torch::Tensor column_indices_t,
	 torch::Tensor block_offsets_t,
	 bool transpose_rhs,
	 torch::Tensor out_t) {
  if (out_t.numel() == 0) return;

  // If either of the inputs is zero sized the output is zero. This
  // can happen in gradient calculation for a product between a zero
  // sized tensor and a non-zero sized tensor.
  if ((data.numel() == 0) || (lhs_t.numel() == 0)) {
    out_t.zero_();
    return;
  }

  // Convert the arguments to sputnik types.
  auto lhs = as_matrix(lhs_t);
  bool transpose_lhs = is_transposed(lhs_t);
  standardize_shape(shape, transpose_rhs);
  auto rhs = as_block_matrix(shape,
			     data,
			     offsets,
			     row_indices,
			     column_indices,
			     offsets_t,
			     column_indices_t,
			     block_offsets_t);
  auto out = as_matrix(out_t);

  // Validate the problem configuration.
  TORCH_CHECK(sputnik::block::ValidMatmul(lhs,
					  transpose_lhs,
					  rhs,
					  transpose_rhs,
					  out));

  CALL_CUDA(sputnik::block::MatmulEx(lhs,
				     transpose_lhs,
				     rhs,
				     transpose_rhs,
				     out,
				     c10::cuda::getCurrentCUDAStream()));
}

void sdd(torch::Tensor lhs_t,
	 torch::Tensor rhs_t,
	 Shape shape,
	 torch::Tensor data,
	 torch::Tensor offsets,
	 torch::Tensor row_indices,
	 torch::Tensor column_indices) {
  if (data.numel() == 0) return;

  // If either of the inputs is zero sized the output is zero. This
  // can happen in gradient calculation for a product between a zero
  // sized tensor and a non-zero sized tensor.
  if ((lhs_t.numel() == 0) || (rhs_t.numel() == 0)) {
    data.zero_();
    return;
  }

  // Convert the arguments to sputnik types.
  auto lhs = as_matrix(lhs_t);
  bool transpose_lhs = is_transposed(lhs_t);
  auto rhs = as_matrix(rhs_t);
  bool transpose_rhs = is_transposed(rhs_t);
  auto out = as_block_matrix(shape,
			     data,
			     offsets,
			     row_indices,
			     column_indices);

  // Validate the problem configuration.
  TORCH_CHECK(sputnik::block::ValidMatmul(lhs,
					  transpose_lhs,
					  rhs,
					  transpose_rhs,
					  out));

  CALL_CUDA(sputnik::block::Matmul(lhs,
				   transpose_lhs,
				   rhs,
				   transpose_rhs,
				   out,
				   c10::cuda::getCurrentCUDAStream()));
}

void ssd(Shape lhs_shape,
	 torch::Tensor lhs_data,
	 torch::Tensor lhs_offsets,
	 torch::Tensor lhs_row_indices,
	 torch::Tensor lhs_column_indices,
	 bool transpose_lhs,
	 torch::Tensor rhs_t,
	 Shape out_shape,
	 torch::Tensor out_data,
	 torch::Tensor out_offsets,
	 torch::Tensor out_row_indices,
	 torch::Tensor out_column_indices) {
  if (out_data.numel() == 0) return;

  // If either of the inputs is zero sized the output is zero. This
  // can happen in gradient calculation for a product between a zero
  // sized tensor and a non-zero sized tensor.
  if ((lhs_data.numel() == 0) || (rhs_t.numel() == 0)) {
    out_data.zero_();
    return;
  }

  // Convert the arguments to sputnik types.
  standardize_shape(lhs_shape, transpose_lhs);
  auto lhs = as_block_matrix(lhs_shape,
			     lhs_data,
			     lhs_offsets,
			     lhs_row_indices,
			     lhs_column_indices);

  auto rhs = as_matrix(rhs_t);
  bool transpose_rhs = is_transposed(rhs_t);

  auto out = as_block_matrix(out_shape,
			     out_data,
			     out_offsets,
			     out_row_indices,
			     out_column_indices);

  // Validate the problem configuration.
  TORCH_CHECK(sputnik::block::ValidMatmul(lhs,
					  transpose_lhs,
					  rhs,
					  transpose_rhs,
					  out));

  std::vector<torch::Tensor> meta;
  if (transpose_lhs) {
    // Populate the transpose meta-data.
    meta = transpose(lhs_shape,
		     lhs_data,
		     lhs_offsets,
		     lhs_row_indices,
		     lhs_column_indices);

    // Set the data pointers.
    lhs.indices_t = meta[0].data_ptr();
    lhs.offsets_t = meta[1].data_ptr();
    lhs.block_offsets = meta[2].data_ptr();
  }

  CALL_CUDA(sputnik::block::MatmulEx(lhs,
				     transpose_lhs,
				     rhs,
				     transpose_rhs,
				     out,
				     c10::cuda::getCurrentCUDAStream()));
}

void sds(torch::Tensor lhs_t,
	 Shape rhs_shape,
	 torch::Tensor rhs_data,
	 torch::Tensor rhs_offsets,
	 torch::Tensor rhs_row_indices,
	 torch::Tensor rhs_column_indices,
	 bool transpose_rhs,
	 Shape out_shape,
	 torch::Tensor out_data,
	 torch::Tensor out_offsets,
	 torch::Tensor out_row_indices,
	 torch::Tensor out_column_indices) {
  if (out_data.numel() == 0) return;

  // If either of the inputs is zero sized the output is zero. This
  // can happen in gradient calculation for a product between a zero
  // sized tensor and a non-zero sized tensor.
  if ((lhs_t.numel() == 0) || (rhs_data.numel() == 0)) {
    out_data.zero_();
    return;
  }

  // Convert the arguments to sputnik types.
  auto lhs = as_matrix(lhs_t);
  bool transpose_lhs = is_transposed(lhs_t);

  standardize_shape(rhs_shape, transpose_rhs);
  auto rhs = as_block_matrix(rhs_shape,
			     rhs_data,
			     rhs_offsets,
			     rhs_row_indices,
			     rhs_column_indices);
  auto out = as_block_matrix(out_shape,
			     out_data,
			     out_offsets,
			     out_row_indices,
			     out_column_indices);

  // Validate the problem configuration.
  TORCH_CHECK(sputnik::block::ValidMatmul(lhs,
					  transpose_lhs,
					  rhs,
					  transpose_rhs,
					  out));

  std::vector<torch::Tensor> meta;
  if (!transpose_rhs) {
    // Populate the transpose meta-data.
    meta = transpose(rhs_shape,
		     rhs_data,
		     rhs_offsets,
		     rhs_row_indices,
		     rhs_column_indices);

    // Set the data pointers.
    rhs.indices_t = meta[0].data_ptr();
    rhs.offsets_t = meta[1].data_ptr();
    rhs.block_offsets = meta[2].data_ptr();
  }

  CALL_CUDA(sputnik::block::MatmulEx(lhs,
				     transpose_lhs,
				     rhs,
				     transpose_rhs,
				     out,
				     c10::cuda::getCurrentCUDAStream()));
}

void dss(Shape lhs_shape,
	 torch::Tensor lhs_data,
	 torch::Tensor lhs_offsets,
	 torch::Tensor lhs_row_indices,
	 torch::Tensor lhs_column_indices,
	 bool transpose_lhs,
	 Shape rhs_shape,
	 torch::Tensor rhs_data,
	 torch::Tensor rhs_offsets,
	 torch::Tensor rhs_row_indices,
	 torch::Tensor rhs_column_indices,
	 bool transpose_rhs,
	 torch::Tensor out_t) {
  if (out_t.numel() == 0) return;

  // If either of the inputs is zero sized the output is zero. This
  // can happen in gradient calculation for a product between a zero
  // sized tensor and a non-zero sized tensor.
  if ((lhs_data.numel() == 0) || (rhs_data.numel() == 0)) {
    out_t.zero_();
    return;
  }

  // Convert the arguments to sputnik types.
  standardize_shape(lhs_shape, transpose_lhs);
  auto lhs = as_block_matrix(lhs_shape,
			     lhs_data,
			     lhs_offsets,
			     lhs_row_indices,
			     lhs_column_indices);

  standardize_shape(rhs_shape, transpose_rhs);
  auto rhs = as_block_matrix(rhs_shape,
			     rhs_data,
			     rhs_offsets,
			     rhs_row_indices,
			     rhs_column_indices);
  auto out = as_matrix(out_t);

  // Validate the problem configuration.
  TORCH_CHECK(sputnik::block::ValidMatmul(lhs,
					  transpose_lhs,
					  rhs,
					  transpose_rhs,
					  out));

  // Allocate workspace for the bitmasks.
  auto lhs_bitmask = allocate_bitmask(lhs, transpose_lhs, lhs_data.device());
  lhs.bitmask = lhs_bitmask.data_ptr();
  auto rhs_bitmask = allocate_bitmask(rhs, transpose_rhs, rhs_data.device());
  rhs.bitmask = rhs_bitmask.data_ptr();

  // Handle exposed transposes.
  std::vector<torch::Tensor> lhs_meta, rhs_meta;
  if (transpose_lhs) {
    // Populate the transpose meta-data.
    lhs_meta = transpose(lhs_shape,
			 lhs_data,
			 lhs_offsets,
			 lhs_row_indices,
			 lhs_column_indices);

    // Set the data pointers.
    lhs.indices_t = lhs_meta[0].data_ptr();
    lhs.offsets_t = lhs_meta[1].data_ptr();
    lhs.block_offsets = lhs_meta[2].data_ptr();
  }
  if (!transpose_rhs) {
    // Populate the transpose meta-data.
    rhs_meta = transpose(rhs_shape,
			 rhs_data,
			 rhs_offsets,
			 rhs_row_indices,
			 rhs_column_indices);

    // Set the data pointers.
    rhs.indices_t = rhs_meta[0].data_ptr();
    rhs.offsets_t = rhs_meta[1].data_ptr();
    rhs.block_offsets = rhs_meta[2].data_ptr();
  }

  CALL_CUDA(sputnik::block::MatmulEx(lhs,
				     transpose_lhs,
				     rhs,
				     transpose_rhs,
				     out,
				     c10::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dsd", &dsd, "dense = op(sparse) x op(dense)");
  m.def("dds", &dds, "dense = op(dense) x op(sparse)");
  m.def("sdd", &sdd, "sparse = op(dense) x op(dense)");
  m.def("ssd", &ssd, "sparse = op(sparse) x op(dense)");
  m.def("sds", &sds, "sparse = op(dense) x op(sparse)");
  m.def("dss", &dss, "dense = op(sparse) x op(sparse)");
  m.def("row_indices", &row_indices, "create row indices for sparse matrix");
}
