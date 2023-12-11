# :abacus: Sparse Toolkit

A light-weight PyTorch library for block-sparse matrices and block-sparse matrix multiplication.

STK is built around a core sparse matrix class ([stk.Matrix](stk/matrix.py)), which uses a hybrid [blocked-CSR-COO](https://arxiv.org/abs/2211.15841) sparse matrix encoding to enable efficient matrix products with sparse inputs and outputs in transposed or non-transposed order. The library supports the following operations:

```
op: transpose or non-transpose

[Sparse Matrix Multiplication]
stk.ops.dsd: dense = op(sparse) x op(dense)
stk.ops.dds: dense = op(dense) x op(sparse)
stk.ops.sdd: sparse = op(dense) x op(dense)

[Sparse Matrix Conversion]
stk.ops.to_sparse: torch.Tensor => stk.Matrix
stk.ops.to_dense: stk.Matrix => torch.Tensor

[Sparse Matrix Generation]
stk.random.dense_mask: Create a random, block-sparse dense matrix.
stk.random.mask: Create a random, block-sparse sparse matrix.
```

STK is designed for applications where the sparse matrices change rapidly. This is complementary to libraries like [triton-blocksparse](https://github.com/ptillet/torch-blocksparse), which assume that sparse matrix topologies do not change between invocations.

# :rocket: Performance

![STK Performance](media/block_sparse_matmul_benchmarks.png)

Block-sparse matrix multiplication with STK is able to match the performance of cuBLAS on a range of problems. On these benchmarks from [MegaBlocks](https://github.com/stanford-futuredata/megablocks) dMoE models, STK realizes **98.6%** of cuBLAS throughput with `128x128` blocks on average.

```
Hardware: A100-SXM4-80GB
Software: CUDA 11.5, CUTLASS 2.5
```

# :building_construction: Installation

NOTE: This assumes that you have `torch` and `numpy` installed.

`pip install stanford-stk`

# :writing_hand: Citation

```
@article{megablocks-arxiv,
  author    = {Trevor Gale and Deepak Narayanan and Cliff Young and Matei Zaharia},
  title     = {MegaBlocks: Efficient Sparse Training with Mixture-of-Experts},
  journal   = {CoRR},
  volume    = {abs/2211.15841},
  year      = {2022},
}
```
