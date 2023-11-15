import unittest
import itertools
import numpy as np
import torch
from absl.testing import parameterized

import stk


def allclose(x, y, pct=0.25):
    mask = torch.isclose(x, y, rtol=5e-2)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True


# element-wise mul only uses (m, n)
_MUL_MATRIX_SIZES = (
    (128, 128, 0.0),
    (256, 256, 0.5),
    (2048, 1024, 0.8),
    (512, 128, 0.0),
    (128, 512, 0.0),
    (1024, 512, 0.0),
    (1024, 512, 0.5),
    (1024, 512, 0.75),
    (512, 1024, 0.0),
    (512, 1024, 0.5),
    (512, 1024, 0.75),
    (1024, 1024, 0.0),
    (1024, 1024, 0.5),
    (1024, 1024, 0.75),
)


_TRANSPOSE = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)

# both must be true 
_MUL_TRANSPOSE = (
    (False, False),
    (True, True),
)

_DTYPE = (
    torch.float16, torch.bfloat16
)

def _generate_testcases(_SIZES, _CUSTOM_TRANSPOSE):
    testcases = itertools.product(_SIZES, _CUSTOM_TRANSPOSE, _DTYPE)
    testcases = [(*size, *trans, 128, dtype) for 
        (size, trans, dtype) in testcases]
    return testcases

_MUL_TEST = _generate_testcases(_MUL_MATRIX_SIZES, _MUL_TRANSPOSE)

def _dense_and_sparse_like(x, std=0.1):

    dense_data = torch.randn_like(x.data) * std
    sparse = stk.Matrix(x.size(),
                        dense_data,
                        x.row_indices,
                        x.column_indices,
                        x.offsets)
    dense = stk.ops.to_dense(sparse)

    cuda_device = torch.device("cuda")
    return (dense.to(cuda_device).requires_grad_(True),
            sparse.to(cuda_device).requires_grad_(True))

def _dense_and_sparse(rows, cols, sparsity, blocking, dtype, std=0.1):
    mask = stk.random.dense_mask(rows, cols, sparsity, blocking)
    dense = (torch.randn(rows, cols) * std * mask).type(dtype)
    sparse = stk.ops.to_sparse(dense, blocking)
    cuda_device = torch.device("cuda")
    return (dense.to(cuda_device).requires_grad_(True),
            sparse.to(cuda_device).requires_grad_(True))

def _with_transpose(op, a, b, trans_a, trans_b):
    a = a.t() if trans_a else a
    b = b.t() if trans_b else b
    return op(a, b)

def _mask(x, mask):
    mask = stk.ops.to_dense(stk.ops.ones_like(mask))
    return x * mask


class LinearOpsTest(parameterized.TestCase):

    @parameterized.parameters(_MUL_TEST)
    def testElemwiseMulWithLike(self, m, n, sparsity, trans_a, trans_b, blocking, dtype):

        a_shape = (n, m) if trans_a else (m, n)
        # Common mask used to ensure same topology for a and b
        a_dense, a = _dense_and_sparse(*a_shape, sparsity, blocking, dtype)
        b_dense, b = _dense_and_sparse_like(a)

        out = _with_transpose(stk.ops.mul, a, b, trans_a, trans_b)
        expected_out = _with_transpose(torch.mul, a_dense, b_dense, trans_a, trans_b)

        # Compute the gradients w.r.t. the inputs.
        expected_out.sum().backward()
        stk.ops.sum(out).backward()

        # Validate the results.
        out = stk.ops.to_dense(out)

        self.assertEqual(out.dim(), 2)
        self.assertEqual(expected_out.size()[0], out.size()[0])
        self.assertEqual(expected_out.size()[1], out.size()[1])
        self.assertTrue(allclose(out, expected_out)) 

        # LHS gradient.
        grad = stk.ops.to_dense(a.grad)
        expected_grad = a_dense.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

        # RHS gradient.
        grad =  stk.ops.to_dense(b.grad)
        expected_grad = b_dense.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))



if __name__ == '__main__':
    unittest.main()
