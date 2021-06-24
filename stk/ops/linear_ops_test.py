import unittest

from absl.testing import parameterized
import stk
import torch


# An assortment of problems designed to make sure
# the bindings are operating correctly. Extensive
# kernel tests done through Sputnik.
_SINGLY_SPARSE_TESTS = (
    (128, 128, 8, False, False, 128, 0.0),
    (256, 256, 8, False, False, 128, 0.5),
    (2048, 1024, 512, False, False, 128, 0.8),
    (128, 128, 8, False, True, 128, 0.0),
    (256, 256, 8, False, True, 128, 0.5),
    (2048, 1024, 512, False, True, 128, 0.8),
)


def _dense_and_sparse(rows, cols, sparsity, blocking):
    mask = stk.random.dense_mask(rows, cols, sparsity, blocking)
    dense = (torch.randn(rows, cols) * mask).type(torch.float16)
    sparse = stk.ops.to_sparse(dense, blocking)
    cuda_device = torch.device("cuda")
    return dense.to(cuda_device), sparse.to(cuda_device)


def _dense(rows, cols):
    cuda_device = torch.device("cuda")
    return torch.randn(rows, cols).type(torch.float16).to(cuda_device)


def _with_transpose(op, a, b, trans_a, trans_b):
    a = a.t() if trans_a else a
    b = b.t() if trans_b else b
    return op(a, b)


class LinearOpsTest(parameterized.TestCase):

    @parameterized.parameters(*_SINGLY_SPARSE_TESTS)
    def testLinearOps_Dsd(self, m, k, n, trans_a, trans_b, blocking, sparsity):
        # Construct the operands.
        a_shape = (k, m) if trans_a else (m, k)
        a_dense, a = _dense_and_sparse(*a_shape, sparsity, blocking)
        b_shape = (n, k) if trans_b else (k, n)
        b = _dense(*b_shape)

        # Execute the matmul.
        out = _with_transpose(stk.ops.dsd, a, b, trans_a, trans_b)
        expected_out = _with_transpose(torch.mm, a_dense, b, trans_a, trans_b)

        # Validate the results.
        self.assertEqual(out.dim(), 2)
        self.assertEqual(expected_out.size()[0], out.size()[0])
        self.assertEqual(expected_out.size()[1], out.size()[1])
        self.assertTrue(torch.allclose(out, expected_out))


if __name__ == '__main__':
    unittest.main()
