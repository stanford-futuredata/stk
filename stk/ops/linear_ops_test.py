import unittest

from absl.testing import parameterized
import numpy as np
import stk
import torch


def allclose(x, y, pct=0.25):
    mask = torch.isclose(x, y, rtol=5e-2)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True


# An assortment of problems designed to make sure
# the bindings are operating correctly. Extensive
# kernel tests done through Sputnik.
_LINEAR_OP_TESTS = (
    (128, 128, 128, False, False, 128, 0.0),
    (256, 256, 256, False, False, 128, 0.5),
    (2048, 1024, 512, False, False, 128, 0.8),
    (128, 128, 128, False, True, 128, 0.0),
    (256, 256, 256, False, True, 128, 0.5),
    (2048, 1024, 512, False, True, 128, 0.8),
    (128, 128, 128, True, False, 128, 0.0),
    (256, 256, 256, True, False, 128, 0.5),
    (2048, 1024, 512, True, False, 128, 0.8),
    (128, 128, 128, True, True, 128, 0.0),
    (256, 256, 256, True, True, 128, 0.5),
    (2048, 1024, 512, True, True, 128, 0.8)
)


def _dense_and_sparse(rows, cols, sparsity, blocking, std=0.1):
    mask = stk.random.dense_mask(rows, cols, sparsity, blocking)
    dense = (torch.randn(rows, cols) * std * mask).type(torch.float16)
    sparse = stk.ops.to_sparse(dense, blocking)
    cuda_device = torch.device("cuda")
    return (dense.to(cuda_device).requires_grad_(True),
            sparse.to(cuda_device).requires_grad_(True))


def _dense(rows, cols, std=0.1):
    cuda_device = torch.device("cuda")
    out = (torch.randn(rows, cols) * std).type(torch.float16)
    return out.to(cuda_device).requires_grad_(True)


def _dense_2x(rows, cols):
    a = _dense(rows, cols)
    return a, a.detach().requires_grad_(True)


def _with_transpose(op, a, b, trans_a, trans_b):
    a = a.t() if trans_a else a
    b = b.t() if trans_b else b
    return op(a, b)


def _mmm(a, b, topo):
    mask = stk.ops.to_dense(stk.ops.ones_like(topo))
    return torch.mm(a, b) * mask


def _sparse_out_with_transpose(op, a, b, topo, trans_a, trans_b):
    a = a.t() if trans_a else a
    b = b.t() if trans_b else b
    return op(a, b, topo)


def _mask(x, mask):
    mask = stk.ops.to_dense(stk.ops.ones_like(mask))
    return x * mask


@parameterized.parameters(*_LINEAR_OP_TESTS)
class LinearOpsTest(parameterized.TestCase):

    def testLinearOps_Dsd(self, m, k, n, trans_a, trans_b, blocking, sparsity):
        # Construct the operands.
        a_shape = (k, m) if trans_a else (m, k)
        a_dense, a = _dense_and_sparse(*a_shape, sparsity, blocking)
        b_shape = (n, k) if trans_b else (k, n)
        b, bcp = _dense_2x(*b_shape)

        # Execute the matmul.
        out = _with_transpose(stk.ops.dsd, a, b, trans_a, trans_b)
        expected_out = _with_transpose(torch.mm, a_dense, bcp, trans_a, trans_b)

        # Compute the gradients w.r.t. the inputs.
        expected_out.sum().backward()
        out.sum().backward()

        # Validate the results.
        self.assertEqual(out.dim(), 2)
        self.assertEqual(expected_out.size()[0], out.size()[0])
        self.assertEqual(expected_out.size()[1], out.size()[1])
        self.assertTrue(allclose(out, expected_out))

        # LHS gradient.
        grad = stk.ops.to_dense(a.grad)
        expected_grad = _mask(a_dense.grad, a.grad)
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

        # RHS gradient.
        grad = b.grad
        expected_grad = bcp.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

    def testLinearOps_Dds(self, m, k, n, trans_a, trans_b, blocking, sparsity):
        # Construct the operands.
        a_shape = (k, m) if trans_a else (m, k)
        a, acp = _dense_2x(*a_shape)
        b_shape = (n, k) if trans_b else (k, n)
        b_dense, b = _dense_and_sparse(*b_shape, sparsity, blocking)

        # Execute the matmul.
        out = _with_transpose(stk.ops.dds, a, b, trans_a, trans_b)
        expected_out = _with_transpose(torch.mm, acp, b_dense, trans_a, trans_b)

        # Compute the gradients w.r.t. the inputs.
        expected_out.sum().backward()
        out.sum().backward()

        # Validate the results.
        self.assertEqual(out.dim(), 2)
        self.assertEqual(expected_out.size()[0], out.size()[0])
        self.assertEqual(expected_out.size()[1], out.size()[1])
        self.assertTrue(allclose(out, expected_out))

        # LHS gradient.
        grad = a.grad
        expected_grad = acp.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

        # RHS gradient.
        grad = stk.ops.to_dense(b.grad)
        expected_grad = _mask(b_dense.grad, b.grad)
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

    def testLinearOps_Sdd(self, m, k, n, trans_a, trans_b, blocking, sparsity):
        # Construct the operands.
        a_shape = (k, m) if trans_a else (m, k)
        a, acp = _dense_2x(*a_shape)
        b_shape = (n, k) if trans_b else (k, n)
        b, bcp = _dense_2x(*b_shape)
        _, topo = _dense_and_sparse(m, n, sparsity, blocking)

        # Execute the matmul.
        out = _sparse_out_with_transpose(stk.ops.sdd, a, b, topo, trans_a, trans_b)
        expected_out = _sparse_out_with_transpose(_mmm, acp, bcp, topo, trans_a, trans_b)

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
        grad = a.grad
        expected_grad = acp.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

        # RHS gradient.
        grad = b.grad
        expected_grad = bcp.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

    def testLinearOps_Ssd(self, m, k, n, trans_a, trans_b, blocking, sparsity):
        # Construct the operands.
        a_shape = (k, m) if trans_a else (m, k)
        a_dense, a = _dense_and_sparse(*a_shape, sparsity, blocking)
        b_shape = (n, k) if trans_b else (k, n)
        b, bcp = _dense_2x(*b_shape)
        _, topo = _dense_and_sparse(m, n, sparsity, blocking)

        # Execute the matmul.
        out = _sparse_out_with_transpose(stk.ops.ssd, a, b, topo, trans_a, trans_b)
        expected_out = _sparse_out_with_transpose(_mmm, a_dense, bcp, topo, trans_a, trans_b)

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
        expected_grad = _mask(a_dense.grad, a.grad)
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

        # RHS gradient.
        grad = b.grad
        expected_grad = bcp.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

    def testLinearOps_Sds(self, m, k, n, trans_a, trans_b, blocking, sparsity):
        # Construct the operands.
        a_shape = (k, m) if trans_a else (m, k)
        a, acp = _dense_2x(*a_shape)
        b_shape = (n, k) if trans_b else (k, n)
        b_dense, b = _dense_and_sparse(*b_shape, sparsity, blocking)
        _, topo = _dense_and_sparse(m, n, sparsity, blocking)

        # Execute the matmul.
        out = _sparse_out_with_transpose(stk.ops.sds, a, b, topo, trans_a, trans_b)
        expected_out = _sparse_out_with_transpose(_mmm, acp, b_dense, topo, trans_a, trans_b)

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
        grad = a.grad
        expected_grad = acp.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

        # RHS gradient.
        grad = stk.ops.to_dense(b.grad)
        expected_grad = _mask(b_dense.grad, b.grad)
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

    def testLinearOps_Dss(self, m, k, n, trans_a, trans_b, blocking, sparsity):
        # Construct the operands.
        a_shape = (k, m) if trans_a else (m, k)
        a_dense, a = _dense_and_sparse(*a_shape, sparsity, blocking)
        b_shape = (n, k) if trans_b else (k, n)
        b_dense, b = _dense_and_sparse(*b_shape, sparsity, blocking)

        # Execute the matmul.
        out = _with_transpose(stk.ops.dss, a, b, trans_a, trans_b)
        expected_out = _with_transpose(torch.mm, a_dense, b_dense, trans_a, trans_b)

        # Compute the gradients w.r.t. the inputs.
        expected_out.sum().backward()
        out.sum().backward()

        # Validate the results.
        self.assertEqual(out.dim(), 2)
        self.assertEqual(expected_out.size()[0], out.size()[0])
        self.assertEqual(expected_out.size()[1], out.size()[1])
        self.assertTrue(allclose(out, expected_out))

        # LHS gradient.
        grad = stk.ops.to_dense(a.grad)
        expected_grad = _mask(a_dense.grad, a.grad)
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))

        # RHS gradient.
        grad = stk.ops.to_dense(b.grad)
        expected_grad = _mask(b_dense.grad, b.grad)
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size()[0], grad.size()[0])
        self.assertEqual(expected_grad.size()[1], grad.size()[1])
        self.assertTrue(allclose(grad, expected_grad))


if __name__ == '__main__':
    unittest.main()
