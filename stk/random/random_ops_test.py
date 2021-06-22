import unittest

from absl.testing import parameterized
import stk
import torch


# TODO(tgale): Expand test cases.
@parameterized.parameters((8, 16, 0.0, 1))
class RandomOpsTest(parameterized.TestCase):

    def testRandomOps_DenseMask(self, rows, cols, sparsity, blocking):
        mask = stk.random.dense_mask(
            rows, cols, sparsity, blocking)

        # Validate the shape.
        self.assertEqual(mask.dim(), 2)
        self.assertEqual(mask.size()[0], rows)
        self.assertEqual(mask.size()[1], cols)

        # Validate the sparsity
        numblocks = rows // blocking * cols // blocking
        nnz = round(numblocks * (1 - sparsity))
        self.assertEqual(
            torch.count_nonzero(mask).item(),
            nnz)

        # Check values are zero or one.
        self.assertTrue(
            torch.all(torch.logical_or(
                torch.eq(mask, 0),
                torch.eq(mask, 1))))

    def testRandomOps_SparseMask(self, rows, cols, sparsity, blocking):
        mask = stk.random.sparse_mask(
            rows, cols, sparsity, blocking)

        # Validate the matrix.
        mask.validate()

        # Validate the shape.
        self.assertEqual(mask.dim(), 2)
        self.assertEqual(mask.size()[0], rows)
        self.assertEqual(mask.size()[1], cols)

        # Validate the sparsity.
        numblocks = rows // blocking * cols // blocking
        nnz = round(numblocks * (1 - sparsity))
        self.assertEqual(mask.nnz, nnz)
        
        # Check values are zero or one.
        self.assertTrue(
            torch.all(torch.logical_or(
                torch.eq(mask.data, 0),
                torch.eq(mask.data, 1))))

        
if __name__ == '__main__':
    unittest.main()
