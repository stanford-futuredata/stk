from stk.matrix import Matrix

def mul(a, b):
    """
    Element-wise multiplication of matrices a and b.

    Note: it is the user's responsibility to make sure that a and b
        follow the same matrix topology.

        This function assumes it is safe to use the topoplogy of a.
    """
    assert isinstance(a, Matrix)
    assert isinstance(b, Matrix)
    assert a._size == b._size

    return Matrix(a.size(),
                  a.data * b.data,
                  a.row_indices,
                  a.column_indices,
                  a.offsets,
                  a.column_indices_t,
                  a.offsets_t,
                  a.block_offsets_t)
