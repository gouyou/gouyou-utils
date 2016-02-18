#cython: language_level=3, boundscheck=False, wraparound=False

__author__ = 'Guillaume Taglang <guillaume@taglang.org>'

import numpy as np
cimport numpy as np
import scipy.sparse as sp

from cython cimport view
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

def cs_vector_vector_and(a, b, eliminate_zeros=False):
    if a.format == 'csr':
        matrix = sp.csr_matrix
    elif a.format == 'csc':
        matrix = sp.csc_matrix
    else:
        raise 'Matrix format not supported'

    if eliminate_zeros:
        a.eliminate_zeros()
        b.eliminate_zeros()

    cdef int i_a = 0
    cdef int i_b = 0
    cdef int s_a = a.indices.shape[0]
    cdef int s_b = b.indices.shape[0]
    cdef int indices_length = 0

    cdef int [:] indices = np.empty(max(s_a, s_b), dtype=np.int32)
    cdef int [:] a_indices = a.indices
    cdef int [:] b_indices = b.indices


    while i_a < s_a and i_b < s_b:
        if a_indices[i_a] < b_indices[i_b]:
            i_a += 1
        elif b_indices[i_b] < a_indices[i_a]:
            i_b += 1
        else:
            indices[indices_length] = a_indices[i_a]
            indices_length += 1
            i_a += 1
            i_b += 1

    return matrix(
        (
            np.ones(indices_length, dtype='bool'),
            indices[:indices_length],
            [0,indices_length]
        ),
        (a.shape[0], a.shape[1]),
        dtype='bool'
    )


def csr_vector_matrix_and(a, b):
    cdef int i
    cdef int j
    cdef int i_a
    cdef int i_b
    cdef int s_a = a.indices.shape[0]
    cdef int s_b
    cdef int b_rows = b.shape[0]

    cdef int[:] a_indices = a.indices
    cdef int[:] b_indices = b.indices
    cdef int[:] b_indptr = b.indptr

    cdef int **indices_ = <int **>malloc(b_rows * sizeof(int *))
    cdef int[:] indices_lengths = np.zeros(b_rows, dtype=np.int32)
    cdef int[::view.contiguous] final_indices
    cdef int[:] indptr = np.empty(b_rows + 1, dtype=np.int32)

    for i in prange(b_rows, nogil=True):
        i_a = 0
        i_b = b_indptr[i]
        s_b = b_indptr[i+1]
        indices_[i] = <int *>malloc(min(s_a, s_b)*sizeof(int))

        while i_a < s_a and i_b < s_b:
            if a_indices[i_a] < b_indices[i_b]:
                i_a = i_a + 1
            elif b_indices[i_b] < a_indices[i_a]:
                i_b = i_b + 1
            else:
                indices_[i][indices_lengths[i]] = a_indices[i_a]
                indices_lengths[i] = indices_lengths[i] + 1
                i_a = i_a + 1
                i_b = i_b + 1

    indptr[0] = 0
    for i in xrange(b_rows):
        indptr[i+1] = indptr[i] + indices_lengths[i]

    final_indices = np.empty(indptr[b_rows], dtype=np.int32)
    for i in xrange(b_rows):
        memcpy(&(final_indices[indptr[i]]), (indices_[i]), indices_lengths[i] * sizeof(int))
        free(indices_[i])
    free(indices_)

    return sp.csr_matrix(
        (np.ones(indptr[b_rows], dtype='bool'), final_indices, indptr),
        b.shape,
        dtype='bool'
    )