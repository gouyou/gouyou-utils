__author__ = 'Guillaume Taglang <guillaume@taglang.org>'

import pytest

import numpy as np
import scipy.sparse as sp

from gouyou.utils.sparse.operations import csr_vector_matrix_and

testdata = [
    (
        sp.random(1, 5000000, format='csr').astype('bool'),
        sp.random(3, 5000000, format='csr').astype('bool')
    ),
    (
        sp.random(1, 5000000, density=0, format='csr').astype('bool'),
        sp.random(3, 5000000, format='csr').astype('bool')
    ),
    (
        sp.random(1, 5000000, format='csr').astype('bool'),
        sp.random(3, 5000000, density=0, format='csr').astype('bool')
    ),
    (
        sp.random(1, 5000000, density=0, format='csr').astype('bool'),
        sp.random(3, 5000000, density=0, format='csr').astype('bool')
    ),
    (
        sp.random(1, 5000000, density=1, format='csr').astype('bool'),
        sp.random(3, 5000000, density=1, format='csr').astype('bool')
    ),
]

benchmarkdata = [
    (
        sp.random(1, 5000000, format='csr').astype('bool'),
        sp.random(1000, 5000000, format='csr').astype('bool')
    ),
    (
        sp.random(1, 5000000, density=0.5, format='csr').astype('bool'),
        sp.random(1000, 5000000, format='csr').astype('bool')
    ),
    (
        sp.random(1, 5000000, format='csr').astype('bool'),
        sp.random(1000, 5000000, density=0.0001, format='csr').astype('bool')
    ),
]


# default implementation
def _multiply(a, b):
    r_ = []
    for i in xrange(b.shape[0]):
        r_.append(a.multiply(b.getrow(i)))
    return sp.vstack(r_)


@pytest.mark.parametrize("a,b", testdata)
def test_csr_vector_matrix_and(a, b):
    r = csr_vector_matrix_and(a, b)

    r_ = []
    for i in xrange(b.shape[0]):
        r_.append(a.multiply(b.getrow(i)))
    r_ = sp.vstack(r_)

    assert r.dtype == np.bool_
    assert r.shape == r_.shape
    assert r.getnnz() == r_.getnnz()
    for i in xrange(b.shape[0]):
        assert r.getrow(i).getnnz() == r_.getrow(i).getnnz()


# ------------------------------------------------------------------- benchmarks
@pytest.mark.parametrize("a,b", benchmarkdata)
def test_multiply_matrix_benchmark(benchmark, a, b):
    benchmark(_multiply, a, b)

@pytest.mark.parametrize("a,b", benchmarkdata)
def test_csr_vector_matrix_and_benchmark(benchmark, a, b):
    benchmark(csr_vector_matrix_and, a, b)

