__author__ = 'Guillaume Taglang <guillaume@taglang.org>'

import pytest

import numpy as np
import scipy.sparse as sp

from gouyou.utils.sparse.operations import cs_vector_vector_and

testdata = [
    (
        sp.random(1, 10000, density=0.1, format='csr').astype('bool'),
        sp.random(1, 10000, density=0.1, format='csr').astype('bool')
    ),
    (
        sp.random(1, 100, density=0, format='csr').astype('bool'),
        sp.random(1, 100, format='csr').astype('bool')
    ),
    (
        sp.random(1, 100, format='csr').astype('bool'),
        sp.random(1, 100, density=0, format='csr').astype('bool')
    ),
    (
        sp.random(1, 100, density=0, format='csr').astype('bool'),
        sp.random(1, 100, density=0, format='csr').astype('bool')
    ),
    (
        sp.random(1, 100, density=1, format='csr').astype('bool'),
        sp.random(1, 100, density=1, format='csr').astype('bool')
    ),
    (
        sp.random(10000, 1, density=0.1, format='csc').astype('bool'),
        sp.random(10000, 1, density=0.1, format='csc').astype('bool')
    ),
]

benchmarkdata = [
    testdata[0],
    (
        sp.random(1, 5000000, density=0.5, format='csr').astype('bool'),
        sp.random(1, 5000000, format='csr').astype('bool')
    )
]

@pytest.mark.parametrize("a,b", testdata)
def test_csr_vector_vector_and(a, b):
    r = cs_vector_vector_and(a, b)
    r_ = a.multiply(b)
    assert r.dtype == np.bool_
    assert r.shape == r_.shape
    assert r.getnnz() == r_.getnnz()


# ------------------------------------------------------------------- benchmarks
@pytest.mark.parametrize("a,b", benchmarkdata)
def test_multiply_vector_benchmark(benchmark, a, b):
    benchmark(a.multiply, b)


@pytest.mark.parametrize("a,b", benchmarkdata)
def test_csr_vector_vector_and_benchmark(benchmark, a, b):
    benchmark(cs_vector_vector_and, a, b)