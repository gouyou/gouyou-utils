__author__ = 'Guillaume Taglang <guillaume@taglang.org>'

from setuptools import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

import numpy


setup(
    name='gouyou-utils',
    version='0.1',
    description='Various semi useful stuff',
    author='Guillaume Taglang',
    author_email='guillaume@taglang.org',
    license='MIT',

    packages=['gouyou', 'gouyou.utils', 'gouyou.utils.sparse'],

    ext_modules=[
        Extension(
            'gouyou.utils.sparse.operations',
            sources=['gouyou/utils/sparse/operations.pyx'],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
            include_dirs=[numpy.get_include()]
        )
    ],
    cmdclass = {'build_ext': build_ext},

    requires=['numpy', 'scipy'],
    setup_requires=['pytest-runner', 'cython'],
    tests_require=['pytest', 'pytest-benchmark', 'pytest-cov']
)