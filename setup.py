import os
from setuptools import setup

from distutils.extension import Extension
from distutils.sysconfig import get_python_inc

from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(
        'pysfm._gencut',
        sources = [
            'pysfm/_gencut.pyx',
            'core/python/gencut.cpp'
            ],
        include_dirs = [numpy.get_include(), get_python_inc(), os.path.abspath('.')],
        language = 'c++',
        extra_compile_args = ['-std=c++11', '-std=c++14']
    )
]


setup(
    name = 'pysfm',
    author = 'Kentaro Minami',
    author_email = '',
    url = '',
    license = '',
    description = 'Submodular function optimization package',
    long_description = '',
    packages = ['pysfm'],
    ext_modules = cythonize(extensions),
    cmdclass = { 'build_ext': build_ext },
    classifiers = [
        'Development Status :: 1 - Planning',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics'
    ]
)
