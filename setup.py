import os
from setuptools import setup

from distutils.extension import Extension
from distutils.sysconfig import get_python_inc

from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(
        'pysfm._core',
        sources = [
            'pysfm/_core.pyx'
            ],
        include_dirs = [
            numpy.get_include(),
            get_python_inc(),
            os.path.abspath('./third_party/eigen3_3_4/'),
            os.path.abspath('.')
            ],
        language = 'c++',
        extra_compile_args = ['-std=c++11', '-std=c++14']
    ),
    Extension(
        'pysfm._gencut',
        sources = [
            'pysfm/_gencut.pyx'
            ],
        include_dirs = [numpy.get_include(), get_python_inc(), os.path.abspath('.')],
        language = 'c++',
        extra_compile_args = ['-std=c++11', '-std=c++14']
    )
]


setup(
    name = 'pysfm',
    author = 'Kentaro Minami',
    author_email = 'kentaro.minami1991@gmail.com',
    url = 'https://github.com/ktrmnm/SFM',
    license = 'Apache License Version 2.0',
    description = 'Submodular function optimization package',
    long_description = '',
    packages = ['pysfm'],
    ext_modules = cythonize(extensions),
    cmdclass = { 'build_ext': build_ext },
    classifiers = [
        'Development Status :: 1 - Planning',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: Apache Software License'
    ]
)
