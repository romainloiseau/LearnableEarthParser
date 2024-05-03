from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize
import numpy

def get_extensions():
    return cythonize([
        Extension(
            "learnableearthparser.fast_sampler._sampler",
            [
                "learnableearthparser/fast_sampler/_sampler.pyx",
                "learnableearthparser/fast_sampler/sampling.cpp"
            ],
            language="c++11",
            libraries=["stdc++"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-std=c++11", "-O3"]
        )
    ])

setup(
    name="learnableearthparser",
    ext_modules=get_extensions(),
    packages=find_packages(exclude=["docs", "tests"]),
    include_dirs=[numpy.get_include()],
    install_requires=[],
)