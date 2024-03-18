from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
    ext_modules=cythonize(["Tools.pyx", "ConstrutiveHeuristics.pyx"]),
    include_dirs=[numpy.get_include()]
)
