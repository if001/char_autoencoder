from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension("cy_preprocessing", ["cy_preprocessing.pyx"])
]

setup(
    name="Sample calculate app",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()]
)
