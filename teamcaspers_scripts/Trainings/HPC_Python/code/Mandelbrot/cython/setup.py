from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("mandelbrot_3", ["Mandelbrot_3.pyx"],
include_dirs=['/usr/lib64/python2.7/site-packages/numpy/core/include'])]

setup(
    name = 'Mandelbrot set',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
		      )
