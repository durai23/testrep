from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("mandelbrot", ["mandelbrot_c.pyx", "mandelbrot.c"],
extra_compile_args=["-std=c99"],
include_dirs=['/usr/lib64/python2.7/site-packages/numpy/core/include'])]

setup(
    name = 'Mandelbrot set',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
		      )
