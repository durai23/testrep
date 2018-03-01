from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules=[Extension(
                   "mandelbrot",                 # name of extension
                   ["Mandelbrot.pyx", "mandelbrot.cpp"], #  our Cython source
                   language="c++")],  # causes Cython to create C++ source
                   include_dirs=['/usr/lib/python2.6/site-packages/numpy/core/include', '/usr/lib64/python2.7/site-packages/numpy/core/include', '/usr/lib64/python2.6/site-packages/numpy/core/include'],
      cmdclass={'build_ext': build_ext})

