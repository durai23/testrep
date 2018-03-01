from numba import cuda
import numpy as np
from pylab import imshow, show
 
@cuda.jit('int32(float64, float64, int32)', device=True)
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = complex(0, 0)
    for i in range(max_iters):
        z = z*z + c
        if z.real * z.real + z.imag * z.imag >= 4:
            return i
    return 255
 
@cuda.autojit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]
 
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color
 
image = np.zeros((500, 750), dtype=np.uint8)
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
imshow(image)
show()