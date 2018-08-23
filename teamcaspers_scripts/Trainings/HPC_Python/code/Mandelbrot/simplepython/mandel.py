#!/usr/bin/env python
from numba import jit

@jit
def mandel_iteration(c, max_iter = 20):
    z = 0
    for i in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return i
    return max_iter

def mandelbrot(xmin, xmax, ymin, ymax, xres, yres, max_iter):
    M = [[0 for j in range(yres)] for i in range(xres)]
    dx = (xmax - xmin) / xres 
    dy = (ymax - ymin) / yres 
    for i in range(xres):
        for j in range(yres):
            c = xmin + i * dx + (ymin + j * dy) * 1j
            M[i][j] = mandel_iteration(c, max_iter)
    return M


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    M = mandelbrot(-2.3, 2, -1.3, 1.3, 800, 600, 20)
    plt.imshow(M)
    plt.savefig("new_mandel.png")
