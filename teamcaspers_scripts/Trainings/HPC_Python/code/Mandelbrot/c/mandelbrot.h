#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <complex.h>

int init(double complex* q, double xmin, double xmax, double ymin, double ymax, int xres, int yres);

void calculate(int* iterations, double complex* q, int n, int maxiter);

#endif

