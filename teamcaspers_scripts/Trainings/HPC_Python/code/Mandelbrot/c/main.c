#include <stdlib.h>
#include <complex.h>
#include "mandelbrot.h"

int main(int argc, char** argv){
	int xres = 1200;
	int yres = 1200;
	double complex* q = malloc(xres * yres * sizeof(double complex));
	int* iterations = malloc(xres * yres * sizeof(int));
	init(q, -2.13, 0.77, -1.3, 1.3, xres, yres);
	calculate(iterations, q, xres * yres, 1000);
	return 0;
}
