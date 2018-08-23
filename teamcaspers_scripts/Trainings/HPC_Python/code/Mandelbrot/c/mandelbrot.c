#include <complex.h>
#include <stdio.h>
int init(double complex* q, double xmin, double xmax, double ymin, double ymax, int xres, int yres){
	double dx = (double) (xmax - xmin) / (xres - 1);
	double dy = (double) (ymax - ymin) / (yres - 1);
	int idx = 0;
	for (int i = 0; i < xres; ++i){
		for (int j = 0; j < yres; ++j){
			q[idx++] = xmin + i * dx + (ymin + j * dy) * I;
		}
	}
	return xres * yres;
}


void calculate(int* iterations, double complex* q, int n, int maxiter){
	double complex z;
	double complex c;
	for (int i = 0; i < n; ++i){
		z = 0.0;
		c = q[i];
		for (int j = 0; j < maxiter; ++j){
		       z = z * z + c;
		       if (creal(z * conj(z)) > 4.0){
			       iterations[i] = j;
			       break;
		       }
		}
	}
}



