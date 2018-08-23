#include "mandelbrot.h"
#include <vector>
#include <iostream>
#include <complex>

Mandelbrot::Mandelbrot(double xmin, double xmax, double ymin, double ymax, int xres, int yres,
                       int maxiter) :
    iterations((xres) * (yres)),
    maximumNumberOfIterations(maxiter),
    minRealValue(xmin),
    maxRealValue(xmax),
    minImagValue(ymin),
    maxImagValue(ymax),
    resRealAxis(xres),
    resImagAxis(yres),
    dx(double(xmax - xmin) / (xres - 1)),
    dy(double(ymax - ymin) / (yres - 1)),
    q((xres) * (yres))
{
    size_t idx = 0;
    for (double real = minRealValue; real < maxRealValue; real += dx){
        for (double imag = minImagValue; imag < maxImagValue; imag += dy){
            q[idx++] = std::complex<double>(real, imag);
        }
    }
}

void Mandelbrot::calculate(){
    const int unrolldepth = 4;
    const std::complex<double> zero(0.0, 0.0);
    for (size_t i = 0; i <= (q.size() - 4); i += unrolldepth){
        std::complex<double> z[4];
        std::complex<double> c[4];
        z[0] = 0;
        z[1] = 0;
        z[2] = 0;
        z[3] = 0;

        c[0] = q[i];
        c[1] = q[i + 1];
        c[2] = q[i + 2];
        c[3] = q[i + 3];
        for (int j = 0; j < maximumNumberOfIterations; ++j){
            z[0] *= z[0];
            z[0] += c[0];
            z[1] *= z[1];
            z[1] += c[1];
            z[2] *= z[2];
            z[2] += c[2];
            z[3] *= z[3];
            z[3] += c[3];
//            z[0] = z[0] * z[0] + c[0];
//            z[1] = z[1] * z[1] + c[1];
//            z[2] = z[2] * z[2] + c[2];
//            z[3] = z[3] * z[3] + c[3];

            if (norm(z[0]) > 4.0){
                iterations[i] = j;
                z[0] = zero;
                c[0] = zero;
            }
            if (norm(z[1]) > 4.0){
                iterations[i + 1] = j;
                z[1] = zero;
                c[1] = zero;
            }
            if (norm(z[2]) > 4.0){
                iterations[i + 2] = j;
                z[2] = zero;
                c[2] = zero;
            }
            if (norm(z[3]) > 4.0){
                iterations[i + 3] = j;
                z[3] = zero;
                c[3] = zero;
            }
            if ((z[0] == zero) && (z[1] == zero) && (z[2] == zero) && (z[3] == zero)) break;
        }
    }
}

