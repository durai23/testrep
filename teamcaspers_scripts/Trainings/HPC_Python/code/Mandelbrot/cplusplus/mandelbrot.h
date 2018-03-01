#ifndef MANDELBROT_H
#define MANDELBROT_H
#include <complex>
#include <vector>

class Mandelbrot
{
public:
    /** Initialize the Mandelbrot set.
     *
     *  @param xmin minimum real value
     *  @param xmax maximum real value
     *  @param ymin minimum imaginary value
     *  @param ymax maximum imaginary value
     *  @param xres number of pixels to calculate. Step size on real axis is (xmax - xmin) / xres
     *  @param xres number of pixels to calculate. Step size on real axis is (ymax - ymin) / yres
     *  @param maxiter maximum number of iteration before a point is determined to not be a part of
     *         the mandelbrot set
     */
    Mandelbrot(double xmin = -2.13, double xmax = 0.77, double ymin = -1.3, double ymax = 1.3,
               int xres = 1200, int yres = 1200, int maxiter = 1000);

    void calculate();
    std::vector<int> iterations;

private:
    int maximumNumberOfIterations;
    double minRealValue;
    double maxRealValue;
    double minImagValue;
    double maxImagValue;
    int resRealAxis;
    int resImagAxis;
    double dx;
    double dy;
    std::vector<std::complex<double> > q;
};

#endif // MANDELBROT_H
