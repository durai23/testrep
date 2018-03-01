#include "mandelbrot.h"
#include <vector>
#include <algorithm>
#include <iostream>
int main(int argc, char** argv){
    int xres = 1200;
    int yres = 1200;
    double xmin = -2.13;
    double xmax = 0.77;
    double ymin = -1.3;
    double ymax = 1.3;
    int maxiter = 1000;
    Mandelbrot myMandelbrot(xmin, xmax, ymin, ymax, xres, yres, maxiter);
    myMandelbrot.calculate();
//    for (std::vector<int>::iterator e = myMandelbrot.iterations.begin();
//         e != myMandelbrot.iterations.end(); ++e){
//        std::cout << *e << '\n';
//    }
}
