#!/bin/bash
for n in 64 256 1024 4096 8192; 
do 
    ./mandelbrot $n
done | tee mandelbrot_bench.dat
#python Mandelbrot.py gpu &
python mandelbrot_bench.py &
