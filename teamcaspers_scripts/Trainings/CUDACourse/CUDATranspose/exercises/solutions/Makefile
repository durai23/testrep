all: run

transpose: transpose.cu Makefile
	nvcc -lineinfo -arch=sm_37 -o transpose transpose.cu

run: transpose
	./transpose

profile: transpose
	nvprof --analysis-metrics -o transpose.nvvp ./transpose

clean:
	rm -f transpose transpose.nvvp
