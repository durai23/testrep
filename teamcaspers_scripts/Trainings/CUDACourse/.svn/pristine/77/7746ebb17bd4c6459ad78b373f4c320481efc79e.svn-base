FC=pgf90
CFLAGS=-fast -Minfo=all
ACCFLAGS=-Minfo=accel -ta=nvidia,cuda3.2,time

all: acc cpu

acc: mm.f90 mmdriv.f90
	${FC} -o mm ${CFLAGS} ${ACCFLAGS} mm.f90 mmdriv.f90 

cpu: mm_cpu.f90 mmdriv.f90
	${FC} -o mm_cpu ${CFLAGS} -Mconcur mm_cpu.f90 mmdriv.f90 

clean:
	rm -f mm mm_cpu *.o *.mod
