module mandelbrot

complex, allocatable, dimension(:) :: q
integer, allocatable, dimension(:) :: iterations
integer maximumNumberOfIterations, numberOfPoints

contains
    subroutine init(minRealValue, maxRealValue, minImagValue, maxImagValue, xres, yres, maxiter)

        real, intent(in) :: minRealValue, maxRealValue, minImagValue, maxImagValue
        real :: dx, dy
        integer, intent(in) :: xres, yres, maxiter
        integer ::  idx, i, j
        complex :: XJ = (0, 1)
        
        numberOfPoints = xres * yres
        dx = (maxRealValue - minRealValue) / (xres - 1)
        dy = (maxImagValue - minImagValue) / (yres - 1)
        maximumNumberOfIterations = maxiter
        
        allocate(q(numberOfPoints))
        allocate(iterations(numberOfPoints))
        
        idx = 1
        do i = 0, xres - 1
            do j = 0, yres - 1
                q(idx) = cmplx(minRealValue + i * dx,  minImagValue  + j * dy)
                idx = idx + 1
            end do
        end do
        
    end subroutine init
        
    subroutine calculate()

        complex :: z, c
        integer i, j
        do i = 1, numberOfPoints
            z = (0.0, 0.0)
            c = q(i)
            do j = 1, maximumNumberOfIterations
                z = z * z + c
                if (real(z * conjg(z)) > 4.0) then
                    iterations(i) = j
                    exit
                end if
            end do
        end do
    end subroutine calculate

end module mandelbrot
