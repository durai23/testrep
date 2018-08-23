program main

    use mandelbrot

    call init(-2.13, 0.77, -1.3, 1.3, 1200, 1200, 1000)
    call calculate()
!    print *, iterations
end program
