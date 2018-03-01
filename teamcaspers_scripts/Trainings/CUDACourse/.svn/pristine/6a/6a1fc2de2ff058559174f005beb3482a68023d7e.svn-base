      program MMDriver
      
      use mymm

      real, dimension(:,:),allocatable :: a, b, c 
      integer nargs,n
      character(len=10) value

      nargs = command_argument_count()
      if (nargs > 0) then
          call get_command_argument(1, value)
          read(value, fmt = '(i5)') n
      else
          n = 2000
      endif

      allocate(a(n,n))
      allocate(b(n,n))
      allocate(c(n,n))
     
      call mm1(a, b, c, n)

      end program MMDriver
