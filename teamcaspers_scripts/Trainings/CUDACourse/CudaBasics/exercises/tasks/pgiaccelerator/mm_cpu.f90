! Simple matmul example

      module mymm
      contains
      subroutine mm1(a, b, c, m)
        real, dimension(:,:) :: a, b, c
        do j = 1, m
           do i = 1, m
               a(i, j) = 0.0
           enddo
           do k = 1, m
               do i = 1, m 
                   a(i, j) = a(i, j) + b(i, k) * c(k, j)
               enddo
           enddo
        enddo
      end subroutine
      end module 
