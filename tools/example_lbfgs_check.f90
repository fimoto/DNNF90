! Does the two-loop recursion produce the direction it should?
!
! On a quadratic f(w) = w.A w /2 - b.w with A positive definite, L-BFGS
! with n exact pairs reproduces the Newton direction -A^{-1} g, so the
! recursion can be checked against a direct solve.  With fewer pairs it
! reproduces it in the subspace those pairs span, which is harder to
! state, so the test uses n pairs and asks for the Newton direction.
!
! A wrong sign or a mis-ordered loop in the recursion still produces a
! plausible-looking vector, which is why this is worth checking rather
! than reading.
program verify_lbfgs
  use optimizer_module, only: lbfgs_direction
  implicit none
  integer,parameter :: N = 6
  real(8) :: A(N,N), b(N), w(N), g(N), d(N), dnewton(N)
  real(8) :: sv(N,N), yv(N,N), rhov(N)
  real(8) :: M(N,N), rhs(N), sy, err
  integer :: i, j, k

  ! A diagonal A, so that the coordinate directions used as the pairs
  ! below are A-conjugate.  That is what the test needs: L-BFGS with n
  ! conjugate pairs reproduces the Newton direction exactly, whereas with
  ! n arbitrary pairs it does not -- so a check built on arbitrary pairs
  ! would report a failure of its own making.
  call random_seed()
  A = 0.d0
  do i = 1, N
     A(i,i) = dble(i)*2.d0 + 1.d0        ! spread eigenvalues
  end do
  call random_number( b )

  ! pairs from n independent directions: s_k arbitrary, y_k = A s_k,
  ! which is what the true Hessian gives
  sv = 0.d0
  do k = 1, N
     sv(k,k) = 1.d0 + 0.3d0*dble(k)
     yv(1:N,k) = matmul( A, sv(1:N,k) )
     sy = dot_product( sv(1:N,k), yv(1:N,k) )
     rhov(k) = 1.d0/sy
  end do

  call random_number( w )
  g = matmul( A, w ) - b

  call lbfgs_direction( g, d, sv, yv, rhov, N, N )

  ! the Newton direction, by a direct solve
  M = A;  rhs = -g
  call solve( M, rhs, N )
  dnewton = rhs

  err = 0.d0
  do i = 1, N
     err = max( err, abs( d(i) - dnewton(i) ) )
  end do
  write(*,'(a,e13.5)') " L-BFGS direction vs Newton, conjugate pairs: ", &
       err/max( maxval(abs(dnewton)), 1.d-300 )
  if ( err/max( maxval(abs(dnewton)), 1.d-300 ) > 1.d-8 ) then
     write(*,'(a)') " FAILED"
     stop 1
  end if
  write(*,'(a)') " passed"

CONTAINS

  SUBROUTINE solve( M, r, n )
    implicit none
    integer,intent(IN) :: n
    real(8),intent(INOUT) :: M(n,n), r(n)
    integer :: i, j, k
    real(8) :: f
    do k = 1, n
       do i = k+1, n
          f = M(i,k)/M(k,k)
          do j = k, n
             M(i,j) = M(i,j) - f*M(k,j)
          end do
          r(i) = r(i) - f*r(k)
       end do
    end do
    do i = n, 1, -1
       do j = i+1, n
          r(i) = r(i) - M(i,j)*r(j)
       end do
       r(i) = r(i)/M(i,i)
    end do
  END SUBROUTINE solve

end program verify_lbfgs
