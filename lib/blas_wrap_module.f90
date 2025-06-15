! -----------------------------------------------------------------------
! This file (blas_wrap_module.f90) is part of DNNF90.
! (MIT License; see LICENSE at the repository root.)
!
! One matrix product behind an explicit interface.
!
! Callers hand whole arrays to this wrapper, whose dummies are explicit
! shape, and the wrapper alone speaks to BLAS.  One compilation unit for
! the one implicit interface: no caller's module infers its own
! interface for dgemm, and a BLAS-free build swaps in the reference
! kernels below without touching any caller.
! -----------------------------------------------------------------------
MODULE blas_wrap_module
  implicit none
  private
  public :: bgemm, bposv

CONTAINS

  !> Solve A x = b for symmetric positive definite A, in place (b <- x,
  !! A is destroyed).  With BLAS=1 this is LAPACK's dposv; without, a
  !! reference Cholesky, so the module still links against nothing.
  !! The Levenberg-Marquardt normal equations are the caller: their
  !! matrix is J^T J + mu I, which is SPD for any mu > 0.
  SUBROUTINE bposv( n, A, b, ok )
    implicit none
    integer,intent(IN) :: n
    real(8),intent(INOUT) :: A(n,n)
    real(8),intent(INOUT) :: b(n)
    logical,intent(OUT) :: ok
    integer :: info, i, j, k
    real(8) :: s
#ifdef USE_BLAS
    call dposv( 'L', n, 1, A, n, b, n, info )
    ok = ( info == 0 )
#else
    ! Cholesky A = L L^T in the lower triangle, then two substitutions.
    ok = .true.
    do j = 1, n
       s = A(j,j)
       do k = 1, j-1
          s = s - A(j,k)**2
       end do
       if ( s <= 0.d0 ) then
          ok = .false.
          return
       end if
       A(j,j) = sqrt(s)
       do i = j+1, n
          s = A(i,j)
          do k = 1, j-1
             s = s - A(i,k)*A(j,k)
          end do
          A(i,j) = s/A(j,j)
       end do
    end do
    do i = 1, n                      ! L y = b
       s = b(i)
       do k = 1, i-1
          s = s - A(i,k)*b(k)
       end do
       b(i) = s/A(i,i)
    end do
    do i = n, 1, -1                  ! L^T x = y
       s = b(i)
       do k = i+1, n
          s = s - A(k,i)*b(k)
       end do
       b(i) = s/A(i,i)
    end do
#endif
  END SUBROUTINE bposv

  !> C(1:m,1:n) = op(A) op(B), with alpha = 1 and beta = 0.
  SUBROUTINE bgemm( ta, tb, m, n, k, A, lda, B, ldb, C, ldc )
    implicit none
    character(1),intent(IN) :: ta, tb
    integer,intent(IN) :: m, n, k, lda, ldb, ldc
    real(8),intent(IN) :: A(lda,*), B(ldb,*)
    real(8),intent(OUT) :: C(ldc,*)
    integer :: i, j, p
    real(8) :: acc, aval, bval
#ifdef USE_BLAS
    call dgemm( ta, tb, m, n, k, 1.d0, A, lda, B, ldb, 0.d0, C, ldc )
#else
    ! Reference product, so that the module links without a BLAS.  The
    ! library's own loop kernels are used in that configuration, so this
    ! path exists for completeness rather than for speed.
    do j=1,n
       do i=1,m
          acc = 0.d0
          do p=1,k
             if ( ta == 'T' ) then
                aval = A(p,i)
             else
                aval = A(i,p)
             end if
             if ( tb == 'T' ) then
                bval = B(j,p)
             else
                bval = B(p,j)
             end if
             acc = acc + aval*bval
          end do
          C(i,j) = acc
       end do
    end do
#endif
  END SUBROUTINE bgemm

END MODULE blas_wrap_module
