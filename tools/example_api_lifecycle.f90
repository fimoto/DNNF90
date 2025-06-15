! Is the API's derivative table the same before and after the first
! evaluation, and does init -> free -> init work?
!
! dnnf90_nderiv fixes the length a caller must allocate for
! dnnf90_eval_hod, so it has to answer the same number from the moment
! init returns.  Run this from a case directory.
program api_lifecycle
  use api_module, only: dnnf90_init, dnnf90_free, dnnf90_nderiv, dnnf90_eval_hod
  implicit none
  integer :: n0, n1
  real(8) :: x(1), t(64)
  call dnnf90_init
  n0 = dnnf90_nderiv()
  x = 0.3d0
  call dnnf90_eval_hod( x, t )
  n1 = dnnf90_nderiv()
  write(*,'(a,i0,a,i0)') ' nderiv after init = ', n0, '   after first eval = ', n1
  if ( n0 /= n1 ) write(*,'(a)') ' MISMATCH: the table changed size'
  call dnnf90_free
  call dnnf90_init
  write(*,'(a,i0)') ' nderiv after re-init = ', dnnf90_nderiv()
  call dnnf90_free
  write(*,'(a)') ' passed'
end program api_lifecycle
