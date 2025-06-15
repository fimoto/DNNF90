! Is the metric row of a system correct?
!
! The natural gradient builds its Gauss-Newton metric from rows
!
!     j_n = dL_n/dw,     L_n = (1/2) sum_r R_r(x_n)^2,
!
! one per collocation point.  If the row is wrong the metric is wrong and
! the step goes somewhere arbitrary, which on a large network looks like
! divergence and is easy to blame on the damping.  This checks the row
! itself, on a network small enough that the check is cheap: the row is
! compared with central differences of L_n in every weight.
!
! The residual is the Taylor-Green system, whose cross terms are the
! part that could go wrong: a term u du/dx seeds both the derivative slot
! of u and the value slot of u, and a term v du/dy seeds the derivative
! slot of u and the value slot of v.
program verify_ngd_multi
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, alpha_index
  use net_module, only: net_t, net_init, net_free, work_t, work_init, &
       work_free, net_eval_hod_multi
  use train_module, only: twork_t, grad_t, twork_init, grad_init, &
       twork_free, grad_free, net_forward_point, net_backward_point_multi
  use global_variables, only: sys_nres, sys_nterm, sys_res, sys_cmp, &
       sys_ind, sys_fac, sys_coeff, sys_has_src
  use pinn_module, only: calc_sys_residual, set_sys_seed
  implicit none
  integer,parameter :: D0 = 3, KMAX = 2, NC = 3, W = 6
  real(8),parameter :: NU = 0.1d0
  type(net_t) :: nt
  type(work_t) :: wk
  type(twork_t) :: tw
  type(grad_t) :: g
  real(8),allocatable :: tm(:,:), sm(:,:), tdum(:)
  real(8) :: x(D0), R(NC), src(NC)
  real(8) :: hw, lp, lm, fd, an, emax, gmax, wsave
  integer :: sd(D0,1), dims(4), l, j, k, nchk
  integer :: ix, iy, it, ixx, iyy

  sd = 0
  call init_hod_tables( D0, KMAX, 0, sd )
  dims = (/ D0, W, W, NC /)
  call net_init( nt, 4, dims )
  call work_init( wk, nt )
  call twork_init( tw, nt )
  call grad_init( g, nt )
  allocate( tm(NC,NUM_alpha), sm(NC,NUM_alpha), tdum(NUM_alpha) )

  ix  = alpha_index( (/ 1,0,0 /) );  iy  = alpha_index( (/ 0,1,0 /) )
  it  = alpha_index( (/ 0,0,1 /) )
  ixx = alpha_index( (/ 2,0,0 /) );  iyy = alpha_index( (/ 0,2,0 /) )

  sys_nres = 3;  sys_nterm = 0;  sys_has_src = .false.
  call add( 1, 1, it , 0,  1.d0 );  call add( 1, 1, ix , 1, 1.d0 )
  call add( 1, 1, iy , 2,  1.d0 );  call add( 1, 3, ix , 0, 1.d0 )
  call add( 1, 1, ixx, 0, -NU  );  call add( 1, 1, iyy, 0, -NU )
  call add( 2, 2, it , 0,  1.d0 );  call add( 2, 2, ix , 1, 1.d0 )
  call add( 2, 2, iy , 2,  1.d0 );  call add( 2, 3, iy , 0, 1.d0 )
  call add( 2, 2, ixx, 0, -NU  );  call add( 2, 2, iyy, 0, -NU )
  call add( 3, 1, ix , 0,  1.d0 );  call add( 3, 2, iy , 0, 1.d0 )

  call random_seed()
  call random_number( nt%w );  nt%w = 0.6d0*( nt%w - 0.5d0 )
  x = (/ 0.41d0, -0.27d0, 0.13d0 /)
  src = 0.d0

  ! the row, built the way the trainer builds it: unit factor, residuals
  g%nabla = 0.d0
  call net_forward_point( nt, tw, x, tdum )
  call net_eval_hod_multi( nt, wk, x, tm )
  call calc_sys_residual( tm, src, R )
  call set_sys_seed( tm, R, 1.d0, sm )
  call net_backward_point_multi( nt, tw, sm, g )

  hw = 1.d-5;  emax = 0.d0;  gmax = 0.d0;  nchk = 0
  do l = 2, nt%nlayer
     do j = 1, nt%ndim(l)
        do k = 0, nt%ndim(l-1)
           wsave = nt%w(l,j,k)
           nt%w(l,j,k) = wsave + hw
           call net_eval_hod_multi( nt, wk, x, tm )
           call calc_sys_residual( tm, src, R )
           lp = 0.5d0*sum( R(1:NC)**2 )
           nt%w(l,j,k) = wsave - hw
           call net_eval_hod_multi( nt, wk, x, tm )
           call calc_sys_residual( tm, src, R )
           lm = 0.5d0*sum( R(1:NC)**2 )
           nt%w(l,j,k) = wsave
           fd = ( lp - lm )/( 2.d0*hw )
           an = g%nabla(l,j,k)
           emax = max( emax, abs(an-fd) )
           gmax = max( gmax, abs(fd) )
           nchk = nchk + 1
        end do
     end do
  end do

  write(*,'(a,i0,a,e13.5)') " metric row vs FD of L_n, ", nchk, &
       " weights, max rel = ", emax/max(gmax,1.d-300)

  call grad_free( g );  call twork_free( tw );  call work_free( wk )
  call net_free( nt );  deallocate( tm, sm, tdum )

CONTAINS

  SUBROUTINE add( ir, ic, ia, ifac, c )
    implicit none
    integer,intent(IN) :: ir, ic, ia, ifac
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir;  sys_cmp(sys_nterm) = ic
    sys_ind(sys_nterm) = ia;  sys_fac(sys_nterm) = ifac
    sys_coeff(sys_nterm) = c
  END SUBROUTINE add

end program verify_ngd_multi
