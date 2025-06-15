! Does the collocation loss of the electrohydrodynamic system have the
! gradient the trainer computes?
!
! Everything else about this case has checked out: the term table
! reproduces the equations, the network can represent the solution, the
! observation row of the filter is right.  What has not been checked is
! the gradient of the collocation loss in the configuration the case
! actually uses -- five residuals, three of them carrying a source, and a
! product of two derivatives among the terms.
!
! The loss is
!
!     L = (1/2) sum_r w_r ( R_r - S_r )^2
!
! summed over a few points, and its gradient is compared with central
! differences of the same quantity.  A source enters R additively and
! does not depend on the weights, so it must not appear in the seed; if
! it does, this check catches it.
program verify_ehd_grad
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, alpha_index
  use net_module, only: net_t, net_init, net_free, work_t, work_init, &
       work_free, net_eval_hod_multi
  use train_module, only: twork_t, grad_t, twork_init, grad_init, &
       twork_free, grad_free, net_forward_point, net_backward_point_multi
  use global_variables, only: sys_nres, sys_nterm, sys_res, sys_coeff, &
       sys_nfac, sys_fcomp, sys_find, sys_has_src, sys_wres
  use pinn_module, only: calc_sys_residual, set_sys_seed
  implicit none
  integer,parameter :: D0 = 2, KMAX = 2, NC = 5, W = 6, NPT = 4
  real(8),parameter :: EPSC = 1.d0, MU = 0.5d0, DIF = 0.1d0, &
       NU = 0.1d0, CF = 0.2d0
  type(net_t) :: nt
  type(work_t) :: wk
  type(twork_t) :: tw
  type(grad_t) :: g
  real(8),allocatable :: tm(:,:), sm(:,:), tdum(:)
  real(8) :: xs(D0,NPT), src(NC,NPT), R(NC)
  real(8) :: hw, lp, lm, fd, an, emax, gmax, wsave
  integer :: sd(D0,1), dims(4), l, j, k, n, nchk
  integer :: ix, iy, ixx, iyy

  sd = 0
  call init_hod_tables( D0, KMAX, 0, sd )
  dims = (/ D0, W, W, NC /)
  call net_init( nt, 4, dims )
  call work_init( wk, nt );  call twork_init( tw, nt );  call grad_init( g, nt )
  allocate( tm(NC,NUM_alpha), sm(NC,NUM_alpha), tdum(NUM_alpha) )

  ix  = alpha_index( (/ 1,0 /) );  iy  = alpha_index( (/ 0,1 /) )
  ixx = alpha_index( (/ 2,0 /) );  iyy = alpha_index( (/ 0,2 /) )

  ! the term table of zwork/pinn_ehd
  sys_nres = 5;  sys_nterm = 0;  sys_has_src = .false.
  sys_wres = 1.d0;  sys_nfac = 1;  sys_fcomp = 0;  sys_find = 1
  call t1( 1, 1, ixx, 1.d0 );  call t1( 1, 1, iyy, 1.d0 )
  call t1( 1, 2, 1  , 1.d0/EPSC )
  call t2( 2, 2, ix , 3, 1  , 1.d0 )
  call t2( 2, 2, iy , 4, 1  , 1.d0 )
  call t2( 2, 2, ix , 1, ix , -MU )        ! rho_x phi_x, the DXD form
  call t2( 2, 2, iy , 1, iy , -MU )
  call t2( 2, 2, 1  , 1, ixx, -MU )
  call t2( 2, 2, 1  , 1, iyy, -MU )
  call t1( 2, 2, ixx, -DIF );  call t1( 2, 2, iyy, -DIF )
  call t2( 3, 3, ix , 3, 1  , 1.d0 )
  call t2( 3, 3, iy , 4, 1  , 1.d0 )
  call t1( 3, 5, ix , 1.d0 )
  call t1( 3, 3, ixx, -NU );  call t1( 3, 3, iyy, -NU )
  call t2( 3, 1, ix , 2, 1  , CF )
  call t2( 4, 4, ix , 3, 1  , 1.d0 )
  call t2( 4, 4, iy , 4, 1  , 1.d0 )
  call t1( 4, 5, iy , 1.d0 )
  call t1( 4, 4, ixx, -NU );  call t1( 4, 4, iyy, -NU )
  call t2( 4, 1, iy , 2, 1  , CF )
  call t1( 5, 3, ix , 1.d0 );  call t1( 5, 4, iy , 1.d0 )

  call random_seed()
  call random_number( nt%w );  nt%w = 0.6d0*( nt%w - 0.5d0 )
  xs(:,1) = (/  0.7d0,  1.3d0 /)
  xs(:,2) = (/ -1.4d0,  2.1d0 /)
  xs(:,3) = (/  1.9d0, -0.8d0 /)
  xs(:,4) = (/  0.2d0,  0.5d0 /)
  do n = 1, NPT
     call sources( xs(1,n), xs(2,n), src(:,n) )
  end do

  ! the gradient, accumulated over the points as the trainer does
  g%nabla = 0.d0
  do n = 1, NPT
     call net_forward_point( nt, tw, xs(:,n), tdum )
     call net_eval_hod_multi( nt, wk, xs(:,n), tm )
     call calc_sys_residual( tm, src(:,n), R )
     call set_sys_seed( tm, R, 1.d0, sm )
     call net_backward_point_multi( nt, tw, sm, g )
  end do

  hw = 1.d-6;  emax = 0.d0;  gmax = 0.d0;  nchk = 0
  do l = 2, nt%nlayer
     do j = 1, nt%ndim(l)
        do k = 0, nt%ndim(l-1)
           wsave = nt%w(l,j,k)
           nt%w(l,j,k) = wsave + hw;  lp = loss()
           nt%w(l,j,k) = wsave - hw;  lm = loss()
           nt%w(l,j,k) = wsave
           fd = ( lp - lm )/( 2.d0*hw )
           an = g%nabla(l,j,k)
           emax = max( emax, abs(an-fd) );  gmax = max( gmax, abs(fd) )
           nchk = nchk + 1
        end do
     end do
  end do

  write(*,'(a,i0,a,e13.5)') &
       " collocation gradient vs FD, five residuals with sources, ", &
       nchk, " weights, max rel = ", emax/max(gmax,1.d-300)
  if ( emax/max(gmax,1.d-300) > 1.d-6 ) then
     write(*,'(a)') " FAILED"
     stop 1
  end if
  write(*,'(a)') " passed"

  call grad_free( g );  call twork_free( tw );  call work_free( wk )
  call net_free( nt );  deallocate( tm, sm, tdum )

CONTAINS

  REAL(8) FUNCTION loss()
    implicit none
    integer :: nn
    loss = 0.d0
    do nn = 1, NPT
       call net_eval_hod_multi( nt, wk, xs(:,nn), tm )
       call calc_sys_residual( tm, src(:,nn), R )
       loss = loss + 0.5d0*sum( sys_wres(1:NC)*R(1:NC)**2 )
    end do
  END FUNCTION loss

  SUBROUTINE t1( ir, c1, i1, c )
    implicit none
    integer,intent(IN) :: ir, c1, i1
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir;  sys_coeff(sys_nterm) = c
    sys_nfac(sys_nterm) = 1
    sys_fcomp(1,sys_nterm) = c1;  sys_find(1,sys_nterm) = i1
  END SUBROUTINE t1

  SUBROUTINE t2( ir, c1, i1, c2, i2, c )
    implicit none
    integer,intent(IN) :: ir, c1, i1, c2, i2
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir;  sys_coeff(sys_nterm) = c
    sys_nfac(sys_nterm) = 2
    sys_fcomp(1,sys_nterm) = c1;  sys_find(1,sys_nterm) = i1
    sys_fcomp(2,sys_nterm) = c2;  sys_find(2,sys_nterm) = i2
  END SUBROUTINE t2

  SUBROUTINE sources( xa, ya, so )
    implicit none
    real(8),intent(IN) :: xa, ya
    real(8),intent(OUT) :: so(NC)
    real(8) :: s, c, S2, C2, rho, rx, ry, rxx, ryy
    real(8) :: px, py, pxx, pyy, uu, vv, ux, uy, vx, vy
    real(8) :: uxx, uyy, vxx, vyy, ppx, ppy, drgp
    s = sin(xa);  c = cos(xa);  S2 = sin(ya);  C2 = cos(ya)
    px = c*S2;  py = s*C2;  pxx = -s*S2;  pyy = -s*S2
    rho = c*C2 + 0.5d0
    rx = -s*C2;  ry = -c*S2
    rxx = -c*C2;  ryy = -c*C2
    uu = -c*S2;  vv = s*C2
    ux = s*S2;  uy = -c*C2;  vx = c*C2;  vy = -s*S2
    uxx = c*S2;  uyy = c*S2;  vxx = -s*C2;  vyy = -s*C2
    ppx = sin(2.d0*xa)/2.d0;  ppy = sin(2.d0*ya)/2.d0
    drgp = rx*px + rho*pxx + ry*py + rho*pyy
    so(1) = ( pxx + pyy ) + rho/EPSC
    so(2) = uu*rx + vv*ry - MU*drgp - DIF*( rxx + ryy )
    so(3) = uu*ux + vv*uy + ppx - NU*( uxx + uyy ) + CF*rho*px
    so(4) = uu*vx + vv*vy + ppy - NU*( vxx + vyy ) + CF*rho*py
    so(5) = 0.d0
  END SUBROUTINE sources

end program verify_ehd_grad
