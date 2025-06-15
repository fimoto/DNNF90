! Does the system residual, and its adjoint, do what they claim?
!
! The test problem is the Taylor-Green vortex, an exact unsteady solution
! of the two-dimensional incompressible Navier-Stokes equations:
!
!   u = -cos(x) sin(y) exp(-2 nu t)
!   v =  sin(x) cos(y) exp(-2 nu t)
!   p = -(cos 2x + cos 2y) exp(-4 nu t) / 4
!
! Three field components over three variables (x, y, t).  The momentum
! equations carry u du/dx + v du/dy: one component multiplying the
! derivative of another, which the scalar residual language cannot write.
!
! Two checks:
!   (a) the residual of the exact solution, evaluated through the system
!       machinery, is zero -- so the term table means what it should;
!   (b) the adjoint seed reproduces central differences of the loss,
!       which is where the cross terms are easy to get wrong, since each
!       contributes to two entries of the seed.
program taylorgreen_check
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, &
       alpha_list, alpha_deg, alpha_index
  use net_module, only: net_t, net_init, net_free, work_t, work_init, &
       work_free, net_eval_hod_multi
  use train_module, only: twork_t, grad_t, twork_init, grad_init, &
       grad_zero, twork_free, grad_free, net_forward_point, &
       net_backward_point_multi
  use global_variables, only: sys_nres, sys_nterm, sys_res, sys_cmp, &
       sys_ind, sys_fac, sys_coeff, sys_has_src, sys_wres, &
       sys_nfac, sys_fcomp, sys_find, sys_fac_ind
  use pinn_module, only: calc_sys_residual, set_sys_seed
  implicit none
  integer,parameter :: D0 = 3, KMAX = 2, NC = 3, W = 12
  real(8),parameter :: NU = 0.1d0
  type(net_t) :: nt
  type(work_t) :: wk
  type(twork_t) :: tw
  type(grad_t) :: g
  real(8),allocatable :: tm(:,:), seedm(:,:), tdum(:)
  real(8) :: x(D0), R(NC), src(NC), texact(NC,64)
  real(8) :: hw, lp, lm, fd, an, emax, gmax, wsave, e0
  integer :: sd(D0,1), dims(4), l, j, k, nchk, ix, iy, it, ixx, iyy

  sd = 0
  call init_hod_tables( D0, KMAX, 0, sd )
  dims = (/ D0, W, W, NC /)
  call net_init( nt, 4, dims )
  call work_init( wk, nt )
  call twork_init( tw, nt )
  call grad_init( g, nt )
  allocate( tm(NC,NUM_alpha), seedm(NC,NUM_alpha), tdum(NUM_alpha) )

  ! slots of the derivatives the equations need
  ix  = alpha_index( (/ 1,0,0 /) )
  iy  = alpha_index( (/ 0,1,0 /) )
  it  = alpha_index( (/ 0,0,1 /) )
  ixx = alpha_index( (/ 2,0,0 /) )
  iyy = alpha_index( (/ 0,2,0 /) )

  ! ---- the term table: three residuals over three components
  !   R1 = u_t + u u_x + v u_y + p_x - nu (u_xx + u_yy)
  !   R2 = v_t + u v_x + v v_y + p_y - nu (v_xx + v_yy)
  !   R3 = u_x + v_y
  sys_nres = 3;  sys_nterm = 0;  sys_has_src = .false.
  sys_wres = 1.d0            ! the loss weights of the residuals
  sys_nfac = 1;  sys_fcomp = 0;  sys_find = 1
  call add( 1, 1, it , 0,  1.d0 )      ! u_t
  call add( 1, 1, ix , 1,  1.d0 )      ! u * u_x
  call add( 1, 1, iy , 2,  1.d0 )      ! v * u_y
  call add( 1, 3, ix , 0,  1.d0 )      ! p_x
  call add( 1, 1, ixx, 0, -NU  )
  call add( 1, 1, iyy, 0, -NU  )
  call add( 2, 2, it , 0,  1.d0 )      ! v_t
  call add( 2, 2, ix , 1,  1.d0 )      ! u * v_x
  call add( 2, 2, iy , 2,  1.d0 )      ! v * v_y
  call add( 2, 3, iy , 0,  1.d0 )      ! p_y
  call add( 2, 2, ixx, 0, -NU  )
  call add( 2, 2, iyy, 0, -NU  )
  call add( 3, 1, ix , 0,  1.d0 )      ! u_x
  call add( 3, 2, iy , 0,  1.d0 )      ! v_y

  ! ---- (a) the exact solution must give zero residual
  x = (/ 0.7d0, 1.3d0, 0.25d0 /)
  call exact_slots( x, texact )
  src = 0.d0
  call calc_sys_residual( texact(1:NC,1:NUM_alpha), src, R )
  write(*,'(a,3e13.5)') " residual of the exact Taylor-Green solution:", R(1:NC)

  ! ---- (b) the adjoint against differences of the loss
  call random_seed()
  call random_number( nt%w )
  nt%w = 0.6d0*( nt%w - 0.5d0 )

  call grad_zero( g )
  call net_forward_point( nt, tw, x, tdum )
  call net_eval_hod_multi( nt, wk, x, tm )
  call calc_sys_residual( tm, src, R )
  call set_sys_seed( tm, R, 1.d0, seedm )
  call net_backward_point_multi( nt, tw, seedm, g )

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
  write(*,'(a,i0,a,e13.5)') " adjoint vs FD of the loss, ", nchk, &
       " weights, max rel = ", emax/max(gmax,1.d-300)

  call grad_free( g );  call twork_free( tw );  call work_free( wk )
  call net_free( nt );  deallocate( tm, seedm, tdum )

CONTAINS

  SUBROUTINE add( ir, ic, ia, ifac, c )
    implicit none
    integer,intent(IN) :: ir, ic, ia, ifac
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm)   = ir
    sys_cmp(sys_nterm)   = ic
    sys_ind(sys_nterm)   = ia
    sys_fac(sys_nterm)   = ifac
    sys_coeff(sys_nterm) = c
    sys_fac_ind(sys_nterm) = 1
    sys_fcomp(1,sys_nterm) = ic;  sys_find(1,sys_nterm) = ia
    if ( ifac > 0 ) then
       sys_nfac(sys_nterm) = 2
       sys_fcomp(2,sys_nterm) = ifac;  sys_find(2,sys_nterm) = 1
    else
       sys_nfac(sys_nterm) = 1
    end if
  END SUBROUTINE add

  !> Every carried derivative of the exact solution, filled by hand.
  SUBROUTINE exact_slots( xx, te )
    implicit none
    real(8),intent(IN) :: xx(D0)
    real(8),intent(OUT) :: te(NC,64)
    real(8) :: cx, sx, cy, sy, e2, e4, xa, ya, ta
    xa = xx(1);  ya = xx(2);  ta = xx(3)
    cx = cos(xa);  sx = sin(xa);  cy = cos(ya);  sy = sin(ya)
    e2 = exp(-2.d0*NU*ta);  e4 = exp(-4.d0*NU*ta)
    te = 0.d0
    ! u
    te(1,1)   = -cx*sy*e2
    te(1,ix)  =  sx*sy*e2
    te(1,iy)  = -cx*cy*e2
    te(1,it)  =  2.d0*NU*cx*sy*e2
    te(1,ixx) =  cx*sy*e2
    te(1,iyy) =  cx*sy*e2
    ! v
    te(2,1)   =  sx*cy*e2
    te(2,ix)  =  cx*cy*e2
    te(2,iy)  = -sx*sy*e2
    te(2,it)  = -2.d0*NU*sx*cy*e2
    te(2,ixx) = -sx*cy*e2
    te(2,iyy) = -sx*cy*e2
    ! p = -(cos2x + cos2y) e4 / 4
    te(3,1)   = -( cos(2.d0*xa) + cos(2.d0*ya) )*e4/4.d0
    te(3,ix)  =  sin(2.d0*xa)*e4/2.d0
    te(3,iy)  =  sin(2.d0*ya)*e4/2.d0
  END SUBROUTINE exact_slots

end program taylorgreen_check
