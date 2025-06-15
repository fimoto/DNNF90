! Is the filter's observation row right for a system?
!
! The extended Kalman filter presents one residual at a time and needs
! its row dR_ir/dw, which the trainer builds by calling set_sys_seed
! with a residual vector holding a one in slot ir.  That is a different
! call from the one the gradient path makes, so it needs its own check:
! the row is compared against central differences of R_ir itself, not of
! the loss.
!
! The residual below carries a product of two derivatives, DXD, which is
! the form the electrohydrodynamic case adds and the Taylor-Green case
! does not have.
program verify_filter_row
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, alpha_index
  use net_module, only: net_t, net_init, net_free, work_t, work_init, &
       work_free, net_eval_hod_multi
  use train_module, only: twork_t, grad_t, twork_init, grad_init, &
       twork_free, grad_free, net_forward_point, net_backward_point_multi
  use global_variables, only: sys_nres, sys_nterm, sys_res, sys_coeff, &
       sys_nfac, sys_fcomp, sys_find, sys_has_src, sys_wres
  use pinn_module, only: calc_sys_residual, set_sys_seed
  implicit none
  integer,parameter :: D0 = 2, KMAX = 2, NC = 3, W = 5
  type(net_t) :: nt
  type(work_t) :: wk
  type(twork_t) :: tw
  type(grad_t) :: g
  real(8),allocatable :: tm(:,:), sm(:,:), rsel(:), tdum(:)
  real(8) :: x(D0), R(NC), src(NC)
  real(8) :: hw, rp, rm, fd, an, emax, gmax, wsave
  integer :: sd(D0,1), dims(4), l, j, k, ir, nchk
  integer :: ix, iy, ixx

  sd = 0
  call init_hod_tables( D0, KMAX, 0, sd )
  dims = (/ D0, W, W, NC /)
  call net_init( nt, 4, dims )
  call work_init( wk, nt );  call twork_init( tw, nt );  call grad_init( g, nt )
  allocate( tm(NC,NUM_alpha), sm(NC,NUM_alpha), rsel(NC), tdum(NUM_alpha) )

  ix = alpha_index( (/ 1,0 /) )
  iy = alpha_index( (/ 0,1 /) )
  ixx = alpha_index( (/ 2,0 /) )

  sys_nres = 3;  sys_nterm = 0;  sys_has_src = .false.
  sys_wres = 1.d0
  sys_nfac = 1;  sys_fcomp = 0;  sys_find = 1

  ! R1: linear plus a product of two derivatives of different components
  call t1( 1, 1, ixx, 1.d0 )
  call t2( 1, 2, ix , 3, iy , 0.8d0 )        ! u2_x * u3_y   <- DXD form
  ! R2: a value times a derivative, the XUX form
  call t2( 2, 3, 1  , 1, ix , 1.1d0 )
  call t1( 2, 2, iy , -0.4d0 )
  ! R3: two derivatives of the same component
  call t2( 3, 1, ix , 1, iy , 0.6d0 )
  call t1( 3, 3, 1  , 0.3d0 )

  call random_seed()
  call random_number( nt%w );  nt%w = 0.7d0*( nt%w - 0.5d0 )
  x = (/ 0.29d0, -0.33d0 /);  src = 0.d0

  emax = 0.d0;  gmax = 0.d0;  nchk = 0
  do ir = 1, sys_nres
     ! the row exactly as the trainer builds it for the filter
     call net_forward_point( nt, tw, x, tdum )
     call net_eval_hod_multi( nt, wk, x, tm )
     rsel = 0.d0;  rsel(ir) = 1.d0
     call set_sys_seed( tm, rsel, 1.d0, sm, .false. )
     g%nabla = 0.d0
     call net_backward_point_multi( nt, tw, sm, g )

     hw = 1.d-6
     do l = 2, nt%nlayer
        do j = 1, nt%ndim(l)
           do k = 0, nt%ndim(l-1)
              wsave = nt%w(l,j,k)
              nt%w(l,j,k) = wsave + hw
              call net_eval_hod_multi( nt, wk, x, tm )
              call calc_sys_residual( tm, src, R );  rp = R(ir)
              nt%w(l,j,k) = wsave - hw
              call net_eval_hod_multi( nt, wk, x, tm )
              call calc_sys_residual( tm, src, R );  rm = R(ir)
              nt%w(l,j,k) = wsave
              fd = ( rp - rm )/( 2.d0*hw )
              an = g%nabla(l,j,k)
              emax = max( emax, abs(an-fd) );  gmax = max( gmax, abs(fd) )
              nchk = nchk + 1
           end do
        end do
     end do
  end do

  write(*,'(a,i0,a,e13.5)') " filter row vs FD of the residual, ", nchk, &
       " checks, max rel = ", emax/max(gmax,1.d-300)
  if ( emax/max(gmax,1.d-300) > 1.d-5 ) then
     write(*,'(a)') " FAILED"
     stop 1
  end if
  write(*,'(a)') " passed"

  call grad_free( g );  call twork_free( tw );  call work_free( wk )
  call net_free( nt );  deallocate( tm, sm, rsel, tdum )

CONTAINS

  SUBROUTINE t1( ir_, c1, i1, c )
    implicit none
    integer,intent(IN) :: ir_, c1, i1
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir_;  sys_coeff(sys_nterm) = c
    sys_nfac(sys_nterm) = 1
    sys_fcomp(1,sys_nterm) = c1;  sys_find(1,sys_nterm) = i1
  END SUBROUTINE t1

  SUBROUTINE t2( ir_, c1, i1, c2, i2, c )
    implicit none
    integer,intent(IN) :: ir_, c1, i1, c2, i2
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir_;  sys_coeff(sys_nterm) = c
    sys_nfac(sys_nterm) = 2
    sys_fcomp(1,sys_nterm) = c1;  sys_find(1,sys_nterm) = i1
    sys_fcomp(2,sys_nterm) = c2;  sys_find(2,sys_nterm) = i2
  END SUBROUTINE t2

end program verify_filter_row
