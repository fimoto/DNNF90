! Does the multi-component filter update move the weights the right way?
!
! One rank-one update on a network small enough to check by hand.  The
! filter is meant to reduce the innovation, so after one update on a
! single observable the residual of that observable must be smaller in
! magnitude than before.  If it grows, the sign of the innovation is
! wrong somewhere between the seed and the weight update.
!
! The scalar path is checked the same way in the same run, so the two can
! be compared: whatever the multi-component path does, the scalar one is
! the behaviour it should reproduce when the system has one component.
program verify_kf_multi
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, alpha_index
  use net_module, only: net_t, net_init, net_free, work_t, work_init, &
       work_free, net_eval_hod_multi
  use train_module, only: twork_t, grad_t, twork_init, grad_init, &
       twork_free, grad_free, net_forward_point
  use kalman_module, only: kalman_t, kf_init, kf_free, kf_update_resid, &
       kf_update_resid_multi
  use global_variables, only: sys_nres, sys_nterm, sys_res, sys_cmp, &
       sys_ind, sys_fac, sys_coeff, sys_has_src
  use pinn_module, only: calc_sys_residual, set_sys_seed
  implicit none
  integer,parameter :: D0 = 2, KMAX = 2, NC = 2, W = 4
  type(net_t) :: nt
  type(work_t) :: wk
  type(twork_t) :: tw
  type(grad_t) :: g
  type(kalman_t) :: kf
  real(8),allocatable :: tm(:,:), sm(:,:), rsel(:), tdum(:)
  real(8) :: x(D0), R0(NC), R1(NC), src(NC)
  integer :: sd(D0,1), dims(4), ix, iy, ir, k

  sd = 0
  call init_hod_tables( D0, KMAX, 0, sd )
  dims = (/ D0, W, W, NC /)
  call net_init( nt, 4, dims )
  call work_init( wk, nt )
  call twork_init( tw, nt )
  call grad_init( g, nt )
  call kf_init( kf, nt, 1.d-2, 0.999d0, 1.d0 )
  allocate( tm(NC,NUM_alpha), sm(NC,NUM_alpha), rsel(NC), tdum(NUM_alpha) )

  ix = alpha_index( (/ 1,0 /) );  iy = alpha_index( (/ 0,1 /) )

  ! a small coupled system:  R1 = u_x + v,   R2 = v_y - u
  sys_nres = 2;  sys_nterm = 0;  sys_has_src = .false.
  call add( 1, 1, ix, 0,  1.d0 )
  call add( 1, 2, 1 , 0,  1.d0 )
  call add( 2, 2, iy, 0,  1.d0 )
  call add( 2, 1, 1 , 0, -1.d0 )

  call random_seed()
  call random_number( nt%w );  nt%w = 0.6d0*( nt%w - 0.5d0 )
  x = (/ 0.3d0, -0.4d0 /)
  src = 0.d0

  call net_eval_hod_multi( nt, wk, x, tm )
  call calc_sys_residual( tm, src, R0 )
  write(*,'(a,2e13.5)') " residuals before the update:", R0(1:NC)

  ! one update per residual, as the trainer does
  do ir = 1, sys_nres
     rsel = 0.d0;  rsel(ir) = 1.d0
     call set_sys_seed( tm, rsel, 1.d0, sm )
     call net_forward_point( nt, tw, x, tdum )
     call kf_update_resid_multi( nt, tw, g, kf, x, sm, R0(ir) )
  end do

  call net_eval_hod_multi( nt, wk, x, tm )
  call calc_sys_residual( tm, src, R1 )
  write(*,'(a,2e13.5)') " residuals after  the update:", R1(1:NC)
  write(*,'(a)') " a filter update should shrink them; growth means the"
  write(*,'(a)') " innovation enters with the wrong sign"
  do ir = 1, NC
     if ( abs(R1(ir)) < abs(R0(ir)) ) then
        write(*,'(a,i0,a)') "   residual ", ir, ": shrank"
     else
        write(*,'(a,i0,a)') "   residual ", ir, ": GREW"
     end if
  end do

  call kf_free( kf );  call grad_free( g );  call twork_free( tw )
  call work_free( wk );  call net_free( nt )
  deallocate( tm, sm, rsel, tdum )

CONTAINS

  SUBROUTINE add( ir_, ic_, ia_, ifac_, c_ )
    implicit none
    integer,intent(IN) :: ir_, ic_, ia_, ifac_
    real(8),intent(IN) :: c_
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir_;  sys_cmp(sys_nterm) = ic_
    sys_ind(sys_nterm) = ia_;  sys_fac(sys_nterm) = ifac_
    sys_coeff(sys_nterm) = c_
  END SUBROUTINE add

end program verify_kf_multi
