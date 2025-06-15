! Does the multi-output adjoint give the right weight gradient?
!
! The loss couples the components, which is the point of having them in
! one network:
!
!    L = 1/2 sum_i sum_alpha ( T^alpha_i )^2
!
! so the seed is dL/dT^alpha_i = T^alpha_i for every component.  The
! gradient it returns is compared with central differences of L in the
! weights.  A single-output network is the special case, and is checked
! too: seeding component one alone must reproduce net_backward_point.
program verify_multi_adj
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha
  use net_module, only: net_t, net_init, net_free, work_t, work_init, &
       work_free, net_eval_hod, net_eval_hod_multi
  use train_module, only: twork_t, grad_t, twork_init, grad_init, &
       grad_zero, twork_free, grad_free, net_forward_point, &
       net_backward_point, net_backward_point_multi
  implicit none
  integer,parameter :: D0 = 3, KMAX = 2, NOUT = 3, W = 10
  type(net_t) :: nt
  type(work_t) :: wk
  type(twork_t) :: tw
  type(grad_t) :: g
  real(8),allocatable :: tm(:,:), seedm(:,:), tdum(:)
  real(8) :: x(D0), hw, lp, lm, fd, an, emax, gmax, wsave
  integer :: sd(D0,1), dims(4), l, j, k, nchk

  sd = 0
  call init_hod_tables( D0, KMAX, 0, sd )
  dims = (/ D0, W, W, NOUT /)
  call net_init( nt, 4, dims )
  call work_init( wk, nt )
  call twork_init( tw, nt )
  call grad_init( g, nt )
  allocate( tm(NOUT,NUM_alpha), seedm(NOUT,NUM_alpha), tdum(NUM_alpha) )

  call random_seed()
  call random_number( nt%w )
  nt%w = 0.6d0*( nt%w - 0.5d0 )
  x = (/ 0.31d0, -0.22d0, 0.47d0 /)

  ! ---- analytic gradient of L over every component
  call grad_zero( g )
  call net_forward_point( nt, tw, x, tdum )
  call net_eval_hod_multi( nt, wk, x, tm )
  seedm(1:NOUT,1:NUM_alpha) = tm(1:NOUT,1:NUM_alpha)
  call net_backward_point_multi( nt, tw, seedm, g )

  ! ---- central differences of the same loss
  hw = 1.d-5;  emax = 0.d0;  gmax = 0.d0;  nchk = 0
  do l = 2, nt%nlayer
     do j = 1, nt%ndim(l)
        do k = 0, nt%ndim(l-1)
           wsave = nt%w(l,j,k)
           nt%w(l,j,k) = wsave + hw
           call net_eval_hod_multi( nt, wk, x, tm )
           lp = 0.5d0*sum( tm(1:NOUT,1:NUM_alpha)**2 )
           nt%w(l,j,k) = wsave - hw
           call net_eval_hod_multi( nt, wk, x, tm )
           lm = 0.5d0*sum( tm(1:NOUT,1:NUM_alpha)**2 )
           nt%w(l,j,k) = wsave
           fd = ( lp - lm )/( 2.d0*hw )
           an = g%nabla(l,j,k)
           emax = max( emax, abs(an-fd) )
           gmax = max( gmax, abs(fd) )
           nchk = nchk + 1
        end do
     end do
  end do

  write(*,'(a,i0,a,i0,a)') " network ", D0, " -> ... -> ", NOUT, &
       " outputs, loss over every component"
  write(*,'(a,i0,a,e12.4)') " weight gradient vs FD, ", nchk, &
       " weights, max rel = ", emax/max(gmax,1.d-300)

  call grad_free( g );  call twork_free( tw );  call work_free( wk )
  call net_free( nt )
  deallocate( tm, seedm, tdum )
end program verify_multi_adj
