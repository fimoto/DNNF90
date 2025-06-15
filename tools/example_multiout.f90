! Does the multi-output extraction give the right derivatives?
!
! Two independent checks, because the extension must not be trusted on
! the grounds that it compiles.
!
!   (a) every carried derivative of every output component, against
!       central differences of that component;
!   (b) component 1 against net_eval_hod, which the whole distribution
!       is already verified on.
program verify_multi
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, &
       alpha_list, alpha_deg
  use net_module, only: net_t, net_init, net_free, work_t, work_init, &
       work_free, net_eval_hod, net_eval_hod_multi
  implicit none
  integer,parameter :: D0 = 3, KMAX = 2, NOUT = 4
  type(net_t) :: nt
  type(work_t) :: wk
  real(8),allocatable :: tm(:,:), t1(:), tp(:,:), tmm(:,:)
  real(8) :: x(D0), xp(D0), h, fd, an, emax, gmax
  integer :: seeds(D0,1), dims(4), i, j, ia, io, nchk

  seeds = 0
  call init_hod_tables( D0, KMAX, 0, seeds )
  dims = (/ D0, 12, 12, NOUT /)
  call net_init( nt, 4, dims )
  call work_init( wk, nt )
  allocate( tm(NOUT,NUM_alpha), t1(NUM_alpha) )
  allocate( tp(NOUT,NUM_alpha), tmm(NOUT,NUM_alpha) )

  ! net_init zeroes the weights by design; give them values, or every
  ! derivative is trivially zero and the check proves nothing
  call random_seed()
  call random_number( nt%w )
  nt%w = 0.6d0*( nt%w - 0.5d0 )

  x = (/ 0.31d0, -0.22d0, 0.47d0 /)
  call net_eval_hod_multi( nt, wk, x, tm )
  call net_eval_hod( nt, wk, x, t1 )
  write(*,'(a,4f13.7)') " value of each output      :", tm(1:NOUT,1)
  write(*,'(a,4f13.7)') " d/dx1 of each output      :", tm(1:NOUT,2)

  ! (b) component one must reproduce the single-output routine exactly
  write(*,'(a,e12.4)') " component 1 vs net_eval_hod, max |diff| = ", &
       maxval( abs( tm(1,1:NUM_alpha) - t1(1:NUM_alpha) ) )

  ! (a) first derivatives of every component against central differences
  h = 1.d-4;  emax = 0.d0;  gmax = 0.d0;  nchk = 0
  do ia = 1, NUM_alpha
     if ( alpha_deg(ia) /= 1 ) cycle
     j = 0
     do i = 1, D0
        if ( alpha_list(i,ia) == 1 ) j = i
     end do
     xp = x;  xp(j) = x(j) + h
     call net_eval_hod_multi( nt, wk, xp, tp )
     xp = x;  xp(j) = x(j) - h
     call net_eval_hod_multi( nt, wk, xp, tmm )
     do io = 1, NOUT
        fd = ( tp(io,1) - tmm(io,1) )/( 2.d0*h )
        an = tm(io,ia)
        emax = max( emax, abs(an-fd) )
        gmax = max( gmax, abs(fd) )
        nchk = nchk + 1
     end do
  end do
  write(*,'(a,i0,a,e12.4)') " first derivatives of all ", NOUT, &
       " outputs vs FD, max rel = ", emax/max(gmax,1.d-300)

  ! second derivatives too
  emax = 0.d0;  gmax = 0.d0
  do ia = 1, NUM_alpha
     if ( alpha_deg(ia) /= 2 ) cycle
     j = 0
     do i = 1, D0
        if ( alpha_list(i,ia) == 2 ) j = i
     end do
     if ( j == 0 ) cycle          ! mixed ones need a two-axis difference
     xp = x;  xp(j) = x(j) + h
     call net_eval_hod_multi( nt, wk, xp, tp )
     xp = x;  xp(j) = x(j) - h
     call net_eval_hod_multi( nt, wk, xp, tmm )
     do io = 1, NOUT
        fd = ( tp(io,1) - 2.d0*tm(io,1) + tmm(io,1) )/( h*h )
        an = tm(io,ia)
        emax = max( emax, abs(an-fd) )
        gmax = max( gmax, abs(fd) )
     end do
  end do
  write(*,'(a,e12.4)') " pure second derivatives vs FD, max rel = ", &
       emax/max(gmax,1.d-300)

  call work_free( wk );  call net_free( nt )
  deallocate( tm, t1, tp, tmm )
end program verify_multi
