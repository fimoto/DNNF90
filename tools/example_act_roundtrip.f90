! Does a weight file survive save -> free -> load for every activation?
!
! The file records the activation it was trained with, and net_load
! refuses a table set that carries a different one, so that a J_0 or
! sine network cannot be read back and evaluated as tanh.  This compares
! the carried derivatives before and after the round trip; they must
! agree bit for bit, since nothing is recomputed.
program rt
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, tabset_t, &
       tabset_from_current
  use net_module, only: net_t, net_init, net_save, net_load, net_free, &
       work_t, work_init, net_eval_hod
  implicit none
  type(net_t) :: nt, nt2
  type(tabset_t) :: ts
  type(work_t) :: wk, wk2
  real(8) :: x(2), t1(64), t2(64), emax
  integer :: sd(2,1), dims(4), ia, q, nfail
  character(len=8) :: nm(0:4)
  nm = (/ "TANH    ","SIN     ","ERF     ","BESSEL  ","BESSEL1 " /)
  sd = 0;  dims = (/ 2, 5, 5, 1 /);  nfail = 0
  call init_hod_tables( 2, 2, 0, sd )
  x = (/ 0.31d0, -0.22d0 /)
  do ia = 0, 4
     call tabset_from_current( ts );  ts%iact = ia
     call net_init( nt, 4, dims, ts )
     call random_seed();  call random_number( nt%w );  nt%w = 0.5d0*(nt%w-0.5d0)
     call work_init( wk, nt )
     call net_eval_hod( nt, wk, x, t1(1:NUM_alpha) )
     call net_save( nt, '/tmp/rt_w.dat' )
     call net_free( nt )
     call net_load( nt2, '/tmp/rt_w.dat', ts )
     call work_init( wk2, nt2 )
     call net_eval_hod( nt2, wk2, x, t2(1:NUM_alpha) )
     emax = 0.d0
     do q = 1, NUM_alpha
        emax = max( emax, abs(t1(q)-t2(q)) )
     end do
     write(*,'(a,a,a,es12.4)') ' ', nm(ia), ' save/load max |diff| = ', emax
     if ( emax > 0.d0 ) nfail = nfail + 1
     call net_free( nt2 )
  end do
  if ( nfail == 0 ) then
     write(*,'(a)') ' passed'
  else
     write(*,'(a,i0,a)') ' FAILED on ', nfail, ' activation(s)';  stop 1
  end if
end program rt
