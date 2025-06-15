!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (example_uq.f90) is part of DNNF90.
!
!  DNNF90 is free software released under the MIT License.
!  You should have received a copy of the MIT License (file LICENSE
!  in the root directory of this distribution) along with DNNF90.
!  If not, see <https://opensource.org/licenses/MIT>.
!
!  DNNF90 is distributed in the hope that it will be useful,
!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!  MIT License for more details.
!
! -----------------------------------------------------------------------
! Three demonstrations for on-the-fly workflows, self contained:
!
!   test 1  committee uncertainty: five members trained from different
!           initial weights on x in [-2,2]; the committee spread must
!           track the true error, in particular it must grow strongly
!           outside the sampled region (the OTF trigger signal)
!   test 2  Kalman filter training (the n2p2 method) against Adam at an
!           equal number of pattern presentations
!   test 3  two table sets with different D0 and K coexisting in one
!           process, each bound to its own network (per species tables)
!
! Each test carries a criterion and the program stops with a nonzero
! status if one is not met.
!
! Run: ./uq_example.out
PROGRAM example_uq
  use multi_index_bell_module, only: tabset_t, tabset_init, tabset_free
  use net_module
  use train_module
  use committee_module, only: committee_t, comm_init, comm_free, comm_eval
  use kalman_module
  implicit none

  integer,parameter :: NMEM = 5, NPT = 120, NL = 3
  integer :: dims(NL) = (/ 1, 12, 1 /)
  type(tabset_t) :: ts1, ts2
  type(committee_t) :: cm
  type(work_t)  :: wk
  type(twork_t) :: tw
  type(grad_t)  :: g
  type(net_t)   :: neta, netk, netk2, net2
  type(kalman_t) :: kf
  real(8) :: x(1), t(16), tm(16), tsd(16)
  real(8) :: xd(NPT), yd(NPT)
  real(8) :: corr_in, grow, la, lk, sd2(16)
  integer :: k, n, ep, dummy(1,1), pres
  integer :: nfail = 0

  ! ---------------- table set 1: one input, first derivatives ----------------
  call tabset_init( ts1, 1, 1, 0, dummy )

  do n=1,NPT
     xd(n) = -2.d0 + 4.d0*dble(n-1)/dble(NPT-1)
     yd(n) = target_f( xd(n) )
  end do

  ! ---------------- test 1: committee ----------------
  call comm_init( cm, NMEM, NL, dims, ts1 )
  do k=1,NMEM
     call seed_weights( cm%mem(k), k )
  end do
  call work_init( wk, cm%mem(1) )
  call twork_init( tw, cm%mem(1) )
  call grad_init( g, cm%mem(1) )
  do k=1,NMEM
     do ep=1,4000
        call grad_zero( g )
        do n=1,NPT
           call one_seed( cm%mem(k), xd(n), yd(n) )
        end do
        call opt_adam_step( cm%mem(k), g, 5.d-3, 0.9d0, 0.999d0, 1.d-8, NPT, ep )
     end do
  end do
  call spread_vs_error( corr_in, grow )
  write(*,'(a,f6.3)') "test 1  committee: corr(log spread, log |error|)      = ", corr_in
  write(*,'(a,f8.1,a)') "        spread outside the sampled region grows by x", grow, &
       merge("   (extrapolation detected)","   (NO DETECTION)          ", grow > 10.d0 )
  if ( .not. ( grow > 10.d0 ) ) then
     write(*,'(a)') "        FAILED: the spread does not mark the extrapolation"
     nfail = nfail + 1
  end if

  ! ---------------- test 2: Kalman vs Adam ----------------
  call net_init( neta, NL, dims, ts1 );  call seed_weights( neta, 1 )
  call net_init( netk, NL, dims, ts1 );  call seed_weights( netk, 1 )
  pres = 10*NPT                           ! equal presentation budget
  do ep=1,10                              ! Adam: 10 full-batch epochs
     call grad_zero( g )
     do n=1,NPT
        call one_seed( neta, xd(n), yd(n) )
     end do
     call opt_adam_step( neta, g, 5.d-3, 0.9d0, 0.999d0, 1.d-8, NPT, ep )
  end do
  call kf_init( kf, netk, 100.d0, 0.98d0, 0.9987d0 )
  do ep=1,10                              ! Kalman: 10 sweeps, one update each
     do n=1,NPT
        call kf_update( netk, tw, g, kf, (/ xd(n) /), 1, yd(n) )
     end do
  end do
  la = loss_of( neta );  lk = loss_of( netk )
  write(*,'(a,i0,a,2e12.3)') "test 2  after ", pres, &
       " presentations, loss  Adam | Kalman :", la, lk
  write(*,'(a)') merge("        Kalman converges far faster per presentation    ", &
                       "        UNEXPECTED                                      ", lk < 0.1d0*la )
  if ( .not. ( lk < 0.1d0*la ) ) then
     write(*,'(a)') "        FAILED: the filter did not beat Adam at equal budget"
     nfail = nfail + 1
  end if

  ! -------- test 2b: Kalman on a DERIVATIVE observable --------
  ! the observed quantity is du/dx (slot 2), the way a force enters a
  ! force-field Kalman update; targets are the analytic derivative
  call net_init( netk2, NL, dims, ts1 );  call seed_weights( netk2, 1 )
  call kf_free( kf )
  call kf_init( kf, netk2, 100.d0, 0.98d0, 0.9987d0 )
  do ep=1,10
     do n=1,NPT
        sd2 = 0.d0
        sd2(2) = 1.d0
        call kf_update_obs( netk2, tw, g, kf, (/ xd(n) /), sd2, dtarget_f( xd(n) ) )
     end do
  end do
  lk = dloss_of( netk2 )
  write(*,'(a,e12.3)') "test 2b Kalman observing du/dx (slot 2), derivative loss :", lk
  write(*,'(a)') merge("        any carried derivative slot is observable       ", &
                       "        UNEXPECTED                                      ", lk < 1.d-3 )
  if ( .not. ( lk < 1.d-3 ) ) then
     write(*,'(a)') "        FAILED: observing the derivative slot did not fit it"
     nfail = nfail + 1
  end if

  ! ---------------- test 3: coexisting table sets ----------------
  call tabset_init( ts2, 3, 2, 0, dummy )     ! a different species: D0=3, K=2
  call net_init( net2, 3, (/ 3, 8, 1 /), ts2 )
  write(*,'(a,i0,a,i0)') "test 3  slots per point, species A (D0=1,K=1): ", &
       neta%tab%na, "   species B (D0=3,K=2): ", net2%tab%na
  x(1) = 0.3d0
  call net_eval_hod( neta, wk, x, t )         ! net A still on its own tables
  write(*,'(a,2f10.5)') "        species A evaluates after B was built: u,du = ", t(1), t(2)
  write(*,'(a)') merge("        independent table sets coexist                 ", &
                       "        BROKEN                                         ", &
                       neta%tab%na == 2 .and. net2%tab%na == 10 )
  if ( .not. ( neta%tab%na == 2 .and. net2%tab%na == 10 ) ) then
     write(*,'(a)') "        FAILED: the two table sets interfered"
     nfail = nfail + 1
  end if

  ! Release everything the demonstration built.  This file is also a
  ! model of how the instances are meant to be used, so it shows the
  ! teardown and not only the setup.
  call kf_free( kf )
  call grad_free( g );  call twork_free( tw );  call work_free( wk )
  call comm_free( cm )
  call net_free( neta );  call net_free( netk )
  call net_free( netk2 ); call net_free( net2 )
  call tabset_free( ts1 ); call tabset_free( ts2 )

  if ( nfail == 0 ) then
     write(*,'(a)') "uq_example: ALL PASSED"
  else
     write(*,'(a,i0,a)') "uq_example: ", nfail, " CHECK(S) FAILED"
     stop 1
  end if

CONTAINS

  REAL(8) FUNCTION dtarget_f( xx )
    implicit none
    real(8),intent(IN) :: xx
    dtarget_f = ( 2.d0*cos(2.d0*xx) - xx*sin(2.d0*xx) )*exp( -0.5d0*xx*xx )
  END FUNCTION dtarget_f

  REAL(8) FUNCTION dloss_of( nt )
    implicit none
    type(net_t),intent(IN) :: nt
    real(8) :: tv(nt%tab%na)
    integer :: n
    dloss_of = 0.d0
    do n=1,NPT
       call net_eval_hod( nt, wk, (/ xd(n) /), tv )
       dloss_of = dloss_of + 0.5d0*( tv(2) - dtarget_f( xd(n) ) )**2
    end do
    dloss_of = dloss_of/dble(NPT)
  END FUNCTION dloss_of

  REAL(8) FUNCTION target_f( xx )
    implicit none
    real(8),intent(IN) :: xx
    target_f = sin( 2.d0*xx )*exp( -0.5d0*xx*xx )
  END FUNCTION target_f

  SUBROUTINE seed_weights( nt, member )
    implicit none
    type(net_t),intent(INOUT) :: nt
    integer,intent(IN) :: member
    integer :: l, j, i
    do l=2,nt%nlayer
       do j=1,nt%ndim(l)
          do i=0,nt%ndim(l-1)
             nt%w(l,j,i) = 0.5d0*sin( 1.7d0*l + 0.9d0*j + 0.3d0*i + 2.1d0*member )
          end do
       end do
    end do
  END SUBROUTINE seed_weights

  SUBROUTINE one_seed( nt, xx, yy )
    implicit none
    type(net_t),intent(IN) :: nt
    real(8),intent(IN) :: xx, yy
    real(8) :: seed(nt%tab%na), tv(nt%tab%na)
    call net_eval_hod( nt, wk, (/ xx /), tv )
    seed = 0.d0
    seed(1) = tv(1) - yy
    call net_grad_point( nt, tw, (/ xx /), seed, g )
  END SUBROUTINE one_seed

  REAL(8) FUNCTION loss_of( nt )
    implicit none
    type(net_t),intent(IN) :: nt
    real(8) :: tv(nt%tab%na)
    integer :: n
    loss_of = 0.d0
    do n=1,NPT
       call net_eval_hod( nt, wk, (/ xd(n) /), tv )
       loss_of = loss_of + 0.5d0*( tv(1) - yd(n) )**2
    end do
    loss_of = loss_of/dble(NPT)
  END FUNCTION loss_of

  SUBROUTINE spread_vs_error( corr, growth )
    implicit none
    real(8),intent(OUT) :: corr, growth
    integer,parameter :: NS = 81
    real(8) :: xs, err(NS), sd(NS), me, ms, ce, cs, cc, sin_reg, sout_reg
    integer :: n, nin
    me=0.d0; ms=0.d0; nin=0; sin_reg=0.d0; sout_reg=0.d0
    do n=1,NS
       xs = -4.d0 + 8.d0*dble(n-1)/dble(NS-1)
       call comm_eval( cm, wk, (/ xs /), tm, tsd )
       err(n) = log10( abs( tm(1) - target_f( xs ) ) + 1.d-12 )
       sd(n)  = log10( tsd(1) + 1.d-12 )
       if ( abs(xs) <= 2.d0 ) then
          nin = nin + 1;  sin_reg = sin_reg + 10.d0**sd(n)
          me = me + err(n);  ms = ms + sd(n)
       else
          sout_reg = sout_reg + 10.d0**sd(n)
       end if
    end do
    me = sum(err)/dble(NS);  ms = sum(sd)/dble(NS)
    ce=0.d0; cs=0.d0; cc=0.d0
    do n=1,NS
       xs = -4.d0 + 8.d0*dble(n-1)/dble(NS-1)
       cc = cc + ( err(n)-me )*( sd(n)-ms )
       ce = ce + ( err(n)-me )**2
       cs = cs + ( sd(n)-ms )**2
    end do
    corr = cc/sqrt( max( ce*cs, 1.d-300 ) )
    growth = ( sout_reg/dble(NS-nin) )/max( sin_reg/dble(nin), 1.d-300 )
  END SUBROUTINE spread_vs_error

END PROGRAM example_uq
