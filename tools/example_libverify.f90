!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (example_libverify.f90) is part of DNNF90.
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
! Self verification of the library path, independent of the trainer:
!
!   test 1  table sanity across the force-field regime: |A| against the
!           closed form sum_p C(p+D0-1,p) for D0 up to 100 and K up to 5
!   test 2  finite-difference check of the weight gradient with the loss
!           seeded on the HIGHEST-order slot (this is the property that
!           arbitrary high-order training rests on), on a wide network
!   test 3  finite-difference check of the carried derivatives
!           themselves: slot values against central differences of the
!           network output in the inputs
!   test 4  the extended Kalman filter on a network that is linear in its
!           weights, against the ridge normal-equation solution obtained
!           by an independent dense solve
!
! Each test carries a tolerance and the program stops with a nonzero
! status if any of them is exceeded, so it can be used as a test and not
! only as a demonstration.
!
! Run: ./libverify_example.out
PROGRAM example_libverify
  use multi_index_bell_module
  use net_module
  use train_module
  implicit none

  integer :: nfail = 0

  call table_sanity
  call grad_fd_check
  call slot_fd_check
  call kalman_rls_check

  if ( nfail == 0 ) then
     write(*,'(a)') "libverify: ALL PASSED"
  else
     write(*,'(a,i0,a)') "libverify: ", nfail, " CHECK(S) FAILED"
     stop 1
  end if

CONTAINS

  !> Report one test and count it if it exceeded its tolerance.
  SUBROUTINE verdict( name, err, tol )
    implicit none
    character(*),intent(IN) :: name
    real(8),intent(IN) :: err, tol
    if ( err > tol ) then
       write(*,'(a,a,a,e11.3,a,e11.3)') "  ", name, " FAILED: ", err, " > ", tol
       nfail = nfail + 1
    else
       write(*,'(a,a,a,e11.3,a,e11.3)') "  ", name, " passed: ", err, " <= ", tol
    end if
  END SUBROUTINE verdict

  SUBROUTINE table_sanity
    implicit none
    integer :: d0s(6) = (/ 10, 20, 30, 50, 100, 6 /)
    integer :: ks(6)  = (/  2,  3,  2,  2,   2, 5 /)
    integer :: c, p, i, dummy(1,1)
    integer(8) :: expect, term
    type(tabset_t) :: ts
    logical :: ok
    ok = .true.
    do c=1,6
       call tabset_init( ts, d0s(c), ks(c), 0, dummy )
       expect = 0_8
       do p=0,ks(c)
          term = 1_8
          do i=1,p
             term = term*int(d0s(c)-1+i,8)/int(i,8)
          end do
          expect = expect + term
       end do
       if ( int(ts%na,8) /= expect ) then
          write(*,'(a,i4,i3,2i10)') "  MISMATCH D0,K,got,expect:", &
               d0s(c), ks(c), ts%na, expect
          ok = .false.
       end if
    end do
    write(*,'(a)') "test 1  table sizes vs closed form (D0<=100, K<=5)"
    if ( ok ) then
       call verdict( "test 1", 0.d0, 1.d0 )
    else
       call verdict( "test 1", 1.d0, 0.d0 )
    end if
    call tabset_free( ts )
  END SUBROUTINE table_sanity

  SUBROUTINE grad_fd_check
    implicit none
    integer,parameter :: NL = 4, W = 256, D0 = 8, K = 4
    integer :: dims(NL), dummy(1,1), i, l, j, islot, nchk
    type(tabset_t) :: ts
    type(net_t) :: nt
    type(twork_t) :: tw
    type(work_t) :: wk
    type(grad_t) :: g
    real(8),allocatable :: x(:), t(:), seed(:)
    real(8) :: c0, lp, lm, fd, an, emax, gmax, h
    dims = (/ D0, W, W, 1 /)
    call tabset_init( ts, D0, K, 0, dummy )
    call net_init( nt, NL, dims, ts )
    do l=2,NL
       do j=1,nt%ndim(l)
          do i=0,nt%ndim(l-1)
             nt%w(l,j,i) = 0.1d0*sin( 1.7d0*l + 0.9d0*j + 0.3d0*i ) + 0.02d0
          end do
       end do
    end do
    call twork_init( tw, nt );  call work_init( wk, nt );  call grad_init( g, nt )
    allocate( x(D0), t(ts%na), seed(ts%na) )
    do i=1,D0
       x(i) = 0.1d0*dble(i) - 0.35d0
    end do
    islot = ts%na                        ! highest-order slot |alpha| = K
    c0 = 0.3d0
    ! analytic gradient of L = 0.5 (T_islot - c0)^2
    call net_eval_hod( nt, wk, x, t )
    seed = 0.d0
    seed(islot) = t(islot) - c0
    call grad_zero( g )
    call net_grad_point( nt, tw, x, seed, g )
    ! central finite differences on a spread of weights
    h = 1.d-6
    emax = 0.d0;  gmax = 0.d0;  nchk = 0
    do l=2,NL
       do j=1,nt%ndim(l),max(1,nt%ndim(l)/4)
          do i=0,nt%ndim(l-1),max(1,nt%ndim(l-1)/4)
             nt%w(l,j,i) = nt%w(l,j,i) + h
             call net_eval_hod( nt, wk, x, t );  lp = 0.5d0*( t(islot)-c0 )**2
             nt%w(l,j,i) = nt%w(l,j,i) - 2.d0*h
             call net_eval_hod( nt, wk, x, t );  lm = 0.5d0*( t(islot)-c0 )**2
             nt%w(l,j,i) = nt%w(l,j,i) + h
             fd = ( lp - lm )/( 2.d0*h )
             an = g%nabla(l,j,i)
             emax = max( emax, abs(an-fd) )
             gmax = max( gmax, abs(fd) )
             nchk = nchk + 1
          end do
       end do
    end do
    ! relative to the size of the gradient over the scan, not element by
    ! element with a floor, which is an absolute test in disguise
    emax = emax/max( gmax, 1.d-300 )
    write(*,'(a,i2,a,i4,a,i0,a)') "test 2  grad FD on |alpha|=", K, &
         " slot, width ", W, ", ", nchk, " weights"
    call verdict( "test 2", emax, 1.d-4 )
    deallocate( x, t, seed )
    call grad_free( g );  call work_free( wk );  call twork_free( tw )
    call net_free( nt );  call tabset_free( ts )
  END SUBROUTINE grad_fd_check

  SUBROUTINE slot_fd_check
    implicit none
    integer,parameter :: NL = 3, W = 24, D0 = 3, K = 3
    integer :: dims(NL), dummy(1,1), i, l, j, ia, iv, a(D0)
    type(tabset_t) :: ts
    type(net_t) :: nt
    type(work_t) :: wk
    real(8) :: x(D0), h, emax
    real(8),allocatable :: t0(:), tp(:), tm(:)
    dims = (/ D0, W, 1 /)
    call tabset_init( ts, D0, K, 0, dummy )
    call net_init( nt, NL, dims, ts )
    do l=2,NL
       do j=1,nt%ndim(l)
          do i=0,nt%ndim(l-1)
             nt%w(l,j,i) = 0.2d0*sin( 1.3d0*l + 0.7d0*j + 0.5d0*i )
          end do
       end do
    end do
    call work_init( wk, nt )
    allocate( t0(ts%na), tp(ts%na), tm(ts%na) )
    x = (/ 0.2d0, -0.1d0, 0.3d0 /)
    h = 1.d-5
    emax = 0.d0
    ! every slot of degree p equals the FD (in one variable) of a slot of
    ! degree p-1: T_{alpha} = d T_{alpha - e_iv} / d x_iv
    do ia=2,ts%na
       a = ts%alpha_list(:,ia)
       do iv=1,D0
          if ( a(iv) > 0 ) exit
       end do
       a(iv) = a(iv) - 1
       x(iv) = x(iv) + h;  call net_eval_hod( nt, wk, x, tp )
       x(iv) = x(iv) - 2.d0*h;  call net_eval_hod( nt, wk, x, tm )
       x(iv) = x(iv) + h
       call net_eval_hod( nt, wk, x, t0 )
       emax = max( emax, abs( t0(ia) - ( tp(alpha_index(a)) - tm(alpha_index(a)) ) &
                                        /( 2.d0*h ) ) &
                         /max( abs(t0(ia)), 1.d-8 ) )
    end do
    write(*,'(a,i2)') "test 3  every slot vs FD of its parent up to K=", K
    call verdict( "test 3", emax, 1.d-6 )
    deallocate( t0, tp, tm )
    call work_free( wk );  call net_free( nt );  call tabset_free( ts )
  END SUBROUTINE slot_fd_check

  !> The filter on a 2-layer (linear-in-weights) net must reproduce the
  !! ridge normal-equation solution exactly: this validates the full
  !! rank-1 update algebra (gain, covariance, forgetting at lambda=1)
  !! against an independent direct solve.
  SUBROUTINE kalman_rls_check
    use kalman_module
    implicit none
    integer,parameter :: D0 = 3, NS = 60
    integer :: dims(2), dummy(1,1), i, j, n
    type(tabset_t) :: ts
    type(net_t) :: nt
    type(twork_t) :: tw
    type(grad_t) :: g
    type(kalman_t) :: kf
    real(8) :: x(D0,NS), y(NS), xe(D0+1), p0
    real(8) :: A(D0+1,D0+1), b(D0+1), wkf(D0+1), emax, piv
    call tabset_init( ts, D0, 1, 0, dummy )
    dims = (/ D0, 1 /)
    call net_init( nt, 2, dims, ts )
    call twork_init( tw, nt );  call grad_init( g, nt )
    p0 = 50.d0
    call kf_init( kf, nt, p0, 1.d0, 1.d0 )
    do n=1,NS
       do i=1,D0
          x(i,n) = sin( 1.7d0*n + 0.9d0*i )      ! deterministic samples
       end do
       y(n) = 0.7d0 - 1.3d0*x(1,n) + 0.4d0*x(2,n) + 2.1d0*x(3,n)
    end do
    do n=1,NS
       call kf_update( nt, tw, g, kf, x(:,n), 1, y(n) )
    end do
    wkf(1) = nt%w(2,1,0);  wkf(2:D0+1) = nt%w(2,1,1:D0)
    A = 0.d0;  b = 0.d0
    do i=1,D0+1
       A(i,i) = 1.d0/p0
    end do
    do n=1,NS
       xe(1) = 1.d0;  xe(2:D0+1) = x(:,n)
       do i=1,D0+1
          A(i,:) = A(i,:) + xe(i)*xe(:)
          b(i)   = b(i)   + xe(i)*y(n)
       end do
    end do
    ! small dense solve by Gaussian elimination with partial pivoting
    do i=1,D0+1
       j = maxloc( abs(A(i:D0+1,i)), 1 ) + i - 1
       if ( j /= i ) then
          xe = A(i,:);  A(i,:) = A(j,:);  A(j,:) = xe
          piv = b(i);  b(i) = b(j);  b(j) = piv
       end if
       do j=i+1,D0+1
          piv = A(j,i)/A(i,i)
          A(j,:) = A(j,:) - piv*A(i,:)
          b(j)   = b(j)   - piv*b(i)
       end do
    end do
    do i=D0+1,1,-1
       b(i) = ( b(i) - dot_product( A(i,i+1:D0+1), b(i+1:D0+1) ) )/A(i,i)
    end do
    emax = maxval( abs(wkf-b) )/maxval( abs(b) )
    write(*,'(a)') "test 4  Kalman on a linear net vs ridge solve"
    call verdict( "test 4", emax, 1.d-9 )
    call kf_free( kf );  call grad_free( g );  call twork_free( tw )
    call net_free( nt );  call tabset_free( ts )
  END SUBROUTINE kalman_rls_check

END PROGRAM example_libverify
