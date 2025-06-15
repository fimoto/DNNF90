!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (example_train.f90) is part of DNNF90.
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
! (MIT License; see LICENSE at the repository root.)
!
! Verification and demonstration of the instance based training path.
!
!   test 1  the gradient of one point is compared with central
!           differences of the seeded loss.  The library kernel is the
!           only propagation engine in the tree, so it is checked against
!           the definition and not against a second implementation
!   test 2  gradients are accumulated over many points with one work
!           space and one accumulator per thread, then reduced, and the
!           result is compared with the serial accumulation
!   test 3  several networks train side by side in one process, which is
!           the shape a force field needs, one network per element
!
! Each test carries a criterion and the program stops with a nonzero
! status if one is not met.
!
! Run it in a trained benchmark directory, for example:
!
!     cd bench/kdv
!     sed -i 's/^Restart      0 /Restart      1 /' input_nn.dat
!     OMP_NUM_THREADS=4 ../../build/train_example.out
!
PROGRAM example_train
!$ use omp_lib
  use global_variables, only: Nlayer, ndim, weight, ndim_max
  use api_module
  use net_module
  use train_module
  implicit none

  integer,parameter :: NSPEC = 3
  integer,parameter :: NPT   = 4000
  type(net_t)  :: nets(NSPEC)
  type(twork_t),allocatable :: tw(:)
  type(grad_t),allocatable  :: gth(:)
  type(grad_t),allocatable :: gsp(:,:)
  type(grad_t) :: gtot, gser, gspec(NSPEC)
  real(8),allocatable :: xall(:,:), seed(:,:), tfd(:)
  integer :: nalpha, ndin, n, k, ith, nth, ia, ep, l, i, j
  real(8) :: hfd, wsave, lossp, lossm, fdval, gmax
  real(8) :: dmax, e0, e1
  real(8),allocatable :: tbuf(:)
  integer :: nfail = 0

  call dnnf90_init
  nalpha = dnnf90_nderiv()
  ndin   = dnnf90_ndim_in()

  do k=1,NSPEC
     call net_load( nets(k), 'nn_weight.dat' )
  end do

  allocate( xall(ndin,NPT), seed(nalpha,NPT), tbuf(nalpha) )
  allocate( tfd(nalpha) )
  do n=1,NPT
     do ia=1,ndin
        xall(ia,n) = -1.d0 + 2.d0*dble(mod(11*n+5*ia,997))/997.d0
     end do
     do ia=1,nalpha
        seed(ia,n) = 0.01d0*dble(mod(3*n+ia,17)-8)
     end do
  end do

  ! ---------------- test 1: one point against finite differences ----------
  ! The library kernel is the only propagation engine in the tree, so it
  ! is verified against the definition rather than against a second
  ! implementation: for the seeded loss L = sum_alpha seed_alpha T_alpha,
  ! dL/dw must equal the central difference of L in that weight.
  call grad_init( gser, nets(1) )
  allocate( tw(0:0) )
  call twork_init( tw(0), nets(1) )
  call net_grad_point( nets(1), tw(0), xall(:,1), seed(:,1), gser )

  hfd = 1.d-5
  dmax = 0.d0
  gmax = 0.d0
  do l=2,Nlayer
     do j=1,ndim(l)
        do i=0,ndim(l-1)
           wsave = nets(1)%w(l,j,i)
           nets(1)%w(l,j,i) = wsave + hfd
           call net_forward_point( nets(1), tw(0), xall(:,1), tfd )
           lossp = sum( seed(1:nalpha,1)*tfd(1:nalpha) )
           nets(1)%w(l,j,i) = wsave - hfd
           call net_forward_point( nets(1), tw(0), xall(:,1), tfd )
           lossm = sum( seed(1:nalpha,1)*tfd(1:nalpha) )
           nets(1)%w(l,j,i) = wsave
           fdval = (lossp-lossm)/(2.d0*hfd)
           dmax = max( dmax, abs(fdval-gser%nabla(l,j,i)) )
           gmax = max( gmax, abs(gser%nabla(l,j,i)) )
        end do
     end do
  end do
  ! The difference is measured against the size of the gradient itself:
  ! a central difference cannot resolve a component that is many orders
  ! below the largest one, so a per-component ratio would report the
  ! noise of the reference, not an error of the adjoint.
  dmax = dmax/max( gmax, 1.d-12 )
  if ( dmax < 1.d-6 ) then
     write(*,'(a,e12.4,a)') "test 1  one point gradient vs finite diff: max rel ", dmax, &
          "  passed"
  else
     write(*,'(a,e12.4,a)') "test 1  one point gradient vs finite diff: max rel ", dmax, &
          "  FAILED"
     nfail = nfail + 1
  end if
  call twork_free( tw(0) )
  deallocate( tw )

  ! ---------------- test 2: threaded accumulation ----------------
  call grad_zero( gser )
  allocate( tw(0:0) )
  call twork_init( tw(0), nets(1) )
  do n=1,NPT
     call net_grad_point( nets(1), tw(0), xall(:,n), seed(:,n), gser )
  end do
  call twork_free( tw(0) )
  deallocate( tw )

  nth = 1
!$ nth = omp_get_max_threads()
  allocate( tw(0:nth-1), gth(0:nth-1) )
  do k=0,nth-1
     call twork_init( tw(k), nets(1) )
     call grad_init( gth(k), nets(1) )
  end do
  !$omp parallel default(shared) private(n,ith)
  ith = 0
!$ ith = omp_get_thread_num()
  !$omp do schedule(static)
  do n=1,NPT
     call net_grad_point( nets(1), tw(ith), xall(:,n), seed(:,n), gth(ith) )
  end do
  !$omp end do
  !$omp end parallel

  call grad_init( gtot, nets(1) )
  do k=0,nth-1                      ! fixed reduction order
     call grad_add( gtot, gth(k) )
  end do
  dmax = maxval( abs( gtot%nabla - gser%nabla ) ) &
       / max( maxval( abs(gser%nabla) ), 1.d-300 )
  ! The two paths visit the same points with the same seeds and reduce in
  ! a fixed order, so they may differ only by the order of the sums
  ! within a thread: anything above rounding is a race or a lost term.
  write(*,'(a,i0,a,e12.4)') "test 2  threads=", nth, &
       " accumulation vs serial : max rel diff ", dmax
  if ( dmax > 1.d-12 ) then
     write(*,'(a)') "        FAILED: the threaded sum does not match the serial one"
     nfail = nfail + 1
  end if

  ! ---------------- test 3: several networks training at once ----------------
  ! per-thread AND per-species accumulators: two threads must never add
  ! into the same grad_t (that would be a data race)
  allocate( gsp(NSPEC,0:nth-1) )
    do k=1,NSPEC
       call grad_init( gspec(k), nets(k) )
       do n=0,nth-1
          call grad_init( gsp(k,n), nets(k) )
       end do
    end do
    call net_eval_energy( nets(1), xall(:,1), tbuf, e0 )
    do ep=1,200
       do k=1,NSPEC
          call grad_zero( gspec(k) )
          do n=0,nth-1
             call grad_zero( gsp(k,n) )
          end do
       end do
       !$omp parallel default(shared) private(n,ith)
       ith = 0
!$     ith = omp_get_thread_num()
       !$omp do schedule(static)
       do n=1,NPT
          call net_grad_point( nets(1+mod(n,NSPEC)), tw(ith), xall(:,n), &
                               seed(:,n), gsp(1+mod(n,NSPEC),ith) )
       end do
       !$omp end do
       !$omp end parallel
       do k=1,NSPEC
          do n=0,nth-1                  ! fixed reduction order
             call grad_add( gspec(k), gsp(k,n) )
          end do
          call opt_adam_step( nets(k), gspec(k), 1.d-4, 0.9d0, 0.999d0, 1.d-8, NPT, ep )
       end do
    end do
    do k=1,NSPEC
       do n=0,nth-1
          call grad_free( gsp(k,n) )
       end do
    end do
  call net_eval_energy( nets(1), xall(:,1), tbuf, e1 )
  write(*,'(a,i0,a,2e13.5)') "test 3  ", NSPEC, &
       " networks trained together, output before/after: ", e0, e1
  ! Training must have moved the network and left it finite: a silent
  ! no-op and a diverged net both look like a printed pair of numbers.
  if ( .not. ( abs(e1-e0) > 0.d0 .and. abs(e1) < huge(1.d0) ) ) then
     write(*,'(a)') "        FAILED: the trained output did not move, or is not finite"
     nfail = nfail + 1
  end if

  do k=0,nth-1
     call twork_free( tw(k) )
     call grad_free( gth(k) )
  end do
  do k=1,NSPEC
     call grad_free( gspec(k) )
     call net_free( nets(k) )
  end do
  call grad_free( gser );  call grad_free( gtot )
  deallocate( tw, gth, gsp )
  deallocate( xall, seed, tbuf, tfd )

  if ( nfail == 0 ) then
     write(*,'(a)') "train_example: ALL PASSED"
  else
     write(*,'(a,i0,a)') "train_example: ", nfail, " CHECK(S) FAILED"
     stop 1
  end if

CONTAINS

  SUBROUTINE net_eval_energy( nt, x, t, e )
    implicit none
    type(net_t),intent(IN) :: nt
    real(8),intent(IN) :: x(:)
    real(8),intent(INOUT) :: t(:)
    real(8),intent(OUT) :: e
    type(work_t) :: w1
    call work_init( w1, nt )
    call net_eval_hod( nt, w1, x, t )
    e = t(1)
    call work_free( w1 )
  END SUBROUTINE net_eval_energy

END PROGRAM example_train
