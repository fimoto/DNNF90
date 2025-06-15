!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (example_mlff.f90) is part of DNNF90.
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
! Demonstration of the instance based evaluation path, in the shape a
! machine learning force field needs it:
!
!   * several networks resident at the same time (one per element species),
!   * an OpenMP loop over atoms, one work_t per thread,
!   * results checked against the trainer's own evaluator.
!
! Run it in a trained benchmark directory, for example:
!
!     cd bench/kdv
!     sed -i 's/^Restart      0 /Restart      1 /' input_nn.dat
!     OMP_NUM_THREADS=4 ../../build/mlff_example.out
!
PROGRAM example_mlff
!$ use omp_lib
  use api_module
  use net_module
  implicit none

  integer,parameter :: NSPEC = 3          ! pretend: three element species
  integer,parameter :: NATOM = 20000      ! atoms to evaluate
  type(net_t)  :: nets(NSPEC)
  type(work_t),allocatable :: wk(:)
  real(8),allocatable :: xall(:,:), eref(:), epar(:)
  real(8),allocatable :: tbuf(:)
  real(8),allocatable :: tall(:,:)      ! one column per thread (no private allocatables)
  integer :: ic0, ic1, crate
  integer :: nalpha, ndin, ia, k, n, ith, nth
  real(8) :: dmax

  ! shared, read-only tables and the trainer's own copy of the network
  call dnnf90_init
  nalpha = dnnf90_nderiv()
  ndin   = dnnf90_ndim_in()

  ! several independent networks in one process (impossible with the
  ! module level state of the trainer)
  do k=1,NSPEC
     call net_load( nets(k), 'nn_weight.dat' )
  end do
  write(*,'(a,i0,a,i0,a)') "networks resident: ", NSPEC, &
       "   carried derivatives: ", nalpha, ""

  allocate( xall(ndin,NATOM), eref(NATOM), epar(NATOM), tbuf(nalpha) )
  do n=1,NATOM
     do ia=1,ndin
        xall(ia,n) = -1.d0 + 2.d0*dble(mod(7*n+13*ia,1000))/1000.d0
     end do
  end do

  ! reference: the trainer's evaluator, one point at a time
  call system_clock(ic0,crate)
  do n=1,NATOM
     call dnnf90_eval_hod( xall(1,n), tbuf )
     eref(n) = tbuf(1)
  end do
  call system_clock(ic1)
  write(*,'(a,f8.3,a)') "serial reference (module state) : ", dble(ic1-ic0)/dble(crate), " s"

  nth = 1
!$ nth = omp_get_max_threads()
  allocate( wk(0:nth-1) )
  do k=0,nth-1
     call work_init( wk(k), nets(1) )
  end do

  call system_clock(ic0,crate)
  allocate( tall(nalpha,0:nth-1) )
  !$omp parallel default(shared) private(n,ith)
  ith = 0
!$ ith = omp_get_thread_num()
  !$omp do schedule(static)
  do n=1,NATOM
     ! species assignment is arbitrary here; the point is that different
     ! atoms use different network instances concurrently
     call net_eval_hod( nets(1+mod(n,NSPEC)), wk(ith), xall(:,n), tall(:,ith) )
     epar(n) = tall(1,ith)
  end do
  !$omp end do
  !$omp end parallel
  call system_clock(ic1)
  write(*,'(a,i0,a,f8.3,a)') "instance based, threads=", nth, "        : ", &
       dble(ic1-ic0)/dble(crate), " s (wall)"

  dmax = maxval( abs(epar(1:NATOM)-eref(1:NATOM)) )
  write(*,'(a,e12.4)') "max |instance - trainer| over all atoms: ", dmax
  if ( dmax == 0.d0 ) then
     write(*,'(a)') "RESULT: bitwise identical"
  else
     write(*,'(a)') "RESULT: MISMATCH"
  stop 1
  end if

  do k=0,nth-1
     call work_free( wk(k) )
  end do
  do k=1,NSPEC
     call net_free( nets(k) )
  end do

END PROGRAM example_mlff
