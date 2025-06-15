!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (parallel_module.f90) is part of DNNF90.
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
MODULE parallel_module

  implicit none

#ifdef _MPI_
  include 'mpif.h'
#endif

  PUBLIC :: init_parallel,myrank,nprocs,JSTA,JEND
  PUBLIC :: replicated_step, slice_bounds, sum_over_ranks
  PUBLIC :: total_rows, gather_rows

  integer :: myrank = 0, nprocs = 1
  integer,allocatable :: JSTA(:),JEND(:)

  !> True when every rank holds the same weights at every point of the
  !! epoch, which the whole-set methods do: L-BFGS, Levenberg-Marquardt
  !! and the dual natural gradient all take one step from one state.
  !! Their sweeps over the training set may then be split across ranks
  !! and summed, because every partial sum is evaluated at the same
  !! weights.  The local-SGD methods must not do this: between two
  !! averaging points their weights differ, so a sum of partial costs
  !! would add pieces measured at different states.  The trainer sets
  !! this once, from the method name.
  logical :: replicated_step = .false.

#ifndef _MPI_
  ! The serial build carries the same names so that the callers need no
  ! conditionals: one rank, no reduction, the whole range.
#endif

CONTAINS

  SUBROUTINE init_parallel(N)
    implicit none
    integer,intent(IN) :: N
    integer :: irank

    allocate ( JSTA(0:nprocs-1),JEND(0:nprocs-1) )

    do irank=0,nprocs-1
       JSTA(irank) = 1 + irank*floor(dble(N)/dble(nprocs))
       if ( irank==nprocs-1 ) then
          JEND(irank) = N
       else
          JEND(irank) = JSTA(irank) + floor(dble(N)/dble(nprocs)) - 1
       end if
    end do

  END SUBROUTINE init_parallel

  !> This rank's half-open share of 1..n, or the whole of it when the
  !! ranks are not in lockstep (each rank then needs the global sum for
  !! its own state) or when there is one rank.
  SUBROUTINE slice_bounds( n, lo, hi )
    implicit none
    integer,intent(IN) :: n
    integer,intent(OUT) :: lo, hi
    integer :: share
    lo = 1;  hi = n
#ifdef _MPI_
    if ( .not. replicated_step .or. nprocs <= 1 ) return
    share = n/nprocs
    lo = 1 + myrank*share
    if ( myrank == nprocs-1 ) then
       hi = n
    else
       hi = lo + share - 1
    end if
#endif
  END SUBROUTINE slice_bounds

  !> Sum an array over the ranks in place, when and only when the sweep
  !! that filled it was split by slice_bounds.
  SUBROUTINE sum_over_ranks( a, n )
    implicit none
    integer,intent(IN) :: n
    real(8),intent(INOUT) :: a(n)
#ifdef _MPI_
    real(8),allocatable :: buf(:)
    integer :: ierr
    if ( .not. replicated_step .or. nprocs <= 1 .or. n <= 0 ) return
    allocate( buf(n) )
    call mpi_allreduce( a, buf, n, mpi_real8, mpi_sum, mpi_comm_world, ierr )
    a = buf
    deallocate( buf )
#endif
  END SUBROUTINE sum_over_ranks

  !> The number of rows one step is built from, summed over the ranks.
  !! The shares need not be equal: a stratified draw hands the remainder
  !! of each term to the low ranks, so a rank may hold one row more than
  !! its neighbour.
  INTEGER FUNCTION total_rows( nloc )
    implicit none
    integer,intent(IN) :: nloc
    integer :: ntot, ierr
    total_rows = nloc
#ifdef _MPI_
    if ( .not. replicated_step .or. nprocs <= 1 ) return
    call mpi_allreduce( nloc, ntot, 1, mpi_integer, mpi_sum, &
         mpi_comm_world, ierr )
    total_rows = ntot
#endif
  END FUNCTION total_rows

  !> Concatenate the per-rank blocks of a (n, nloc) row array into the
  !! (n, sum nloc) array every rank then solves with.  The rows are
  !! stored transposed for exactly this reason: each rank's block is
  !! contiguous, so the gather needs no packing.  Unequal shares are
  !! handled, which is why this is a v-gather.
  SUBROUTINE gather_rows( a, n, nloc )
    implicit none
    integer,intent(IN) :: n, nloc
    real(8),intent(INOUT) :: a(n,*)
#ifdef _MPI_
    real(8),allocatable :: buf(:,:)
    integer,allocatable :: cnt(:), disp(:)
    integer :: ierr, r, ntot
    if ( .not. replicated_step .or. nprocs <= 1 ) return
    allocate( cnt(0:nprocs-1), disp(0:nprocs-1) )
    call mpi_allgather( nloc, 1, mpi_integer, cnt, 1, mpi_integer, &
         mpi_comm_world, ierr )
    disp(0) = 0
    do r = 1, nprocs-1
       disp(r) = disp(r-1) + cnt(r-1)*n
    end do
    ntot = sum( cnt )
    cnt = cnt*n
    allocate( buf(n,ntot) )
    call mpi_allgatherv( a, n*nloc, mpi_real8, buf, cnt, disp, mpi_real8, &
         mpi_comm_world, ierr )
    a(1:n,1:ntot) = buf
    deallocate( buf, cnt, disp )
#endif
  END SUBROUTINE gather_rows

END MODULE parallel_module
