!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (main.f90) is part of DNNF90.
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
! fit train.dat by Backpropagation; serial job
!
PROGRAM main

#ifdef _MPI_
  use parallel_module
#endif

  use global_variables
  use io_module
  use committee_run_module, only: run_committee
  use sgd_batch_module
  use hod_check_module

  implicit none
  integer :: ti,tf,tr
  integer :: ierr !mpi
  real(8) :: Stime,Etime !mpi

#ifdef _MPI_
  call MPI_INIT(ierr)
  Stime = MPI_WTIME()
  call MPI_COMM_RANK(MPI_COMM_WORLD,myrank,ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,nprocs,ierr)
#endif

  call system_clock(ti)
!----read from input_nn.dat----!
  call read_parameters
  call read_data

#ifdef _MPI_
  if (myrank==0) then
     call write_nn_param
  end if
#else
  call write_nn_param
#endif

! High-order derivative self-tests.  These do NOT evaluate the trained
! weights: run_hod_check overwrites the weight array with a
! deterministic formula first, so that the golden file is a property of
! the architecture and not of any particular run.  Nothing needs to be
! initialized here, and calling get_initial_weight would consume random
! numbers and change every run that follows.
  if ( iswitch_hod_check /= 0 ) then
#ifdef _MPI_
     if (myrank==0) call run_hod_check
#else
     call run_hod_check
#endif
  end if

  if ( iswitch_fit == 2 ) then
     ! Task COMMITTEE: evaluate an ensemble and stop.  No training, no
     ! history, no checkpoints.
#ifdef _MPI_
     if (myrank==0) call run_committee
#else
     call run_committee
#endif
     go to 800
  end if

  if ( iswitch_fit == 1 ) then
     call sgd_minibatch
  end if

#ifdef _MPI_
if (myrank==0) then
  if ( iswitch_out_deriv /= 0 ) call write_deriv !calc & write derivatives
end if
#else
  if ( iswitch_out_deriv /= 0 ) call write_deriv !calc & write derivatives
#endif

#ifdef _MPI_
  if (myrank==0) then
     call write_data("output")
  end if
  call MPI_Barrier(MPI_COMM_WORLD,ierr)
#else
  call write_data("output")
#endif

#ifdef _MPI_
  Etime = MPI_WTIME()
  if ( myrank==0 ) then
     write(*,'(i0,f20.6,a)') myrank, Etime-Stime, "[s]"
     write(*,'(i0,f20.6,a)') myrank, (Etime-Stime)/60.d0, "[min]"
     write(*,'(i0,f20.6,a)') myrank, (Etime-Stime)/3600.d0, "[h]"
  end if
  call MPI_FINALIZE(ierr)
#else
  call system_clock(tf,tr)
  write(*,'(f20.3,a)') (tf-ti)/dble(tr),"[s]"
  write(*,'(f10.3,a)') (tf-ti)/dble(tr)/60.d0,"[min]"
  write(*,'(f10.3,a)') (tf-ti)/dble(tr)/3600.d0,"[h]"
#endif

stop

  ! Landing point of the committee branch, which produces its own files
  ! and skips the training and output blocks above.  It still has to shut
  ! MPI down: leaving without MPI_FINALIZE is an error on every rank.
800 continue
#ifdef _MPI_
  call MPI_FINALIZE(ierr)
#endif
END PROGRAM main
