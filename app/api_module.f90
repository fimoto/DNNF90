!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (api_module.f90) is part of DNNF90.
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
! Minimal embedding API.
!
! Purpose: let a host Fortran code (for example a DFT package) evaluate a
! trained network and all carried mixed derivatives at arbitrary points
! with three calls.  The module only wraps existing routines; it does not
! modify any of them.
!
! Typical use from a host code, executed in a directory that contains
! input_nn.dat (with "Restart 1"), the data files it references, and the
! trained nn_weight.dat:
!
!     use api_module
!     real(8) :: x(2), t(nalpha_max)
!     call dnnf90_init                       ! tables + trained weights
!     n = dnnf90_nderiv()                    ! number of carried indices
!     call dnnf90_eval_hod( x, t )           ! t(1:n) = u and derivatives
!
! The canonical order of the indices is written to hod_alpha_order.dat at
! initialization and can be queried with dnnf90_alpha.
!
! Limitation: the underlying code keeps one network in module storage, so
! one process holds one network instance (thread parallelism inside a
! host code must use one instance per MPI rank, not per thread).
! -----------------------------------------------------------------------
MODULE api_module

  use global_variables
  use io_module,               only: read_parameters, read_data
  use init_weight_module,      only: get_initial_weight
  use multi_index_bell_module, only: NUM_alpha, alpha_list, HOD_D0, reset_hod_tables

  use lib_net_module, only: lnet_forward_hod, lnet_sync_weights, lnet_free, lnet_nalpha
  implicit none

  PRIVATE
  PUBLIC :: dnnf90_init, dnnf90_free
  PUBLIC :: dnnf90_nderiv, dnnf90_ndim_in, dnnf90_alpha, dnnf90_eval_hod

CONTAINS

  SUBROUTINE dnnf90_init
    ! Reads input_nn.dat and the referenced data files from the current
    ! directory, builds the multi-index and Bell tables, and loads the
    ! trained weights (set "Restart 1" in input_nn.dat).
    implicit none
    call read_parameters
    call read_data
    call get_initial_weight
    ! Build and synchronize the evaluation network here, rather than on
    ! the first evaluation.  Without it a plain DATA case leaves the
    ! derivative table empty until something is evaluated, so that
    ! dnnf90_nderiv returns zero immediately after init and a different
    ! number afterwards -- and a caller who sized its buffer from the
    ! first answer would then overrun it.
    call lnet_sync_weights( weight )
  END SUBROUTINE dnnf90_init

  SUBROUTINE dnnf90_free
    ! Releases the network, its work spaces and its tables.  A host that
    ! embeds the library for the length of its own run does not have to
    ! call this, but one that initializes repeatedly does.
    implicit none
    call lnet_free
    call global_free
    call reset_hod_tables
  END SUBROUTINE dnnf90_free

  INTEGER FUNCTION dnnf90_nderiv()
    ! Number of carried multi-indices (u itself is index 1).  Read from
    ! the evaluation network's own table, which dnnf90_init has already
    ! built, so that this is the same number dnnf90_eval_hod will fill
    ! and the same one dnnf90_alpha describes.
    implicit none
    dnnf90_nderiv = lnet_nalpha()
  END FUNCTION dnnf90_nderiv

  INTEGER FUNCTION dnnf90_ndim_in()
    ! Number of input variables of the trained network.
    implicit none
    dnnf90_ndim_in = ndim(1)
  END FUNCTION dnnf90_ndim_in

  SUBROUTINE dnnf90_alpha( ia, a )
    ! The multi-index alpha of carried slot ia (a has HOD_D0 entries).
    implicit none
    integer,intent(IN)  :: ia
    integer,intent(OUT) :: a(HOD_D0)
    ! A host is exactly where a slot index out of range comes from, and
    ! alpha_list would be read past its end without a word.
    if ( ia < 1 .or. ia > NUM_alpha ) then
       write(*,*) "dnnf90_alpha: slot", ia, " is outside 1 ..", NUM_alpha
       stop
    end if
    a(1:HOD_D0) = alpha_list(1:HOD_D0,ia)
  END SUBROUTINE dnnf90_alpha

  SUBROUTINE dnnf90_eval_hod( x, t )
    ! Evaluates the network at x(1:D0).  On return, t(ia) is the mixed
    ! derivative of the scalar output for the multi-index of slot ia, in
    ! the canonical order (t(1) is u itself).
    implicit none
    real(8),intent(IN)  :: x(*)
    real(8),intent(OUT) :: t(*)
    ! The caller owns the weight array and may have just changed it.
    call lnet_sync_weights( weight )
    zmat(1,1:ndim(1)) = x(1:ndim(1))
    call lnet_forward_hod( zmat(1,1:ndim(1)), t(1:NUM_alpha) )

  END SUBROUTINE dnnf90_eval_hod

END MODULE api_module
