!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (committee_module.f90) is part of DNNF90.
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
! Committee (ensemble) utilities for uncertainty estimation.  The
! on-the-fly decision "when must the first-principles code be called" is
! usually taken from the disagreement of independently trained networks;
! this module packages that pattern on top of net_t.  The members are
! ordinary networks (the host trains them independently, for example
! with different initial weights); comm_eval returns the mean and the
! sample standard deviation of every carried derivative slot, so the
! uncertainty of forces (|alpha|=1 slots) is available, not only that of
! the energy.
MODULE committee_module

  use net_module

  implicit none

  PRIVATE
  PUBLIC :: committee_t, comm_init, comm_free, comm_eval

  TYPE :: committee_t
     integer :: nmem = 0
     type(net_t),allocatable :: mem(:)
  END TYPE committee_t

CONTAINS

  !> nmem networks of identical shape; each member deep-copies the
  !! table set at init (the tables are small, see net_init).  The
  !! caller initializes or loads the member weights afterwards
  !! (cm%mem(k)).
  SUBROUTINE comm_init( cm, nmem, nlayer, ndim, ts )
    use multi_index_bell_module, only: tabset_t
    implicit none
    type(committee_t),intent(OUT) :: cm
    integer,intent(IN) :: nmem, nlayer, ndim(nlayer)
    type(tabset_t),intent(IN),optional :: ts
    integer :: k
    cm%nmem = nmem
    allocate( cm%mem(nmem) )
    do k=1,nmem
       if ( present(ts) ) then
          call net_init( cm%mem(k), nlayer, ndim, ts )
       else
          call net_init( cm%mem(k), nlayer, ndim )
       end if
    end do
  END SUBROUTINE comm_init

  SUBROUTINE comm_free( cm )
    implicit none
    type(committee_t),intent(INOUT) :: cm
    integer :: k
    do k=1,cm%nmem
       call net_free( cm%mem(k) )
    end do
    if ( allocated(  cm%mem) ) deallocate( cm%mem )
    cm%nmem = 0
  END SUBROUTINE comm_free

  !> Mean and sample standard deviation of every slot over the members.
  !! wk is one work space (members share the shape); thread safe with
  !! one wk per thread as usual.
  SUBROUTINE comm_eval( cm, wk, x, tmean, tstd )
    implicit none
    type(committee_t),intent(IN) :: cm
    type(work_t),intent(INOUT) :: wk
    real(8),intent(IN)  :: x(:)
    real(8),intent(OUT) :: tmean(:), tstd(:)
    real(8) :: t(cm%mem(1)%tab%na)
    integer :: k, na
    na = cm%mem(1)%tab%na
    ! Running mean and second moment (Welford), accumulated in the
    ! caller's arrays so that no per-call storage is needed.
    !
    ! The textbook form sum(t^2) - sum(t)^2/n cancels catastrophically in
    ! exactly the regime a committee is used in: members that agree to
    ! 1e-7 on values of order one leave a variance of 1e-14 as the
    ! difference of two numbers near 16, which carries an absolute error
    ! of about 4e-15 and is therefore meaningless, or clipped to zero.
    ! The recurrence below never forms that difference.
    tmean(1:na) = 0.d0
    tstd(1:na)  = 0.d0
    do k=1,cm%nmem
       call net_eval_hod( cm%mem(k), wk, x, t )
       t(1:na) = t(1:na) - tmean(1:na)                     ! delta
       tstd(1:na)  = tstd(1:na) + t(1:na)*t(1:na)*dble(k-1)/dble(k)
       tmean(1:na) = tmean(1:na) + t(1:na)/dble(k)
    end do
    if ( cm%nmem > 1 ) then
       tstd(1:na) = sqrt( tstd(1:na)/dble(cm%nmem-1) )
    else
       ! a single member has no spread; the n-1 denominator would be 0/0
       tstd(1:na) = 0.d0
    end if
  END SUBROUTINE comm_eval

END MODULE committee_module
