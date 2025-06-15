!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (hod_dump.f90) is part of DNNF90.
!
!  DNNF90 is free software released under the MIT License.
!  You should have received a copy of the MIT License (file LICENSE
!  in the root directory of this distribution) along with DNNF90.
!  If not, see <https://opensource.org/licenses/MIT>.
!
! Every carried derivative of a trained network, at the points of a file.
!
! The trainer writes first input derivatives (`Output_deriv 1`) and, for
! a MATH_HOD term, the full set against its targets.  A collocation run
! has neither, so a claim about the high derivatives of a PINN solution
! -- for instance how far the seventh derivatives of the seventh-order ZK
! network are from the exact soliton (Section 6) -- has no file to read.
! This tool fills that gap: it loads a weight file, carries the dense set
! through order K, and prints every slot at every point.
!
!   hod_dump.out <weight.dat> <points.dat> <D0> <K> <ncol>
!
!     weight.dat   nn_weight.dat of the run
!     points.dat   whitespace-separated rows; the first D0 columns are
!                  used as the input point (so colloc.dat works as is)
!     ncol         number of columns per row of points.dat
!
! Output: comment lines '# alpha <i>: <alpha(1:D0)>' giving the column
! order, then one line per point with x(1:D0) followed by the NUM_alpha
! derivative values in that order.
!
program hod_dump
  use multi_index_bell_module, only: tabset_t, tabset_init, tabset_free, &
       alpha_list, NUM_alpha
  use net_module, only: net_t, work_t, net_load, net_free, work_init, &
       work_free, net_eval_hod
  implicit none
  type(tabset_t) :: ts
  type(net_t) :: nt
  type(work_t) :: wk
  character(256) :: wfile, pfile, carg
  integer :: d0, kmax, ncol, ios, ia, np
  integer :: dummy(1,1)
  real(8), allocatable :: x(:), t(:), row(:)

  if ( command_argument_count() < 5 ) then
     write(*,*) 'usage: hod_dump.out <weight.dat> <points.dat> <D0> <K> <ncol>'
     stop 1
  end if
  call get_command_argument(1,wfile)
  call get_command_argument(2,pfile)
  call get_command_argument(3,carg); read(carg,*) d0
  call get_command_argument(4,carg); read(carg,*) kmax
  call get_command_argument(5,carg); read(carg,*) ncol

  call tabset_init( ts, d0, kmax, 0, dummy )
  call net_load( nt, trim(wfile), ts )
  call work_init( wk, nt )
  allocate( x(d0), t(nt%tab%na), row(ncol) )

  write(*,'(a,i0)') '# na=', nt%tab%na
  do ia = 1, nt%tab%na
     write(*,'(a,i0,a,20i3)') '# alpha ', ia, ':', alpha_list(:,ia)
  end do

  open(10,file=trim(pfile),status='old')
  np = 0
  do
     read(10,*,iostat=ios) row
     if ( ios /= 0 ) exit
     x(1:d0) = row(1:d0)
     call net_eval_hod( nt, wk, x, t )
     write(*,'(2000es24.15)') x(1:d0), t(1:nt%tab%na)
     np = np + 1
  end do
  close(10)
  write(*,'(a,i0,a)') '# ', np, ' points'

  call work_free(wk); call net_free(nt); call tabset_free(ts)
end program hod_dump
