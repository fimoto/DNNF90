!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (gen_c_ref.f90) is part of DNNF90.
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
! tools/gen_c_ref.f90 - writes c_ref.dat: value and derivatives from the
! Fortran instance path, for the bitwise cross-language check of
! tools/example_c.c.  Run in a trained benchmark directory.
! (MIT License; see LICENSE at the repository root.)
PROGRAM gen_c_ref
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha
  use net_module
  implicit none
  integer,parameter :: NPT = 25
  integer :: seeds(2,3), n, ip, ur
  type(net_t)  :: nt
  type(work_t) :: wk
  real(8) :: x(2), t(64)
  seeds(:,1) = (/0,1/); seeds(:,2) = (/1,0/); seeds(:,3) = (/3,0/)
  call init_hod_tables( 2, 3, 3, seeds )
  n = NUM_alpha
  call net_load( nt, 'nn_weight.dat' )
  call work_init( wk, nt )
  ur = 91
  open( ur, file='c_ref.dat', status='replace' )
  write(ur,'(i0)') NPT
  do ip=1,NPT
     x(1) = -3.d0 + 6.d0*dble(ip-1)/dble(NPT-1)
     x(2) = 0.1d0*dble(mod(ip,10))
     call net_eval_hod( nt, wk, x, t )
     write(ur,'(100e26.17)') x(1:2), t(1:n)
  end do
  close(ur)
  call work_free( wk )
  call net_free( nt )
  write(*,'(a,i0,a)') "c_ref.dat written (", NPT, " points)"
END PROGRAM gen_c_ref
