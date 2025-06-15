!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (example_embed.f90) is part of DNNF90.
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
! Embedding example: evaluates a trained PINN and its carried derivatives
! at a few points, the way a host code (for example a DFT package) would.
! Run it in a benchmark directory after training, with "Restart 1" set in
! input_nn.dat, for example:
!
!     cd bench/kdv
!     sed -i 's/^Restart      0 /Restart      1 /' input_nn.dat
!     ../../build/embed_example.out
!
! For the KdV case it prints u, u_x, u_t, u_xxx and the residual
! u_t + 3 u u_x + u_xxx at three points, which should be small after
! training.
PROGRAM example_embed
  use api_module
  implicit none
  integer,parameter :: mA = 4096
  real(8) :: x(2), t(mA)
  integer :: a(2), n, ia, i_u, i_ux, i_ut, i_u3
  integer :: ip
  real(8) :: res

  call dnnf90_init
  n = dnnf90_nderiv()
  write(*,'(a,i0)') "carried derivative slots: ", n

  ! locate the slots of u, u_x, u_t, u_xxx in the canonical order
  i_u=0; i_ux=0; i_ut=0; i_u3=0
  do ia=1,n
     call dnnf90_alpha( ia, a )
     if ( a(1)==0 .and. a(2)==0 ) i_u  = ia
     if ( a(1)==1 .and. a(2)==0 ) i_ux = ia
     if ( a(1)==0 .and. a(2)==1 ) i_ut = ia
     if ( a(1)==3 .and. a(2)==0 ) i_u3 = ia
  end do

  write(*,'(a)') "    x      t          u          u_x        u_t        u_xxx      residual"
  do ip=1,3
     x(1) = -2.d0 + 2.d0*dble(ip-1)   ! x = -2, 0, 2
     x(2) = 0.5d0                     ! t = 0.5
     call dnnf90_eval_hod( x, t )
     res = t(i_ut) + 3.d0*t(i_u)*t(i_ux) + t(i_u3)
     write(*,'(2f7.2,5e11.3)') x(1), x(2), t(i_u), t(i_ux), t(i_ut), t(i_u3), res
  end do

END PROGRAM example_embed
