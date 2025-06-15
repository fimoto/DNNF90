!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (bench_roofline.f90) is part of DNNF90.
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
! The exact GEMM chain of one full-batch epoch (forward + delta backprop
! + gradient blocks) at width 768, depth 7, batch 190: the f64 floor.
program roofline
  implicit none
  integer,parameter :: W=768, B=190, NH=5
  real(8),allocatable :: wl(:,:), Z(:,:), A(:,:), D(:,:), G(:,:)
  integer :: it, l, ic0, ic1, cr, nrep
  real(8) :: t, fl
  external :: dgemm
  allocate( wl(W,0:W), Z(0:W,B), A(W,B), D(W,B), G(W,0:W) )
  call random_number(wl); call random_number(Z); Z(0,:)=1.d0
  call random_number(A); call random_number(D)
  nrep = 30
  call system_clock(ic0,cr)
  do it=1,nrep
     do l=1,NH                                     ! forward
        call dgemm("N","N", W, B, W+1, 1.d0, wl, W, Z, W+1, 0.d0, A, W)
     end do
     do l=1,NH-1                                   ! delta backprop
        call dgemm("T","N", W, B, W, 1.d0, wl(1,1), W, D, W, 0.d0, A, W)
     end do
     do l=1,NH                                     ! gradient blocks
        call dgemm("N","T", W, W+1, B, 1.d0, D, W, Z, W+1, 0.d0, G, W)
     end do
  end do
  call system_clock(ic1)
  t = dble(ic1-ic0)/cr/nrep
  fl = 2.d0*B*( dble(W)*(W+1)*NH + dble(W)*W*(NH-1) + dble(W)*(W+1)*NH )
  write(*,"(a,f8.2,a,f7.1,a)") "GEMM-chain floor: ", t*1e3, " ms/epoch   ", fl/t/1e9, " GF/s"
  write(*,"(a)") "(the real net's first hidden layer has k=2, so the true"
  write(*,"(a)") " floor is about 6/7 of the number above)"
end program
