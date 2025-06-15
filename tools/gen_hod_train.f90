!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (gen_hod_train.f90) is part of DNNF90.
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
! Generate MATH_HOD training data with exactly known derivatives of all orders.
!
!   u(x) = prod_{i=1..D0} sin( omega_i x_i + phi_i )
!   d^alpha u = prod_i omega_i^{alpha_i} sin( omega_i x_i + phi_i + alpha_i*pi/2 )
!
! Points are uniform in [-1,1]^D0 with a fixed RNG seed (reproducible).
! Output columns: x(1:D0), then y_alpha for every multi-index |alpha|<=K in the
! canonical order of multi_index_bell_module (also written to
! hod_alpha_order.dat), so the file matches what the MATH_HOD reader expects.
!
! Usage:  ./gen_hod.out D0 K Npoints outfile [alpha_seed_file]
!   If alpha_seed_file is given (format: "n" then n lines of D0 integers), the
!   columns are restricted to the downward closure of those seeds, matching a
!   MATH_HOD run with the same Hod_alpha_file.
!
PROGRAM gen_hod_train

  use multi_index_bell_module

  implicit none
  integer :: D0, K, Npt
  character(120) :: outfile, arg
  integer :: n, ia, iv, iseed_size
  integer,allocatable :: iseed(:)
  integer :: nseed
  integer,allocatable :: seeds(:,:)
  character(120) :: seedfile
  real(8),allocatable :: x(:), y(:)
  real(8),parameter :: pi = 3.14159265358979323846d0
  real(8) :: omega(8), phi(8), val
  integer,parameter :: uo=21

  call get_command_argument(1,arg); read(arg,*) D0
  call get_command_argument(2,arg); read(arg,*) K
  call get_command_argument(3,arg); read(arg,*) Npt
  call get_command_argument(4,outfile)
  if ( D0<1 .or. D0>8 ) stop "gen_hod_train: D0 out of range"

  ! frequencies kept <= ~1.1 so that 7th derivatives stay O(1)
  omega(1:8) = (/ 0.9d0, 0.7d0, 1.1d0, 0.5d0, 0.8d0, 0.6d0, 1.0d0, 0.4d0 /)
  phi(1:8)   = (/ 0.3d0,-0.5d0, 0.9d0, 0.1d0,-0.7d0, 0.4d0,-0.2d0, 0.6d0 /)

  nseed = 0
  if ( command_argument_count() >= 5 ) then
     call get_command_argument(5,seedfile)
     open(uo,file=trim(seedfile),status='old')
     read(uo,*) nseed
     allocate( seeds(D0,nseed) )
     do n=1,nseed
        read(uo,*) seeds(1:D0,n)
     end do
     close(uo)
  else
     allocate( seeds(D0,1) ); seeds=0
  end if
  call init_hod_tables( D0, K, nseed, seeds )
  call write_alpha_order( 'hod_alpha_order.dat' )

  call random_seed( size=iseed_size )
  allocate( iseed(iseed_size) )
  iseed = 20260728
  call random_seed( put=iseed )

  allocate( x(D0), y(NUM_alpha) )

  open(uo,file=trim(outfile),status='replace')
  do n=1,Npt
     call random_number( x )
     x = 2.d0*x - 1.d0
     do ia=1,NUM_alpha
        val = 1.d0
        do iv=1,D0
           val = val * ( omega(iv)**alpha_list(iv,ia) ) * &
                sin( omega(iv)*x(iv) + phi(iv) + alpha_list(iv,ia)*pi/2.d0 )
        end do
        y(ia) = val
     end do
     write(uo,'(2000e22.12)') x(1:D0), y(1:NUM_alpha)
  end do
  close(uo)

  write(*,'(a,i0,a,i0,a,i0,a)') "gen_hod_train: D0=",D0,"  K=",K,"  N=",Npt, &
       "  columns = D0 + NUM_alpha"
  write(*,'(a,i0)') "NUM_alpha = ", NUM_alpha
  write(*,'(a,a)') "written: ", trim(outfile)

END PROGRAM gen_hod_train
