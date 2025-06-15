!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (gen_pinn_data.f90) is part of DNNF90.
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
! Generate PINN benchmark data: exact-soliton data points (IC/BC) + collocation.
!
! Benchmark grid (all amplitude-normalised, u_hat = u/A).
!
! The coefficients below are rounded for readability; the input files of
! bench/ carry twelve digits, which is what the runs solve.  Substituting
! each soliton into its own equation gives, at fifty digit working
! precision, a residual below 1e-50 for the two-variable modes and below
! 1.4e-13 for the four-variable ones, the latter being the rounding of
! the twelve-digit DXLAP coefficients and not an error of the solution:
! solving for the coefficients the soliton requires reproduces the
! twelve-digit values and drives the residual to 1e-60.
!
! For the four-variable modes xi = x + y/2 + z/3, so the Laplacian acts
! as (1 + 1/4 + 1/9) d2/dxi2 = (49/36) d2/dxi2 and every dx Lap^j term
! reduces to (49/36)^j d^(2j+1)/dxi^(2j+1).
!
!  mode      D0  ord  equation (coords: x[,y,z],t; x=1st, t=last)
!  kdv        2   3   u_t + 3 u u_x + u_3x = 0
!                     u = sech^2(0.5(x-t)),            x in [-8,8],  t in [0,2]
!  kawahara   2   5   u_t + 13.608 u u_x + 4.68 u_3x - u_5x = 0
!                     u = sech^4(0.3(x-4.6656 t)),     x in [-8,8],  t in [0,1]
!  g7         2   7   u_t + 7.577955 u u_x + 6.2289 u_3x - 4.5 u_5x + u_7x = 0
!                     u = sech^6(0.15(x-2.6244 t)),    x in [-10,10],t in [0,1]
!  zk3        4   3   u_t + 3 u u_x + (36/49) dxLap u = 0
!                     u = sech^2(0.5(xi - t)),  xi = x+y/2+z/3
!  zk5        4   5   u_t + 13.608 u u_x + 3.43836735 dxLap u
!                         - 0.53977509 dxLap^2 u = 0
!                     u = sech^4(0.3(xi - 4.6656 t))
!  zk7        4   7   u_t + 7.577955 u u_x + 4.57633469 dxLap u
!                         - 2.42898792 dxLap^2 u + 0.39656946 dxLap^3 u = 0
!                     u = sech^6(0.15(xi - 2.6244 t))
!  (4-var domains: x,y,z in [-3,3], t in [0,1])
!
! outputs: data.dat   "x(1:D0) u_exact" on the t=0 face (60%) + spatial
!                     boundary faces (40%)                       (MATH set)
!          colloc.dat "x(1:D0) u_exact" interior points          (PINN set;
!                     exact column used only for error reporting)
! usage: ./gen_pinn.out MODE Ndata Ncolloc [seed]
!
PROGRAM gen_pinn_data
  implicit none
  character(20) :: mode, arg
  integer :: D0, Ndata, Ncol, i, nsz, iface, iseed0
  integer,allocatable :: iseed(:)
  real(8) :: x(10), r(10), u, kk, cc, xlo, xhi
  integer :: ppow
  integer,parameter :: ud=21, uc=22

  call get_command_argument(1,mode)
  call get_command_argument(2,arg); read(arg,*) Ndata
  call get_command_argument(3,arg); read(arg,*) Ncol
  iseed0 = 20260728
  if ( command_argument_count() >= 4 ) then
     call get_command_argument(4,arg); read(arg,*) iseed0
  end if

  select case ( trim(mode) )
  case ('kdv');      D0=2; kk=0.5d0;  cc=1.d0;      ppow=2; xlo=-8.d0;  xhi=8.d0
  case ('kawahara'); D0=2; kk=0.3d0;  cc=4.6656d0;  ppow=4; xlo=-8.d0;  xhi=8.d0
  case ('g7');       D0=2; kk=0.15d0; cc=2.6244d0;  ppow=6; xlo=-10.d0; xhi=10.d0
  case ('zk3');      D0=4; kk=0.5d0;  cc=1.d0;      ppow=2; xlo=-3.d0;  xhi=3.d0
  case ('zk5');      D0=4; kk=0.3d0;  cc=4.6656d0;  ppow=4; xlo=-3.d0;  xhi=3.d0
  case ('zk7');      D0=4; kk=0.15d0; cc=2.6244d0;  ppow=6; xlo=-3.d0;  xhi=3.d0
  case ('eyu10d');   D0=10; kk=0.d0;  cc=0.d0;      ppow=0; xlo=0.d0;   xhi=1.d0
  case ('slit');     D0=2; kk=0.d0;   cc=0.d0;      ppow=0; xlo=-1.d0;  xhi=1.d0
  case default
     write(*,*) "gen_pinn_data: unknown mode ",trim(mode); stop
  end select

  call random_seed( size=nsz )
  allocate( iseed(nsz) ); iseed = iseed0
  call random_seed( put=iseed )

  open(ud,file='data.dat',status='replace')
  open(uc,file='colloc.dat',status='replace')

  if ( trim(mode)=='eyu10d' ) then
     ! Laplace equation on (0,1)^10 with u = sum_{k=1..5} x_{2k-1} x_{2k}
     ! on the boundary (E & Yu 2018, Eq.(15)); the exact solution is the
     ! same harmonic polynomial.
     do i=1,Ndata
        call random_number( r )
        x(1:10) = r(1:10)
        iface = 1 + mod(i,10)                    ! which coordinate is fixed
        x(iface) = merge( 0.d0, 1.d0, mod(i/10,2)==0 )
        u = x(1)*x(2)+x(3)*x(4)+x(5)*x(6)+x(7)*x(8)+x(9)*x(10)
        write(ud,'(11e22.12)') x(1:10), u
     end do
     do i=1,Ncol
        call random_number( r )
        x(1:10) = r(1:10)
        u = x(1)*x(2)+x(3)*x(4)+x(5)*x(6)+x(7)*x(8)+x(9)*x(10)
        write(uc,'(11e22.12)') x(1:10), u
     end do
     close(ud); close(uc)
     write(*,'(a,i0,a,i0)') "gen_pinn_data [eyu10d]: Ndata=",Ndata,"  Ncolloc=",Ncol
     stop
  end if

  if ( trim(mode)=='slit' ) then
     ! Laplace equation on (-1,1)^2 \ [0,1)x{0} with the exact solution
     ! u = r^{1/2} sin(theta/2)  (E & Yu 2018, corner singularity).
     ! data.dat: outer boundary (4 edges) + both slit faces (u=0 there)
     ! colloc.dat: interior points, a thin band around the slit excluded
     do i=1,Ndata
        call random_number( r )
        if ( i <= (8*Ndata)/10 ) then           ! outer boundary
           iface = 1 + mod(i,4)
           x(1) = -1.d0 + 2.d0*r(1)
           x(2) = -1.d0 + 2.d0*r(2)
           select case(iface)
           case(1); x(1)=-1.d0
           case(2); x(1)= 1.d0
           case(3); x(2)=-1.d0
           case(4); x(2)= 1.d0
           end select
        else                                    ! slit faces, u=0
           x(1) = r(1)
           x(2) = 0.d0
        end if
        write(ud,'(5e22.12)') x(1:2), u_slit( x(1), x(2) )
     end do
     i = 0
     do
        call random_number( r )
        x(1) = -1.d0 + 2.d0*r(1)
        x(2) = -1.d0 + 2.d0*r(2)
        if ( ( x(1) > -0.05d0 ).and.( abs(x(2)) < 0.05d0 ) ) cycle
        i = i + 1
        write(uc,'(5e22.12)') x(1:2), u_slit( x(1), x(2) )
        if ( i == Ncol ) exit
     end do
     close(ud); close(uc)
     write(*,'(a,i0,a,i0)') "gen_pinn_data [slit]: Ndata=",Ndata,"  Ncolloc=",Ncol
     stop
  end if

  do i=1,Ndata
     call random_number( r )
     x(1:D0-1) = xlo + (xhi-xlo)*r(1:D0-1)
     if ( D0==2 ) then
        x(2) = merge( 2.d0, 1.d0, trim(mode)=='kdv' )*r(2)
     else
        x(4) = r(4)
     end if
     if ( i <= (6*Ndata)/10 ) then
        x(D0) = 0.d0                                  ! initial face
     else if ( D0==2 ) then
        x(1) = merge( xlo, xhi, mod(i,2)==0 )          ! x boundaries
     else
        iface = 1 + mod(i,6)
        select case(iface)
        case(1); x(1)=xlo
        case(2); x(1)=xhi
        case(3); x(2)=xlo
        case(4); x(2)=xhi
        case(5); x(3)=xlo
        case(6); x(3)=xhi
        end select
     end if
     u = uexact( x )
     write(ud,'(5e22.12)') x(1:D0), u
  end do

  do i=1,Ncol
     call random_number( r )
     x(1:D0-1) = xlo + (xhi-xlo)*r(1:D0-1)
     if ( D0==2 ) then
        x(2) = merge( 2.d0, 1.d0, trim(mode)=='kdv' )*r(2)
     else
        x(4) = r(4)
     end if
     u = uexact( x )
     write(uc,'(5e22.12)') x(1:D0), u
  end do

  close(ud); close(uc)
  write(*,'(a,a,a,i0,a,i0)') "gen_pinn_data [",trim(mode),"]: Ndata=",Ndata,"  Ncolloc=",Ncol

CONTAINS

  REAL(8) FUNCTION u_slit( xx, yy )
    implicit none
    real(8),intent(IN) :: xx, yy
    real(8) :: rr, th
    real(8),parameter :: pi = 3.14159265358979323846d0
    rr = sqrt( xx*xx + yy*yy )
    th = atan2( yy, xx )
    if ( th < 0.d0 ) th = th + 2.d0*pi
    u_slit = sqrt(rr)*sin( 0.5d0*th )
  END FUNCTION u_slit

  REAL(8) FUNCTION uexact( xv )
    implicit none
    real(8),intent(IN) :: xv(4)
    real(8) :: xi
    if ( D0==2 ) then
       xi = xv(1) - cc*xv(2)
    else
       xi = xv(1)+0.5d0*xv(2)+xv(3)/3.d0 - cc*xv(4)
    end if
    uexact = 1.d0/cosh( kk*xi )**ppow
  END FUNCTION uexact

END PROGRAM gen_pinn_data
