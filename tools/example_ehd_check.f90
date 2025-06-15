! Does the electrohydrodynamic term table mean what it should?
!
! The case zwork/pinn_ehd is built on a manufactured solution: the five
! fields are chosen first and the sources follow, so the residual of the
! exact fields must equal zero once the sources are subtracted.  Feeding
! those fields through the same machinery the trainer uses is the only
! way to know that the twenty-two lines of the System block were
! transcribed correctly; a sign or a component index out of place would
! otherwise show up only as a fit that does not converge.
!
!   phi = sin x sin y      rho = cos x cos y + 1/2
!   u   = -cos x sin y     v   = sin x cos y
!   p   = -(cos 2x + cos 2y)/4
program verify_ehd
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, alpha_index
  use global_variables, only: sys_nres, sys_nterm, sys_res, sys_cmp, &
       sys_ind, sys_fac, sys_fac_ind, sys_coeff, sys_has_src, &
       sys_nfac, sys_fcomp, sys_find
  use pinn_module, only: calc_sys_residual
  implicit none
  integer,parameter :: D0 = 2, KMAX = 2, NC = 5
  real(8),parameter :: EPSC = 1.d0, MU = 0.5d0, DIF = 0.1d0, &
       NU = 0.1d0, CF = 0.2d0
  real(8),allocatable :: tm(:,:)
  real(8) :: x, y, R(NC), src(NC), emax
  integer :: sd(D0,1), ix, iy, ixx, iyy, k

  sd = 0
  call init_hod_tables( D0, KMAX, 0, sd )
  allocate( tm(NC,NUM_alpha) )
  ix  = alpha_index( (/ 1,0 /) );  iy  = alpha_index( (/ 0,1 /) )
  ixx = alpha_index( (/ 2,0 /) );  iyy = alpha_index( (/ 0,2 /) )

  ! the term table, transcribed from zwork/pinn_ehd/input_nn.dat
  sys_nres = 5;  sys_nterm = 0;  sys_has_src = .false.
  call add( 1, 1, ixx, 0,  1.d0 )
  call add( 1, 1, iyy, 0,  1.d0 )
  call add( 1, 2, 1  , 0,  1.d0/EPSC )
  call add( 2, 2, ix , 3,  1.d0 )
  call add( 2, 2, iy , 4,  1.d0 )
  ! -mu div(rho grad phi), expanded:
  !   grad(rho).grad(phi) + rho lap(phi)
  ! The first two are products of two first derivatives; the term table
  ! carries the slot of each factor, so they are expressible.
  call add2( 2, 2, ix , 1, ix , -MU )     ! rho_x phi_x
  call add2( 2, 2, iy , 1, iy , -MU )     ! rho_y phi_y
  call add2( 2, 2, 1  , 1, ixx, -MU )     ! rho phi_xx
  call add2( 2, 2, 1  , 1, iyy, -MU )     ! rho phi_yy
  call add( 2, 2, ixx, 0, -DIF )
  call add( 2, 2, iyy, 0, -DIF )
  call add( 3, 3, ix , 3,  1.d0 )
  call add( 3, 3, iy , 4,  1.d0 )
  call add( 3, 5, ix , 0,  1.d0 )
  call add( 3, 3, ixx, 0, -NU  )
  call add( 3, 3, iyy, 0, -NU  )
  call add( 3, 1, ix , 2,  CF  )
  call add( 4, 4, ix , 3,  1.d0 )
  call add( 4, 4, iy , 4,  1.d0 )
  call add( 4, 5, iy , 0,  1.d0 )
  call add( 4, 4, ixx, 0, -NU  )
  call add( 4, 4, iyy, 0, -NU  )
  call add( 4, 1, iy , 2,  CF  )
  call add( 5, 3, ix , 0,  1.d0 )
  call add( 5, 4, iy , 0,  1.d0 )

  emax = 0.d0
  call check(  0.7d0,  1.3d0, emax )
  call check( -1.4d0,  2.1d0, emax )
  call check(  1.9d0, -0.8d0, emax )
  call check(  0.2d0,  0.5d0, emax )

  write(*,'(a,e13.5)') " residual of the manufactured solution, max = ", emax
  if ( emax > 1.d-10 ) then
     write(*,'(a)') " FAILED: the term table does not reproduce the equations"
     stop 1
  end if
  write(*,'(a)') " passed"
  deallocate( tm )

CONTAINS

  SUBROUTINE add( ir, ic, ia, ifac, c )
    implicit none
    integer,intent(IN) :: ir, ic, ia, ifac
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir;  sys_cmp(sys_nterm) = ic
    sys_ind(sys_nterm) = ia;  sys_fac(sys_nterm) = ifac
    sys_fac_ind(sys_nterm) = 1
    sys_coeff(sys_nterm) = c
    call setfac( sys_nterm )
  END SUBROUTINE add

  !> A term c * d^beta u_ifac * d^alpha u_ic, the general cross form.
  SUBROUTINE add2( ir, ifac, ibeta, ic, ia, c )
    implicit none
    integer,intent(IN) :: ir, ifac, ibeta, ic, ia
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir;   sys_cmp(sys_nterm) = ic
    sys_ind(sys_nterm) = ia;   sys_fac(sys_nterm) = ifac
    sys_fac_ind(sys_nterm) = ibeta
    sys_coeff(sys_nterm) = c
    call setfac( sys_nterm )
  END SUBROUTINE add2

  !> mirror of what the input parser does: fill the factor list
  SUBROUTINE setfac( k )
    implicit none
    integer,intent(IN) :: k
    sys_fcomp(1,k) = sys_cmp(k);  sys_find(1,k) = sys_ind(k)
    sys_nfac(k) = 1
    if ( sys_fac(k) > 0 ) then
       sys_nfac(k) = 2
       sys_fcomp(2,k) = sys_fac(k);  sys_find(2,k) = sys_fac_ind(k)
    end if
  END SUBROUTINE setfac

  SUBROUTINE check( xa, ya, em )
    implicit none
    real(8),intent(IN) :: xa, ya
    real(8),intent(INOUT) :: em
    real(8) :: s, c, S2, C2
    s = sin(xa);  c = cos(xa);  S2 = sin(ya);  C2 = cos(ya)
    tm = 0.d0
    ! phi
    tm(1,1)   =  s*S2
    tm(1,ix)  =  c*S2
    tm(1,iy)  =  s*C2
    tm(1,ixx) = -s*S2
    tm(1,iyy) = -s*S2
    ! rho = cos x cos y + 1/2   (the field the shipped data carries)
    tm(2,1)   =  c*C2 + 0.5d0
    tm(2,ix)  = -s*C2
    tm(2,iy)  = -c*S2
    tm(2,ixx) = -c*C2
    tm(2,iyy) = -c*C2
    ! u = -cos x sin y
    tm(3,1)   = -c*S2
    tm(3,ix)  =  s*S2
    tm(3,iy)  = -c*C2
    tm(3,ixx) =  c*S2
    tm(3,iyy) =  c*S2
    ! v = sin x cos y
    tm(4,1)   =  s*C2
    tm(4,ix)  =  c*C2
    tm(4,iy)  = -s*S2
    tm(4,ixx) = -s*C2
    tm(4,iyy) = -s*C2
    ! p
    tm(5,1)   = -( cos(2.d0*xa) + cos(2.d0*ya) )/4.d0
    tm(5,ix)  =  sin(2.d0*xa)/2.d0
    tm(5,iy)  =  sin(2.d0*ya)/2.d0

    call sources( xa, ya, src )
    call calc_sys_residual( tm, src, R )
    write(*,'(a,5e12.4)') "   residuals:", R(1:NC)
    do k = 1, NC
       em = max( em, abs(R(k)) )
    end do
  END SUBROUTINE check

  !> the same sources bench/post/gen_ehd_data.py writes
  SUBROUTINE sources( xa, ya, so )
    implicit none
    real(8),intent(IN) :: xa, ya
    real(8),intent(OUT) :: so(NC)
    real(8) :: s, c, S2, C2, rho, rx, ry, rxx, ryy
    real(8) :: px, py, pxx, pyy, uu, vv, ux, uy, vx, vy
    real(8) :: uxx, uyy, vxx, vyy, ppx, ppy, drgp
    s = sin(xa);  c = cos(xa);  S2 = sin(ya);  C2 = cos(ya)
    px = c*S2;  py = s*C2;  pxx = -s*S2;  pyy = -s*S2
    rho = c*C2 + 0.5d0
    rx = -s*C2;  ry = -c*S2
    rxx = -c*C2;  ryy = -c*C2
    uu = -c*S2;  vv = s*C2
    ux = s*S2;  uy = -c*C2;  vx = c*C2;  vy = -s*S2
    uxx = c*S2;  uyy = c*S2;  vxx = -s*C2;  vyy = -s*C2
    ppx = sin(2.d0*xa)/2.d0;  ppy = sin(2.d0*ya)/2.d0
    drgp = rx*px + rho*pxx + ry*py + rho*pyy
    ! rho is no longer -eps lap(phi), so Poisson carries a source too
    so(1) = ( pxx + pyy ) + rho/EPSC
    so(2) = uu*rx + vv*ry - MU*drgp - DIF*( rxx + ryy )
    so(3) = uu*ux + vv*uy + ppx - NU*( uxx + uyy ) + CF*rho*px
    so(4) = uu*vx + vv*vy + ppy - NU*( vxx + vyy ) + CF*rho*py
    so(5) = 0.d0
  END SUBROUTINE sources

end program verify_ehd
