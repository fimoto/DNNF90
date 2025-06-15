!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (bessel_module.f90) is part of DNNF90 and is released under
!  the MIT License (file LICENSE in the root directory), EXCEPT for the
!  four routines jyna, jy01b, msta1, msta2 and envj below, which carry
!  their own terms:
!
!    Those routines are copyrighted by Shanjie Zhang and Jianming Jin,
!    from "Computation of Special Functions" (Wiley, 1996).  They give
!    permission to incorporate them into a user program provided that
!    the copyright is acknowledged.  This notice is that acknowledgement.
!    They have been adapted only in form: wrapped in a module, given the
!    module kind parameter, and with the external-function declarations
!    that a module makes unnecessary removed.  No arithmetic is changed.
!
! -----------------------------------------------------------------------
! Bessel functions of the first kind, J_0 .. J_n, for the BESSEL
! activation.
!
! Why J_0 is used as an activation at all: the engine carries sigma^(q)
! up to q = K+1, so an activation whose high derivatives grow makes the
! deep slots large and ill-conditioned.  Every derivative of J_0 is a
! fixed binomial combination of Bessel functions,
!
!   J_0^(q)(a) = ((-1)^q / 2^q) sum_(k=0..q) (-1)^k C(q,k) J_(q-2k)(a),
!
! and |J_n(a)| <= 1 for every order and every real a, so the whole
! derivative sequence is bounded by one.  Measured on [-20,20] the
! maxima fall with order: 1, 0.5, 0.375, 0.3125, 0.273 at q = 0,2,4,6,8,
! against 1220 for tanh and 1 for sin at q = 8.
!
! The routine of Zhang and Jin below assumes a nonnegative argument and
! silently returns the x = 0 values for a negative one, which for an
! activation would mean half of every layer receiving J_0 = 1.  The
! wrapper reflects instead, using J_n(-x) = (-1)^n J_n(x).
! -----------------------------------------------------------------------
MODULE bessel_module

  implicit none
  integer,parameter :: rk8 = selected_real_kind(15)

  PRIVATE
  PUBLIC :: besj_series

CONTAINS

  !> J_0(x) .. J_n(x) for any real x, in bj(0:n).
  SUBROUTINE besj_series( n, x, bj )
    implicit none
    integer,intent(IN) :: n
    real(rk8),intent(IN) :: x
    real(rk8),intent(OUT) :: bj(0:n)
    real(rk8) :: ax, dj(0:n), by(0:n), dy(0:n)
    integer :: nm, k

    ax = abs( x )
    call jyna( n, ax, nm, bj, dj, by, dy )
    if ( nm < n ) then
       ! the routine reports how far it could go; the rest would be
       ! garbage rather than small, so it is not left to chance
       do k = nm+1, n
          bj(k) = 0.0_rk8
       end do
    end if
    if ( x < 0.0_rk8 ) then
       do k = 1, n, 2
          bj(k) = -bj(k)
       end do
    end if
  END SUBROUTINE besj_series

subroutine jyna ( n, x, nm, bj, dj, by, dy )

!*****************************************************************************80
!
!! jyna() computes Bessel functions Jn(x) and Yn(x) and derivatives.
!
!  Licensing:
!
!    This routine is copyrighted by Shanjie Zhang and Jianming Jin.  However,
!    they give permission to incorporate this routine into a user program
!    provided that the copyright is acknowledged.
!
!  Modified:
!
!    29 April 2012
!
!  Author:
!
!    Original Fortran77 version by Shanjie Zhang, Jianming Jin.
!    This version by John Burkardt.
!
!  Reference:
!
!    Shanjie Zhang, Jianming Jin,
!    Computation of Special Functions,
!    Wiley, 1996,
!    ISBN: 0-471-11963-6,
!    LC: QA351.C45.
!
!  Input:
!
!    integer N, the order.
!
!    real ( kind = rk8 ) X, the argument.
!
!  Output:
!
!    integer NM, the highest order computed.
!
!    real ( kind = rk8 ) BJ(0:N), DJ(0:N), BY(0:N), DY(0:N), the values
!    of Jn(x), Jn'(x), Yn(x), Yn'(x).
!
  implicit none

  integer, parameter :: rk8 = kind ( 1.0D+00 )

  integer n

  real ( kind = rk8 ) bj(0:n)
  real ( kind = rk8 ) bj0
  real ( kind = rk8 ) bj1
  real ( kind = rk8 ) bjk
  real ( kind = rk8 ) by(0:n)
  real ( kind = rk8 ) by0
  real ( kind = rk8 ) by1
  real ( kind = rk8 ) cs
  real ( kind = rk8 ) dj(0:n)
  real ( kind = rk8 ) dj0
  real ( kind = rk8 ) dj1
  real ( kind = rk8 ) dy(0:n)
  real ( kind = rk8 ) dy0
  real ( kind = rk8 ) dy1
  real ( kind = rk8 ) f
  real ( kind = rk8 ) f0
  real ( kind = rk8 ) f1
  real ( kind = rk8 ) f2
  integer k
  integer m
  integer nm
  real ( kind = rk8 ) x

  nm = n

  if ( x < 1.0D-100 ) then

    do k = 0, n
      bj(k) = 0.0D+00
      dj(k) = 0.0D+00
      by(k) = -1.0D+300
      dy(k) = 1.0D+300
    end do
    bj(0) = 1.0D+00
    dj(1) = 0.5D+00
    return

  end if

  call jy01b ( x, bj0, dj0, bj1, dj1, by0, dy0, by1, dy1 )
  bj(0) = bj0
  bj(1) = bj1
  by(0) = by0
  by(1) = by1
  dj(0) = dj0
  dj(1) = dj1
  dy(0) = dy0
  dy(1) = dy1

  if ( n <= 1 ) then
    return
  end if

  if ( n < int ( 0.9D+00 * x) ) then

    do k = 2, n
      bjk = 2.0D+00 * ( k - 1.0D+00 ) / x * bj1 - bj0
      bj(k) = bjk
      bj0 = bj1
      bj1 = bjk
    end do

  else

    m = msta1 ( x, 200 )

    if ( m < n ) then
      nm = m
    else
      m = msta2 ( x, n, 15 )
    end if

    f2 = 0.0D+00
    f1 = 1.0D-100
    do k = m, 0, -1
      f = 2.0D+00 * ( k + 1.0D+00 ) / x * f1 - f2
      if ( k <= nm ) then
        bj(k) = f
      end if
      f2 = f1
      f1 = f
    end do

    if ( abs ( bj1 ) < abs ( bj0 ) ) then
      cs = bj0 / f
    else
      cs = bj1 / f2
    end if

    do k = 0, nm
      bj(k) = cs * bj(k)
    end do

  end if

  do k = 2, nm
    dj(k) = bj(k-1) - k / x * bj(k)
  end do

  f0 = by(0)
  f1 = by(1)
  do k = 2, nm
    f = 2.0D+00 * ( k - 1.0D+00 ) / x * f1 - f0
    by(k) = f
    f0 = f1
    f1 = f
  end do

  do k = 2, nm
    dy(k) = by(k-1) - k * by(k) / x
  end do

  return
end

subroutine jy01b ( x, bj0, dj0, bj1, dj1, by0, dy0, by1, dy1 )

!*****************************************************************************80
!
!! jy01b() computes Bessel functions J0(x), J1(x), Y0(x), Y1(x) and derivatives.
!
!  Licensing:
!
!    This routine is copyrighted by Shanjie Zhang and Jianming Jin.  However,
!    they give permission to incorporate this routine into a user program
!    provided that the copyright is acknowledged.
!
!  Modified:
!
!    02 August 2012
!
!  Author:
!
!    Original Fortran77 version by Shanjie Zhang, Jianming Jin.
!    This version by John Burkardt.
!
!  Reference:
!
!    Shanjie Zhang, Jianming Jin,
!    Computation of Special Functions,
!    Wiley, 1996,
!    ISBN: 0-471-11963-6,
!    LC: QA351.C45.
!
!  Input:
!
!    real ( kind = rk8 ) X, the argument.
!
!  Output:
!
!    real ( kind = rk8 ) BJ0, DJ0, BJ1, DJ1, BY0, DY0, BY1, DY1,
!    the values of J0(x), J0'(x), J1(x), J1'(x), Y0(x), Y0'(x), Y1(x), Y1'(x).
!
  implicit none

  integer, parameter :: rk8 = kind ( 1.0D+00 )

  real ( kind = rk8 ) a0
  real ( kind = rk8 ) bj0
  real ( kind = rk8 ) bj1
  real ( kind = rk8 ) by0
  real ( kind = rk8 ) by1
  real ( kind = rk8 ) dj0
  real ( kind = rk8 ) dj1
  real ( kind = rk8 ) dy0
  real ( kind = rk8 ) dy1
  real ( kind = rk8 ) p0
  real ( kind = rk8 ) p1
  real ( kind = rk8 ) pi
  real ( kind = rk8 ) q0
  real ( kind = rk8 ) q1
  real ( kind = rk8 ) t
  real ( kind = rk8 ) t2
  real ( kind = rk8 ) ta0
  real ( kind = rk8 ) ta1
  real ( kind = rk8 ) x

  pi = 3.141592653589793D+00

  if ( x == 0.0D+00 ) then

    bj0 = 1.0D+00
    bj1 = 0.0D+00
    dj0 = 0.0D+00
    dj1 = 0.5D+00
    by0 = -1.0D+300
    by1 = -1.0D+300
    dy0 = 1.0D+300
    dy1 = 1.0D+300
    return

  else if ( x <= 4.0D+00 ) then

    t = x / 4.0D+00
    t2 = t * t

    bj0 = (((((( &
      - 0.5014415D-03 * t2 &
      + 0.76771853D-02 ) * t2 &
      - 0.0709253492D+00 ) * t2 &
      + 0.4443584263D+00 ) * t2 &
      - 1.7777560599D+00 ) * t2 &
      + 3.9999973021D+00 ) * t2 &
      - 3.9999998721D+00 ) * t2 &
      + 1.0D+00

    bj1 = t * ((((((( &
      - 0.1289769D-03 * t2 &
      + 0.22069155D-02 ) * t2 &
      - 0.0236616773D+00 ) * t2 &
      + 0.1777582922D+00 ) * t2 &
      - 0.8888839649D+00 ) * t2 &
      + 2.6666660544D+00 ) * t2 &
      - 3.9999999710D+00 ) * t2 &
      + 1.9999999998D+00 )

    by0 = ((((((( &
      - 0.567433D-04 * t2 &
      + 0.859977D-03 ) * t2 &
      - 0.94855882D-02 ) * t2 &
      + 0.0772975809D+00 ) * t2 &
      - 0.4261737419D+00 ) * t2 &
      + 1.4216421221D+00 ) * t2 &
      - 2.3498519931D+00 ) * t2 &
      + 1.0766115157D+00 ) * t2 &
      + 0.3674669052D+00

    by0 = 2.0D+00 / pi * log ( x / 2.0D+00 ) * bj0 + by0

    by1 = (((((((( &
        0.6535773D-03 * t2 &
      - 0.0108175626D+00 ) * t2 &
      + 0.107657606D+00 ) * t2 &
      - 0.7268945577D+00 ) * t2 &
      + 3.1261399273D+00 ) * t2 &
      - 7.3980241381D+00 ) * t2 &
      + 6.8529236342D+00 ) * t2 &
      + 0.3932562018D+00 ) * t2 &
      - 0.6366197726D+00 ) / x

    by1 = 2.0D+00 / pi * log ( x / 2.0D+00 ) * bj1 + by1

  else

    t = 4.0D+00 / x
    t2 = t * t
    a0 = sqrt ( 2.0D+00 / ( pi * x ) )

    p0 = (((( &
      - 0.9285D-05 * t2 &
      + 0.43506D-04 ) * t2 &
      - 0.122226D-03 ) * t2 &
      + 0.434725D-03 ) * t2 &
      - 0.4394275D-02 ) * t2 &
      + 0.999999997D+00

    q0 = t * ((((( &
        0.8099D-05 * t2 &
      - 0.35614D-04 ) * t2 &
      + 0.85844D-04 ) * t2 &
      - 0.218024D-03 ) * t2 &
      + 0.1144106D-02 ) * t2 &
      - 0.031249995D+00 )

    ta0 = x - 0.25D+00 * pi
    bj0 = a0 * ( p0 * cos ( ta0 ) - q0 * sin ( ta0 ) )
    by0 = a0 * ( p0 * sin ( ta0 ) + q0 * cos ( ta0 ) )

    p1 = (((( &
        0.10632D-04 * t2 &
      - 0.50363D-04 ) * t2 &
      + 0.145575D-03 ) * t2 &
      - 0.559487D-03 ) * t2 &
      + 0.7323931D-02 ) * t2 &
      + 1.000000004D+00

    q1 = t * ((((( &
      - 0.9173D-05      * t2 &
      + 0.40658D-04 )   * t2 &
      - 0.99941D-04 )   * t2 &
      + 0.266891D-03 )  * t2 &
      - 0.1601836D-02 ) * t2 &
      + 0.093749994D+00 )

    ta1 = x - 0.75D+00 * pi
    bj1 = a0 * ( p1 * cos ( ta1 ) - q1 * sin ( ta1 ) )
    by1 = a0 * ( p1 * sin ( ta1 ) + q1 * cos ( ta1 ) )

  end if

  dj0 = - bj1
  dj1 = bj0 - bj1 / x
  dy0 = - by1
  dy1 = by0 - by1 / x

  return
end

function msta1 ( x, mp )

!*****************************************************************************80
!
!! msta1() determines a backward recurrence starting point for Jn(x).
!
!  Discussion:
!
!    This procedure determines the starting point for backward
!    recurrence such that the magnitude of
!    Jn(x) at that point is about 10^(-MP).
!
!  Licensing:
!
!    This routine is copyrighted by Shanjie Zhang and Jianming Jin.  However,
!    they give permission to incorporate this routine into a user program
!    provided that the copyright is acknowledged.
!
!  Modified:
!
!    08 July 2012
!
!  Author:
!
!    Original Fortran77 version by Shanjie Zhang, Jianming Jin.
!    This version by John Burkardt.
!
!  Reference:
!
!    Shanjie Zhang, Jianming Jin,
!    Computation of Special Functions,
!    Wiley, 1996,
!    ISBN: 0-471-11963-6,
!    LC: QA351.C45.
!
!  Input:
!
!    real ( kind = rk8 ) X, the argument.
!
!    integer MP, the negative logarithm of the
!    desired magnitude.
!
!  Output:
!
!    integer MSTA1, the starting point.
!
  implicit none
  integer msta1

  integer, parameter :: rk8 = kind ( 1.0D+00 )

  real ( kind = rk8 ) a0
  real ( kind = rk8 ) f
  real ( kind = rk8 ) f0
  real ( kind = rk8 ) f1
  integer it
  integer mp
  integer n0
  integer n1
  integer nn
  real ( kind = rk8 ) x

  a0 = abs ( x )
  n0 = int ( 1.1D+00 * a0 ) + 1
  f0 = envj ( n0, a0 ) - mp
  n1 = n0 + 5
  f1 = envj ( n1, a0 ) - mp
  do it = 1, 20
    nn = n1 - int ( real ( n1 - n0, kind = rk8 ) / ( 1.0D+00 - f0 / f1 ) )
    f = envj ( nn, a0 ) - mp
    if ( abs ( nn - n1 ) < 1 ) then
      exit
    end if
    n0 = n1
    f0 = f1
    n1 = nn
    f1 = f
  end do

  msta1 = nn

  return
end

function msta2 ( x, n, mp )

!*****************************************************************************80
!
!! msta2() determines a backward recurrence starting point for Jn(x).
!
!  Discussion:
!
!    This procedure determines the starting point for a backward
!    recurrence such that all Jn(x) has MP significant digits.
!
!    Jianming Jin supplied a modification to this code on 12 January 2016.
!
!  Licensing:
!
!    This routine is copyrighted by Shanjie Zhang and Jianming Jin.  However,
!    they give permission to incorporate this routine into a user program
!    provided that the copyright is acknowledged.
!
!  Modified:
!
!    14 January 2016
!
!  Author:
!
!    Original Fortran77 version by Shanjie Zhang, Jianming Jin.
!    This version by John Burkardt.
!
!  Reference:
!
!    Shanjie Zhang, Jianming Jin,
!    Computation of Special Functions,
!    Wiley, 1996,
!    ISBN: 0-471-11963-6,
!    LC: QA351.C45.
!
!  Input:
!
!    real ( kind = rk8 ) X, the argument of Jn(x).
!
!    integer N, the order of Jn(x).
!
!    integer MP, the number of significant digits.
!
!  Output:
!
!    integer MSTA2, the starting point.
!
  implicit none
  integer msta2

  integer, parameter :: rk8 = kind ( 1.0D+00 )

  real ( kind = rk8 ) a0
  real ( kind = rk8 ) ejn
  real ( kind = rk8 ) f
  real ( kind = rk8 ) f0
  real ( kind = rk8 ) f1
  real ( kind = rk8 ) hmp
  integer it
  integer mp
  integer n
  integer n0
  integer n1
  integer nn
  real ( kind = rk8 ) obj
  real ( kind = rk8 ) x

  a0 = abs ( x )
  hmp = 0.5D+00 * mp
  ejn = envj ( n, a0 )

  if ( ejn <= hmp ) then
    obj = mp
!
!  Original code:
!
!   n0 = int ( 1.1D+00 * a0 )
!
!  Updated code:
!
    n0 = int ( 1.1D+00 * a0 ) + 1
  else
    obj = hmp + ejn
    n0 = n
  end if

  f0 = envj ( n0, a0 ) - obj
  n1 = n0 + 5
  f1 = envj ( n1, a0 ) - obj

  do it = 1, 20
    nn = n1 - int ( real ( n1 - n0, kind = rk8 ) / ( 1.0D+00 - f0 / f1 ) )
    f = envj ( nn, a0 ) - obj
    if ( abs ( nn - n1 ) < 1 ) then
      exit
    end if
    n0 = n1
    f0 = f1
    n1 = nn
    f1 = f
  end do

  msta2 = nn + 10

  return
end

function envj ( n, x )

!*****************************************************************************80
!
!! envj() is a utility function used by MSTA1 and MSTA2.
!
!  Discussion:
!
!    ENVJ estimates -log(Jn(x)) from the estimate
!    Jn(x) approx 1/sqrt(2*pi*n) * ( e*x/(2*n))^n
!
!  Licensing:
!
!    This routine is copyrighted by Shanjie Zhang and Jianming Jin.  However,
!    they give permission to incorporate this routine into a user program
!    provided that the copyright is acknowledged.
!
!  Modified:
!
!    14 January 2016
!
!  Author:
!
!    Original Fortran77 version by Shanjie Zhang, Jianming Jin.
!    This version by John Burkardt.
!    Modifications suggested by Vincent Lafage, 11 January 2016.
!
!  Reference:
!
!    Shanjie Zhang, Jianming Jin,
!    Computation of Special Functions,
!    Wiley, 1996,
!    ISBN: 0-471-11963-6,
!    LC: QA351.C45.
!
!  Input:
!
!    integer N, the order of the Bessel function.
!
!    real ( kind = rk8 ) X, the absolute value of the argument.
!
!  Output:
!
!    real ( kind = rk8 ) ENVJ, the value.
!
  implicit none
  real ( kind = rk8 ) envj

  integer, parameter :: rk8 = kind ( 1.0D+00 )

  real ( kind = rk8 ) logten
  integer n
  real ( kind = rk8 ) n_r8
  real ( kind = rk8 ) r8_gamma_log
  real ( kind = rk8 ) x
!
!  Original code
!
  if ( .true. ) then

    envj = 0.5D+00 * log10 ( 6.28D+00 * n ) &
      - n * log10 ( 1.36D+00 * x / n )
!
!  Modification suggested by Vincent Lafage.
!
  else

    n_r8 = real ( n, kind = rk8 )
    logten = log ( 10.0D+00 )
    envj = r8_gamma_log ( n_r8 + 1.0D+00 ) / logten - n_r8 * log10 ( x )

  end if

  return
end

END MODULE bessel_module
