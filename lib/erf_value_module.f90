! -----------------------------------------------------------------------
! This file (erf_value_module.f90) is part of DNNF90.
! (MIT License; see LICENSE at the repository root.)
!
! The value of the error function, isolated here because ERF is a
! Fortran 2008 intrinsic while the rest of the distribution conforms to
! Fortran 95.  Keeping the single non-conforming call in a file of its
! own lets make f95check continue to cover everything else, in the same
! way the C interoperability module is excepted.
!
! Only sigma^(0) of the ERF activation needs it: every higher derivative
! is a Hermite function built from exp alone, which is Fortran 95.
!
! An own implementation was tried and removed.  Its continued fraction
! lost eleven digits just above the switch point of its two branches
! (8.5e-5 absolute at x = 2, against 1e-16 for the series branch below
! it), which is far above the tolerances the engine is verified to, and
! it was about three times slower than the intrinsic besides.
! -----------------------------------------------------------------------
MODULE erf_value_module
  implicit none
  PRIVATE
  PUBLIC :: erf_value
CONTAINS
  REAL(8) FUNCTION erf_value( x )
    implicit none
    real(8),intent(IN) :: x
    erf_value = erf( x )
  END FUNCTION erf_value
END MODULE erf_value_module
