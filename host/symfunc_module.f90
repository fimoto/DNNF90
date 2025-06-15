!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (symfunc_module.f90) is part of DNNF90.
!  (MIT License; see LICENSE at the repository root.)
!
! -----------------------------------------------------------------------
! Minimal Behler-Parrinello radial symmetry functions with analytic first
! AND second derivatives with respect to the atomic coordinates.
!
!   G_k(i) = sum_{j/=i, r_ij<rc} exp( -eta_k (r_ij - rs_k)^2 ) fc(r_ij)
!   fc(r)  = ( cos(pi r/rc) + 1 ) / 2
!
! This is the functional form of n2p2's radial function (type 2 with the
! CT_COS cutoff), so parameters translate one to one to n2p2 input.nn
! lines "symfunction_short <e> 2 <e> eta rs rc".  The reason the
! functions are implemented here rather than taken from n2p2 is the
! second derivative: training a network on Hessians (force constants,
! phonons) needs d2G/dR2, which n2p2 does not provide because ordinary
! force fields never differentiate G twice.
!
! This module is one dimensional (a chain), which is all the
! demonstration needs; the three dimensional generalization changes the
! chain rule factors dr/dx but not the structure.
!
! Classification: this module is a HOST-SIDE REFERENCE, not part of the
! DNNF90 core.  The engine's boundary is the descriptor vector G: DNNF90
! owns N(G) and every partial derivative of N with respect to G, and the
! host owns G(R) and its derivatives.  This file exists only because no
! descriptor library provides d2G/dR2 yet; it is what n2p2 or any other
! descriptor code would supply in a production coupling, and it is
! therefore excluded from libdnnf90.
!
! Two properties matter to anyone adapting this file.
!
! The cutoff is the cosine one, whose second derivative does not vanish
! at rc: d2fc/dr2 tends to +0.5*(pi/rc)^2 there while fc and dfc tend to
! zero.  The descriptors are therefore C^1 but not C^2 across the cutoff,
! and a configuration with a pair near rc gives a discontinuous Hessian.
! The shipped demonstration is admissible only because it verifies that
! no pair comes near rc; a production coupling wants a C^2 cutoff, such
! as the polynomial ones of n2p2.
!
! The second derivative is returned as a dense (ng, na, na) array, so the
! cost and the storage are quadratic in the number of atoms.  That is
! adequate for a chain of six and not for a real cell: a production
! version would carry only the neighbour pairs inside the cutoff.
!
! No dependence on any other DNNF90 module.
! -----------------------------------------------------------------------
MODULE symfunc_module

  implicit none

  PRIVATE
  PUBLIC :: symset_t, symfunc_eval

  real(8),parameter :: pi = 3.14159265358979323846d0

  !> One set of radial functions shared by all atoms.
  TYPE :: symset_t
     integer :: ng = 0
     real(8),allocatable :: eta(:)
     real(8),allocatable :: rs(:)
     real(8) :: rc = 0.d0
  END TYPE symset_t

CONTAINS

  !> Descriptors of atom i in a chain x(1:na), with analytic derivatives.
  !!
  !!   G(1:ng)          the descriptor values
  !!   dG(1:ng,1:na)    dG_k / dx_p
  !!   d2G(1:ng,1:na,1:na)  d2G_k / dx_p dx_q
  !!
  !! Every pair term depends on x only through r = |x_i - x_j|, so with
  !! s = sign(x_i - x_j) the chain rule factors are dr/dx_i = s,
  !! dr/dx_j = -s and all second chain factors vanish (one dimension).
  SUBROUTINE symfunc_eval( ss, na, x, i, G, dG, d2G )
    implicit none
    type(symset_t),intent(IN) :: ss
    integer,intent(IN) :: na, i
    real(8),intent(IN) :: x(na)
    real(8),intent(OUT) :: G(ss%ng)
    real(8),intent(OUT) :: dG(ss%ng,na)
    real(8),intent(OUT) :: d2G(ss%ng,na,na)
    integer :: j, k
    real(8) :: r, s, e, fc, dfc, d2fc, u, g0, g1, g2

    G   = 0.d0
    dG  = 0.d0
    d2G = 0.d0

    do j=1,na
       if ( j == i ) cycle
       r = abs( x(i) - x(j) )
       if ( r >= ss%rc .or. r <= 1.d-12 ) cycle
       s = sign( 1.d0, x(i) - x(j) )
       u    = pi*r/ss%rc
       fc   = 0.5d0*( cos(u) + 1.d0 )
       dfc  = -0.5d0*(pi/ss%rc)*sin(u)
       d2fc = -0.5d0*(pi/ss%rc)**2*cos(u)
       do k=1,ss%ng
          e  = exp( -ss%eta(k)*( r - ss%rs(k) )**2 )
          ! g(r) = e*fc and its two radial derivatives
          g0 = e*fc
          g1 = e*( dfc - 2.d0*ss%eta(k)*(r-ss%rs(k))*fc )
          g2 = e*( d2fc - 4.d0*ss%eta(k)*(r-ss%rs(k))*dfc &
                 + ( 4.d0*ss%eta(k)**2*(r-ss%rs(k))**2 - 2.d0*ss%eta(k) )*fc )
          G(k) = G(k) + g0
          ! dr/dx_i = s, dr/dx_j = -s
          dG(k,i) = dG(k,i) + g1*s
          dG(k,j) = dG(k,j) - g1*s
          ! d2r/dx dx = 0 in one dimension, so only g2 terms appear
          d2G(k,i,i) = d2G(k,i,i) + g2
          d2G(k,j,j) = d2G(k,j,j) + g2
          d2G(k,i,j) = d2G(k,i,j) - g2
          d2G(k,j,i) = d2G(k,j,i) - g2
       end do
    end do

  END SUBROUTINE symfunc_eval

END MODULE symfunc_module
