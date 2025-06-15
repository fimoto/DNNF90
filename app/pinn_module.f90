! -----------------------------------------------------------------------
! This file (pinn_module.f90) is part of DNNF90.
! (MIT License; see LICENSE at the repository root.)
!
! Collocation policy: the residual of the user's differential operator
! and the loss seed it implies.
!
! Both routines are pure bookkeeping on one point's carried
! derivatives.  They read the forward result the caller obtained from
! the library and write the seed dL/dT_alpha the library's adjoint
! consumes, both as arguments, so no state is shared and a caller may
! run them concurrently on its own buffers.  The partial differential
! equation never enters the library itself.
! -----------------------------------------------------------------------
MODULE pinn_module
  use global_variables, only: pinn_nterm, pinn_nonlin, pinn_coeff, pinn_ind, &
       sys_nres, sys_nterm, sys_res, sys_cmp, sys_ind, sys_fac, &
       sys_coeff, sys_fac_ind, sys_nfac, sys_fcomp, sys_find, &
       sys_wres, sys_has_src, sys_src_coeff, &
       pinn_has_src, pinn_src_coeff
  implicit none
  PRIVATE
  PUBLIC :: calc_pinn_residual, set_pinn_seed
  PUBLIC :: calc_sys_residual, set_sys_seed

CONTAINS

  !> Residual R of the user's operator at one point, from that point's
  !! carried derivatives t(1:na) and the source value src at that point.
  !! t(1) is the value itself.  The source enters additively, so it does
  !! not appear in set_pinn_seed: dR/dT is independent of it.
  SUBROUTINE calc_pinn_residual( t, src, R )
    implicit none
    real(8),intent(IN) :: t(:)
    real(8),intent(IN) :: src
    real(8),intent(OUT) :: R
    integer :: k
    R = 0.d0
    if ( pinn_has_src ) R = pinn_src_coeff*src
    do k=1,pinn_nterm
       if ( pinn_nonlin(k) ) then
          R = R + pinn_coeff(k)*t(1)*t(pinn_ind(k))
       else
          R = R + pinn_coeff(k)*t(pinn_ind(k))
       end if
    end do
  END SUBROUTINE calc_pinn_residual

  !> Seed of the residual loss, dL/dT_alpha = fac * R * dR/dT_alpha.
  SUBROUTINE set_pinn_seed( t, R, fac, seed )
    implicit none
    real(8),intent(IN) :: t(:)
    real(8),intent(IN) :: R, fac
    real(8),intent(OUT) :: seed(:)
    integer :: k
    seed = 0.d0
    do k=1,pinn_nterm
       if ( pinn_nonlin(k) ) then
          seed(pinn_ind(k)) = seed(pinn_ind(k)) + fac*R*pinn_coeff(k)*t(1)
          seed(1) = seed(1) + fac*R*pinn_coeff(k)*t(pinn_ind(k))
       else
          seed(pinn_ind(k)) = seed(pinn_ind(k)) + fac*R*pinn_coeff(k)
       end if
    end do
  END SUBROUTINE set_pinn_seed

  !> The same for a system: several residuals over several field
  !! components, where a term may multiply one component by a derivative
  !! of another.
  !!
  !! A term of residual r is
  !!
  !!     coeff * T(jc, ia)                       (linear)
  !!     coeff * T(ic, 1) * T(jc, ia)            (cross term)
  !!
  !! with ic the component that appears as a factor and jc the one that
  !! is differentiated.  The scalar case is ic = jc = 1, so the routines
  !! above are its special case and are kept because every existing input
  !! goes through them unchanged.
  !!
  !! This is what a coupled system needs and the scalar form cannot
  !! express.  The momentum equation of an incompressible flow carries
  !! u du/dx + v du/dy, a component times the derivative of another; so
  !! does the transport of a charge density by a flow, u.grad(rho), and
  !! so does the body force rho E of an electrohydrodynamic problem.
  SUBROUTINE calc_sys_residual( tm, src, R )
    implicit none
    real(8),intent(IN) :: tm(:,:)          ! (component, multi-index)
    real(8),intent(IN) :: src(:)           ! one source per residual
    real(8),intent(OUT) :: R(:)            ! one value per residual
    integer :: k, ir, m
    real(8) :: prod
    R(1:sys_nres) = 0.d0
    do ir=1,sys_nres
       ! the source enters every residual with a minus sign, since the
       ! equation is written  operator(u) = S  and the residual is
       ! operator(u) - S
       R(ir) = -src(ir)
    end do
    do k=1,sys_nterm
       ir = sys_res(k)
       prod = sys_coeff(k)
       do m = 1, sys_nfac(k)
          prod = prod*tm( sys_fcomp(m,k), sys_find(m,k) )
       end do
       R(ir) = R(ir) + prod
    end do
  END SUBROUTINE calc_sys_residual

  !> Seed of the system residual loss.
  !!
  !! With L = (fac/2) sum_r R_r^2 the seed is
  !!
  !!     dL/dT(ic,ia) = fac * sum_r R_r * dR_r/dT(ic,ia),
  !!
  !! and a cross term contributes to two entries: the derivative slot of
  !! the component being differentiated, and the value slot of the
  !! component acting as a factor.  Getting that second contribution
  !! wrong is the easy mistake here, so the adjoint is checked against
  !! finite differences of the loss in tools/example_taylorgreen.f90.
  !! The residual weights of the loss are applied unless weighted is
  !! present and false.  A gradient rule wants them, since it descends
  !! the weighted objective.  The extended Kalman filter does not: its
  !! observation row is dR/dw for one residual, and a weight folded into
  !! it rescales the row against an innovation that was not rescaled,
  !! which corrupts the covariance rather than the step size.
  SUBROUTINE set_sys_seed( tm, R, fac, seedm, weighted )
    implicit none
    real(8),intent(IN) :: tm(:,:)
    real(8),intent(IN) :: R(:), fac
    real(8),intent(OUT) :: seedm(:,:)
    logical,intent(IN),optional :: weighted
    logical :: usew
    real(8) :: wr
    integer :: k, ir, m, mm
    real(8) :: other
    usew = .true.
    if ( present(weighted) ) usew = weighted
    seedm = 0.d0
    do k=1,sys_nterm
       ir = sys_res(k)
       wr = 1.d0
       if ( usew ) wr = sys_wres(ir)
       ! The product rule: factor m is seeded with the product of every
       ! other factor.  When two factors are the same entry, as in u_x
       ! times u_x, both contributions land there and add, which is the
       ! factor of two the rule asks for; accumulating rather than
       ! assigning is what makes that come out right.
       do m = 1, sys_nfac(k)
          ! sys_wres(ir) is the weight of this residual in the loss, so
          ! it multiplies the seed exactly as it multiplies the term of
          ! the objective it comes from
          other = fac*wr*R(ir)*sys_coeff(k)
          do mm = 1, sys_nfac(k)
             if ( mm == m ) cycle
             other = other*tm( sys_fcomp(mm,k), sys_find(mm,k) )
          end do
          seedm( sys_fcomp(m,k), sys_find(m,k) ) = &
               seedm( sys_fcomp(m,k), sys_find(m,k) ) + other
       end do
    end do
  END SUBROUTINE set_sys_seed

END MODULE pinn_module
