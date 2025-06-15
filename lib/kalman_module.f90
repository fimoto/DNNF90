!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (kalman_module.f90) is part of DNNF90.
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
! Per-pattern global extended Kalman filter, the training method of the
! n2p2 tool chain.  One update presents one scalar observable (an energy,
! or one force component chained by the host into the descriptor space)
! and applies the rank-1 form, so no matrix is ever inverted:
!
!   xi    = target - prediction
!   Pj    = P j             (j = d prediction / d weights)
!   K     = Pj / ( lambda + j^T Pj )
!   w    <- w + K xi
!   P    <- ( P - K (Pj)^T ) / lambda
!   lambda <- lambda0*lambda + 1 - lambda0     (forgetting schedule)
!
! P is nw x nw, which is the intended regime of Kalman training: small
! networks updated very few times per pattern (the usual trade against
! SGD/Adam).  The Jacobian row comes from net_grad_point with a unit
! seed on the observed slot, so any carried derivative can be observed,
! including Hessian slots.
MODULE kalman_module

  use net_module
  use train_module

  implicit none

  PRIVATE
  PUBLIC :: kalman_t, kf_init, kf_free, kf_update, kf_update_obs, kf_num_weights
  PUBLIC :: kf_update_resid, kf_update_resid_multi, kf_update_grad
  PUBLIC :: kf_iekf_begin, kf_iekf_iter

  TYPE :: kalman_t
     integer :: nw = 0
     real(8) :: lam = 0.d0, lam0 = 0.d0
     !> Innovation gate (robust / adaptive-R EKF).  0 = off.  When on,
     !! an observation whose normalized innovation would exceed the gate,
     !!   xi^2 > gate^2 ( lambda + j^T P j ),
     !! has its effective observation noise inflated (both the row and
     !! the innovation scaled by the same s < 1) until equality holds.
     !! That is the classical measurement-validation remedy for EKF
     !! divergence under large innovations: the linearization a rank-1
     !! EKF step trusts is local, and an innovation far outside the
     !! filter's own predicted spread says the step it implies is
     !! outside that neighbourhood.  Gating bounds every step at the
     !! cost of slower absorption of surprising observations; it never
     !! changes an update the gate does not trip.
     real(8) :: gate = 0.d0
     integer :: ngated = 0                 ! gated updates, for reporting
     !> Process-noise injection: after every update, q is added to the
     !! diagonal of P.  The continuous alternative to the forgetting
     !! factor: lambda < 1 multiplies the whole of P and never lets the
     !! gains decay, while q = 0 with lambda = 1 lets them decay to
     !! nothing (the RLS plateau).  A small q keeps the filter alive at
     !! a controlled level between the two.  0 = off.
     real(8) :: q = 0.d0
     !> Decoupled (node-wise) filter: P is block diagonal, one block per
     !! neuron over its fan-in + bias (Puskorius-Feldkamp NDEKF).  The
     !! blocks align exactly with the flatten order of kf_update_grad,
     !! the Kalman gain keeps the GLOBAL innovation denominator
     !! lambda + j^T P j, and only the covariance loses its cross-neuron
     !! entries.  The price is that lost coupling; the gain is memory
     !! and per-update work falling from nw^2 to sum of block^2 (about
     !! 60x on the shipped EHD network).
     logical :: decoupled = .false.
     integer :: nb = 0, maxb = 0
     integer,allocatable :: bsz(:)
     integer,allocatable :: boff(:)
     real(8),allocatable :: Pb(:,:,:)
     real(8),allocatable :: P(:,:)
     real(8),allocatable :: wprior(:) ! IEKF: the prior weights
     real(8),allocatable :: wcur(:)   ! IEKF: scratch, current weights
     real(8),allocatable :: jf(:)  ! scratch
     real(8),allocatable :: pj(:)
     real(8),allocatable :: kv(:)
  END TYPE kalman_t

CONTAINS

  INTEGER FUNCTION kf_num_weights( nt )
    implicit none
    type(net_t),intent(IN) :: nt
    integer :: l
    kf_num_weights = 0
    do l=2,nt%nlayer
       kf_num_weights = kf_num_weights + nt%ndim(l)*( nt%ndim(l-1) + 1 )
    end do
  END FUNCTION kf_num_weights

  !> p0: initial covariance scale (e.g. 100).  lam_ini < 1 and lam0 < 1
  !! give the usual forgetting schedule with lambda -> 1 (e.g. 0.98 and
  !! 0.9987).
  SUBROUTINE kf_init( kf, nt, p0, lam_ini, lam0, decoupled )
    implicit none
    type(kalman_t),intent(OUT) :: kf
    type(net_t),intent(IN) :: nt
    real(8),intent(IN) :: p0, lam_ini, lam0
    !> Node-wise block-diagonal covariance (see the type definition).
    !! An argument rather than a pre-set field because intent(OUT)
    !! resets the derived type to its defaults on entry.
    logical,intent(IN),optional :: decoupled
    integer :: i, j
    if ( present(decoupled) ) kf%decoupled = decoupled
    if ( p0 <= 0.d0 ) then
       write(*,*) "kf_init: p0 must be positive, got", p0
       stop
    end if
    if ( lam_ini <= 0.d0 .or. lam_ini > 1.d0 .or. &
         lam0    <= 0.d0 .or. lam0    > 1.d0 ) then
       write(*,*) "kf_init: forgetting parameters must lie in (0,1]:", &
                  " lam_ini =", lam_ini, "  lam0 =", lam0
       stop
    end if
    kf%nw = kf_num_weights( nt )
    ! The global filter carries a dense covariance, so it costs nw^2
    ! doubles and nw^2 work per observation.  That is the right tool for
    ! the small networks it is meant for and hopeless for a force-field
    ! sized one, where a decoupled (layer-wise) filter is required.  An
    ! unguarded allocate here failed with an allocator message that gave
    ! the user no idea which knob was wrong.
    if ( .not. kf%decoupled ) then
       if ( dble(kf%nw)**2*8.d0 > 4.d9 ) then
          write(*,*) "kf_init: the dense covariance needs", &
                     dble(kf%nw)**2*8.d0/2.d0**30, "GB for", kf%nw, "weights"
          write(*,*) "  the global filter does not scale to this network;", &
                     " use the decoupled filter or a smaller net"
          stop
       end if
       allocate( kf%P(kf%nw,kf%nw) );  kf%P = 0.d0
       do i=1,kf%nw
          kf%P(i,i) = p0
       end do
    else
       ! One block per neuron over (bias, fan-in), which is exactly one
       ! contiguous run of the flatten order (l, j, i-fastest).
       kf%nb = 0;  kf%maxb = 0
       do i=2,nt%nlayer
          kf%nb = kf%nb + nt%ndim(i)
          kf%maxb = max( kf%maxb, nt%ndim(i-1)+1 )
       end do
       allocate( kf%bsz(kf%nb), kf%boff(kf%nb) )
       call kf_block_map( nt, kf%bsz, kf%boff )
       allocate( kf%Pb(kf%maxb,kf%maxb,kf%nb) );  kf%Pb = 0.d0
       do i=1,kf%nb
          do j=1,kf%bsz(i)
             kf%Pb(j,j,i) = p0
          end do
       end do
    end if
    allocate( kf%jf(kf%nw), kf%pj(kf%nw), kf%kv(kf%nw) )
    kf%lam  = lam_ini
    kf%lam0 = lam0
  END SUBROUTINE kf_init

  !> Block sizes and flat offsets of the node-wise partition, in the
  !! order kf_update_grad flattens the weights.
  SUBROUTINE kf_block_map( nt, bsz, boff )
    implicit none
    type(net_t),intent(IN) :: nt
    integer,intent(OUT) :: bsz(:), boff(:)
    integer :: l, j, ib, m
    ib = 0;  m = 0
    do l=2,nt%nlayer
       do j=1,nt%ndim(l)
          ib = ib + 1
          bsz(ib)  = nt%ndim(l-1) + 1
          boff(ib) = m
          m = m + bsz(ib)
       end do
    end do
  END SUBROUTINE kf_block_map

  SUBROUTINE kf_free( kf )
    implicit none
    type(kalman_t),intent(INOUT) :: kf
    if ( allocated(  kf%P) )  deallocate( kf%P )
    if ( allocated(  kf%jf) ) deallocate( kf%jf, kf%pj, kf%kv )
    kf%nw = 0
      if ( allocated( kf%wprior) ) deallocate( kf%wprior )
    if ( allocated( kf%wcur) )   deallocate( kf%wcur )
    if ( allocated( kf%Pb) )   deallocate( kf%Pb )
    if ( allocated( kf%bsz) )  deallocate( kf%bsz )
    if ( allocated( kf%boff) ) deallocate( kf%boff )
  END SUBROUTINE kf_free

  !> Convenience form: the observable is one raw slot.
  SUBROUTINE kf_update( nt, tw, g, kf, x, islot, target )
    implicit none
    type(net_t),intent(INOUT) :: nt
    type(twork_t),intent(INOUT) :: tw
    type(grad_t),intent(INOUT) :: g
    type(kalman_t),intent(INOUT) :: kf
    real(8),intent(IN) :: x(:)
    integer,intent(IN) :: islot
    real(8),intent(IN) :: target
    real(8) :: seed(nt%tab%na)
    if ( islot < 1 .or. islot > nt%tab%na ) then
       write(*,*) "kf_update: islot", islot, " outside 1..", nt%tab%na
       stop
    end if
    seed = 0.d0
    seed(islot) = 1.d0
    call kf_update_obs( nt, tw, g, kf, x, seed, target )
  END SUBROUTINE kf_update

  !> General form: the observable is any functional of the carried
  !! derivative slots at input x.  The caller supplies seed = dy/dT,
  !! that is the gradient of the observable with respect to the slots
  !! (for a linear combination these are its coefficients; for a
  !! nonlinear observable, its linearization at the current weights,
  !! exactly the seeds every loss in this code base already builds).
  !! The predicted value is reconstructed as sum( seed * T ), which is
  !! exact for observables linear in the slots, and the Jacobian row is
  !! one seeded adjoint.  Any carried order can therefore be observed:
  !! forces are |alpha|=1 combinations, Hessian elements combine
  !! |alpha|<=2, and so on up to the K of the table set.
  !> Update from an observable the caller evaluates itself.
  !!
  !! kf_update_obs forms its prediction as the linear functional
  !! dot(seed,T), which is right whenever the observable is linear in the
  !! carried derivatives.  A collocation residual need not be: a term
  !! c*u*d^alpha u is quadratic, and the linear functional built from
  !! dR/dT would count it twice.  Here the caller supplies the residual
  !! value and the seed dR/dT separately, so any nonlinear operator can
  !! be observed.  The target of a residual is zero, hence xi = -resid.
  SUBROUTINE kf_update_resid( nt, tw, g, kf, x, seed, resid )
    implicit none
    type(net_t),intent(INOUT) :: nt
    type(twork_t),intent(INOUT) :: tw
    type(grad_t),intent(INOUT) :: g
    type(kalman_t),intent(INOUT) :: kf
    real(8),intent(IN) :: x(:)
    real(8),intent(IN) :: seed(:)
    real(8),intent(IN) :: resid
    ! The prediction is handed over explicitly, so the rank-1 algebra is
    ! shared with kf_update_obs and the point is propagated once: forming
    ! a target from dot(seed,T) would have required a forward pass here on
    ! top of the one net_grad_point does.
    call kf_update_obs( nt, tw, g, kf, x, seed, 0.d0, resid )
  END SUBROUTINE kf_update_resid

  !> Rank-one update from one residual of a system of field components.
  !!
  !! The filter presents one scalar observable at a time, which is what
  !! keeps the update rank one and free of any matrix inversion.  A
  !! system needs no joint update for that reason: each of its residuals
  !! is one observable, and calling this once per residual visits them in
  !! turn.  What differs from the scalar case is the seed, which carries
  !! dR/dT for every component, because a cross term of the residual
  !! touches two of them.
  !!
  !! The observation row is dR/dw, obtained by seeding the adjoint with
  !! dR/dT, which is the same construction the gradient path uses; the
  !! innovation is the residual itself, since the target is zero.
  !! An observation noise r is applied as scale = 1/sqrt(r).  That is
  !! exact rather than approximate: with the row and the innovation both
  !! scaled by s, the step becomes
  !!
  !!     dw = s^2 P j xi / ( lambda + s^2 j.Pj ),
  !!
  !! which is what a noise r = 1/s^2 on the unscaled quantities gives.
  !! It matters when the residuals of a system are of different size:
  !! the filter treats every observable as equally trustworthy, so one
  !! whose source is ten times the others dominates the covariance and
  !! leaves the rest uncorrected.
  SUBROUTINE kf_update_resid_multi( nt, tw, g, kf, x, seedm, resid, rnoise )
    implicit none
    type(net_t),intent(INOUT) :: nt
    type(twork_t),intent(INOUT) :: tw
    type(grad_t),intent(INOUT) :: g
    type(kalman_t),intent(INOUT) :: kf
    real(8),intent(IN) :: x(:)
    real(8),intent(IN) :: seedm(:,:)
    real(8),intent(IN) :: resid
    real(8),intent(IN),optional :: rnoise
    real(8) :: tdum(nt%tab%na), sc
    call grad_zero( g )
    call net_forward_point( nt, tw, x, tdum )
    call net_backward_point_multi( nt, tw, seedm, g )
    ! The innovation is target minus prediction.  For a residual the
    ! target is zero, so it is -R, not R: passing the residual with the
    ! sign it is computed with drives the filter away from the solution,
    ! which is what the scalar path avoids by forming xi = target - pred.
    sc = 1.d0
    if ( present(rnoise) ) then
       if ( rnoise <= 0.d0 ) then
          write(*,*) "kf_update_resid_multi: observation noise must be > 0"
          stop
       end if
       sc = 1.d0/sqrt( rnoise )
    end if
    call kf_update_grad( nt, kf, g, -resid, sc )
  END SUBROUTINE kf_update_resid_multi

  SUBROUTINE kf_update_obs( nt, tw, g, kf, x, seed, target, pred )
    implicit none
    type(net_t),intent(INOUT) :: nt
    type(twork_t),intent(INOUT) :: tw
    type(grad_t),intent(INOUT) :: g
    type(kalman_t),intent(INOUT) :: kf
    real(8),intent(IN) :: x(:)
    real(8),intent(IN) :: seed(:)
    real(8),intent(IN) :: target
    !> Prediction of the observable.  Absent, it is the linear functional
    !! dot(seed,T), which is what an observable linear in the carried
    !! derivatives evaluates to.  A nonlinear one (a residual with a
    !! u*d^alpha u term) is not that functional and supplies its own.
    real(8),intent(IN),optional :: pred
    real(8) :: xi

    call grad_zero( g )
    call net_grad_point( nt, tw, x, seed, g )
    if ( present(pred) ) then
       xi = target - pred
    else
       xi = target - dot_product( seed(1:nt%tab%na), &
                                  tw%T(1,1:nt%tab%na,nt%nlayer) )
    end if
    call kf_update_grad( nt, kf, g, xi, 1.d0 )
  END SUBROUTINE kf_update_obs

  !> Rank-one update from an observable the caller has already
  !! differentiated.
  !!
  !! The other entries observe one carried derivative at one point, which
  !! is enough for a collocation residual or a value fit.  An observable
  !! of a force field is not of that shape: an energy is a sum of the
  !! network over the atoms of a configuration, a force component is a
  !! sum of seeded first derivatives over those atoms, and a Hessian
  !! entry mixes first and second derivatives over them.  All three are
  !! sums over points of a seeded adjoint, so the caller accumulates
  !! dy/dw into g by calling net_grad_point once per atom and then hands
  !! the row here with the residual xi = target - prediction.
  !!
  !! scale multiplies the row and the residual, which is how observables
  !! of different magnitude are balanced: it plays the part the loss
  !! weights play in a gradient method.  Passing 1 leaves the observation
  !! as it is.
  SUBROUTINE kf_update_grad( nt, kf, g, xi_in, scale )
    implicit none
    type(net_t),intent(INOUT) :: nt
    type(kalman_t),intent(INOUT) :: kf
    type(grad_t),intent(IN) :: g
    real(8),intent(IN) :: xi_in
    real(8),intent(IN) :: scale
    real(8) :: xi, denom
    real(8) :: jpj, s2
    integer :: l, j, i, m
    integer :: ib2, nb2, ob2

    xi = scale*xi_in

    ! The flatten below writes kf%jf, whose length was fixed by the net
    ! the filter was initialized with.  A filter reused on a differently
    ! shaped net would write past the end of jf, which an optimized
    ! build turns into silent heap corruption, so the count is checked.
    m = kf_num_weights( nt )
    if ( m /= kf%nw ) then
       write(*,*) "kf_update_obs: network has", m, " weights but the", &
                  " filter was initialized for", kf%nw
       write(*,*) "  reinitialize the filter (kf_init) for this network"
       stop
    end if
    m = 0
    do l=2,nt%nlayer
       do j=1,nt%ndim(l)
          do i=0,nt%ndim(l-1)
             m = m + 1
             kf%jf(m) = scale*g%nabla(l,j,i)
          end do
       end do
    end do

    ! rank-1 update.  The decoupled filter differs from the dense one in
    ! exactly three places: the P j product, the P downdate, and the
    ! symmetrization run per block instead of over the full matrix.  The
    ! innovation denominator stays GLOBAL (the sum over all blocks), so
    ! the gain still measures the whole row against the whole covariance.
    if ( kf%decoupled ) then
       do ib2 = 1, kf%nb
          nb2 = kf%bsz(ib2);  ob2 = kf%boff(ib2)
          do i = 1, nb2
             kf%pj(ob2+i) = dot_product( kf%Pb(1:nb2,i,ib2), &
                                         kf%jf(ob2+1:ob2+nb2) )
          end do
       end do
    else
       call dsymv_like( kf%P, kf%jf, kf%pj, kf%nw )
    end if
    denom = kf%lam + dot_product( kf%jf(1:kf%nw), kf%pj(1:kf%nw) )
    ! For a positive definite P and lambda > 0 this is strictly positive.
    ! It can only reach zero if the covariance has lost definiteness
    ! through roundoff, in which case dividing would silently fill the
    ! weights with NaN instead of reporting a broken filter state.
    if ( denom <= 0.d0 ) then
       write(*,*) "kf_update_obs: nonpositive innovation denominator:", denom
       write(*,*) "  the covariance has lost positive definiteness;", &
                  " reinitialize the filter (kf_init) or lower p0"
       stop
    end if

    ! Innovation gate (see the type definition).  Inflating the noise is
    ! one more scale s applied to the row and the innovation together,
    ! which is the exact mechanism kf_update_resid_multi already uses
    ! for Sys_rnoise: with jpj = j^T P j, choose s so that
    !     (s xi)^2 = gate^2 ( lambda + s^2 jpj ),
    ! i.e. s^2 = gate^2 lambda / ( xi^2 - gate^2 jpj ), which the gate
    ! condition guarantees positive.  The already-computed pj and denom
    ! are rescaled in place; nothing else changes.
    if ( kf%gate > 0.d0 ) then
       jpj = denom - kf%lam
       if ( xi*xi > kf%gate**2 * denom ) then
          s2 = kf%gate**2 * kf%lam / ( xi*xi - kf%gate**2 * jpj )
          xi = sqrt(s2)*xi
          kf%jf(1:kf%nw) = sqrt(s2)*kf%jf(1:kf%nw)
          kf%pj(1:kf%nw) = sqrt(s2)*kf%pj(1:kf%nw)   ! pj is linear in the row
          denom = kf%lam + s2*jpj
          kf%ngated = kf%ngated + 1
       end if
    end if
    kf%kv(1:kf%nw) = kf%pj(1:kf%nw)/denom

    m = 0
    do l=2,nt%nlayer
       do j=1,nt%ndim(l)
          do i=0,nt%ndim(l-1)
             m = m + 1
             nt%w(l,j,i) = nt%w(l,j,i) + kf%kv(m)*xi
          end do
       end do
    end do

    if ( kf%decoupled ) then
       do ib2 = 1, kf%nb
          nb2 = kf%bsz(ib2);  ob2 = kf%boff(ib2)
          do i = 1, nb2
             kf%Pb(1:nb2,i,ib2) = ( kf%Pb(1:nb2,i,ib2) &
                  - kf%kv(ob2+1:ob2+nb2)*kf%pj(ob2+i) )/kf%lam
          end do
          do i = 1, nb2
             do j = i+1, nb2
                kf%Pb(i,j,ib2) = 0.5d0*( kf%Pb(i,j,ib2) + kf%Pb(j,i,ib2) )
                kf%Pb(j,i,ib2) = kf%Pb(i,j,ib2)
             end do
          end do
          if ( kf%q > 0.d0 ) then
             do i = 1, nb2
                kf%Pb(i,i,ib2) = kf%Pb(i,i,ib2) + kf%q
             end do
          end if
       end do
    else
       do i=1,kf%nw
          kf%P(1:kf%nw,i) = ( kf%P(1:kf%nw,i) - kf%kv(1:kf%nw)*kf%pj(i) )/kf%lam
       end do
       ! The rank-1 downdate is symmetric in exact arithmetic but drifts in
       ! floating point (measured: |P-P^T| about 5e-8 after 5000 nonlinear
       ! updates on a max|P| of 80), and dsymv_like reads P as if symmetric.
       ! Averaging with the transpose removes the drift categorically at
       ! negligible cost next to the O(nw^2) update itself.
       do i=1,kf%nw
          do j=i+1,kf%nw
             kf%P(i,j) = 0.5d0*( kf%P(i,j) + kf%P(j,i) )
             kf%P(j,i) = kf%P(i,j)
          end do
       end do
       ! Process-noise injection (see the type definition): the diagonal
       ! grows by q after the downdate, which keeps the gains from
       ! decaying to zero without inflating the whole matrix the way a
       ! forgetting factor does.
       if ( kf%q > 0.d0 ) then
          do i=1,kf%nw
             kf%P(i,i) = kf%P(i,i) + kf%q
          end do
       end if
    end if
    kf%lam = kf%lam0*kf%lam + 1.d0 - kf%lam0

  END SUBROUTINE kf_update_grad

  !> Iterated EKF over ONE scalar observation (Bell & Cathey: the IEKF
  !! update is Gauss-Newton on the single-observation MAP problem).
  !! The plain EKF linearizes the observation at the prior and steps
  !! once; on an observation quadratic or cubic in the carried slots --
  !! a product-term residual of a coupled system -- that single
  !! linearization is the error the filter then accumulates.  Here
  !! the caller re-evaluates the residual and its row at the CURRENT
  !! iterate between calls, and each call solves the relinearized
  !! problem from the SAME prior:
  !!
  !!     w_{i+1} = w- + K_i ( z - h(w_i) - H_i (w- - w_i) ),  z = 0
  !!
  !! with K_i from H_i and the unchanged prior covariance.  Only the
  !! final iteration downdates P and advances lambda, so one observation
  !! costs one covariance update however many relinearizations are
  !! spent on it.  kf_iekf_begin stores the prior; with a single
  !! iteration the sequence reproduces the plain update exactly (the
  !! correction term vanishes at w_i = w-).
  SUBROUTINE kf_iekf_begin( nt, kf )
    implicit none
    type(net_t),intent(IN) :: nt
    type(kalman_t),intent(INOUT) :: kf
    integer :: l, j, i, m
    if ( .not. allocated( kf%wprior) ) &
         allocate( kf%wprior(kf%nw), kf%wcur(kf%nw) )
    m = 0
    do l=2,nt%nlayer
       do j=1,nt%ndim(l)
          do i=0,nt%ndim(l-1)
             m = m + 1
             kf%wprior(m) = nt%w(l,j,i)
          end do
       end do
    end do
  END SUBROUTINE kf_iekf_begin

  !> One relinearization of the iterated update.  g holds dR/dw at the
  !! current weights, resid the residual there, scale = 1/sqrt(rnoise)
  !! as in kf_update_resid_multi.  final selects the covariance update.
  SUBROUTINE kf_iekf_iter( nt, tw, g, kf, x, seedm, resid, rnoise, final )
    implicit none
    type(net_t),intent(INOUT) :: nt
    type(twork_t),intent(INOUT) :: tw
    type(grad_t),intent(INOUT) :: g
    type(kalman_t),intent(INOUT) :: kf
    real(8),intent(IN) :: x(:)
    real(8),intent(IN) :: seedm(:,:)
    real(8),intent(IN) :: resid
    real(8),intent(IN) :: rnoise
    logical,intent(IN) :: final
    real(8) :: xi, denom, scale
    real(8) :: tdum(nt%tab%na)
    integer :: l, j, i, m, ib2, nb2, ob2

    if ( rnoise <= 0.d0 ) then
       write(*,*) "kf_iekf_iter: observation noise must be > 0"
       stop
    end if
    scale = 1.d0/sqrt( rnoise )
    call grad_zero( g )
    call net_forward_point( nt, tw, x, tdum )
    call net_backward_point_multi( nt, tw, seedm, g )

    m = kf_num_weights( nt )
    if ( m /= kf%nw ) then
       write(*,*) "kf_iekf_iter: network has", m, " weights but the", &
                  " filter was initialized for", kf%nw
       stop
    end if
    m = 0
    do l=2,nt%nlayer
       do j=1,nt%ndim(l)
          do i=0,nt%ndim(l-1)
             m = m + 1
             kf%jf(m)   = scale*g%nabla(l,j,i)
             kf%wcur(m) = nt%w(l,j,i)
          end do
       end do
    end do

    if ( kf%decoupled ) then
       do ib2 = 1, kf%nb
          nb2 = kf%bsz(ib2);  ob2 = kf%boff(ib2)
          do i = 1, nb2
             kf%pj(ob2+i) = dot_product( kf%Pb(1:nb2,i,ib2), &
                                         kf%jf(ob2+1:ob2+nb2) )
          end do
       end do
    else
       call dsymv_like( kf%P, kf%jf, kf%pj, kf%nw )
    end if
    denom = kf%lam + dot_product( kf%jf(1:kf%nw), kf%pj(1:kf%nw) )
    if ( denom <= 0.d0 ) then
       write(*,*) "kf_iekf_iter: nonpositive innovation denominator:", denom
       stop
    end if
    kf%kv(1:kf%nw) = kf%pj(1:kf%nw)/denom

    ! z = 0, so the linearized innovation at the prior is
    !   -scale*R  -  j_scaled . ( w- - w_i ),
    ! whose second term vanishes on the first iteration.
    xi = -scale*resid &
         - dot_product( kf%jf(1:kf%nw), kf%wprior(1:kf%nw)-kf%wcur(1:kf%nw) )

    m = 0
    do l=2,nt%nlayer
       do j=1,nt%ndim(l)
          do i=0,nt%ndim(l-1)
             m = m + 1
             nt%w(l,j,i) = kf%wprior(m) + kf%kv(m)*xi
          end do
       end do
    end do

    if ( final ) then
       if ( kf%decoupled ) then
          do ib2 = 1, kf%nb
             nb2 = kf%bsz(ib2);  ob2 = kf%boff(ib2)
             do i = 1, nb2
                kf%Pb(1:nb2,i,ib2) = ( kf%Pb(1:nb2,i,ib2) &
                     - kf%kv(ob2+1:ob2+nb2)*kf%pj(ob2+i) )/kf%lam
             end do
             do i = 1, nb2
                do j = i+1, nb2
                   kf%Pb(i,j,ib2) = 0.5d0*( kf%Pb(i,j,ib2) + kf%Pb(j,i,ib2) )
                   kf%Pb(j,i,ib2) = kf%Pb(i,j,ib2)
                end do
             end do
             if ( kf%q > 0.d0 ) then
                do i = 1, nb2
                   kf%Pb(i,i,ib2) = kf%Pb(i,i,ib2) + kf%q
                end do
             end if
          end do
       else
          do i=1,kf%nw
             kf%P(1:kf%nw,i) = ( kf%P(1:kf%nw,i) &
                  - kf%kv(1:kf%nw)*kf%pj(i) )/kf%lam
          end do
          do i=1,kf%nw
             do j=i+1,kf%nw
                kf%P(i,j) = 0.5d0*( kf%P(i,j) + kf%P(j,i) )
                kf%P(j,i) = kf%P(i,j)
             end do
          end do
          if ( kf%q > 0.d0 ) then
             do i=1,kf%nw
                kf%P(i,i) = kf%P(i,i) + kf%q
             end do
          end if
       end if
       kf%lam = kf%lam0*kf%lam + 1.d0 - kf%lam0
    end if
  END SUBROUTINE kf_iekf_iter

  SUBROUTINE dsymv_like( P, v, w, n )
    implicit none
    integer,intent(IN) :: n
    real(8),intent(IN) :: P(n,n), v(n)
    real(8),intent(OUT) :: w(n)
    integer :: i
    do i=1,n
       w(i) = dot_product( P(1:n,i), v(1:n) )   ! P symmetric
    end do
  END SUBROUTINE dsymv_like

END MODULE kalman_module
