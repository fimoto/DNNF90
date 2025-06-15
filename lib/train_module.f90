!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (train_module.f90) is part of DNNF90.
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
! Instance based training: gradients and optimizer state carried in
! derived types instead of module variables.
!
! Motivation.  A machine learning force field trains one network per
! element species from a single loss over structures.  With module level
! state that is impossible: one process holds one network, and no two
! threads may accumulate gradients at the same time.  Here
!
!   grad_t  holds the gradient accumulator and the optimizer moments of
!           one network, so several species train side by side, and
!   twork_t holds every mutable intermediate of one gradient evaluation,
!           so one instance per thread makes accumulation thread safe.
!
! The multi-index and Bell tables stay shared: they are written once by
! init_hod_tables and are read only afterwards.
!
! This module deliberately does not use global_variables.  The kernels
! mirror those of feedforward_module, backprop_module and
! nabla_weight_module, and reproduce them bitwise.
! -----------------------------------------------------------------------
MODULE train_module

  use multi_index_bell_module, only: tabset_t, tanh_derivs_ts, act_derivs_ts, erf_derivs
#ifdef USE_BLAS
  use blas_wrap_module, only: bgemm
#endif
  use net_module, only: net_t

  implicit none

  PRIVATE
  PUBLIC :: twork_t, grad_t, bwork_t
  PUBLIC :: twork_init, twork_free, grad_init, grad_free, grad_zero, grad_add
  PUBLIC :: net_grad_point, net_forward_point, net_forward_point_multi, net_backward_point, net_backward_point_multi
  PUBLIC :: bwork_init, bwork_free, net_grad_batch
  PUBLIC :: opt_sgd_step, opt_adam_step
#ifdef PHASE_TIMING
  !> Wall time accumulated in the two phases of the propagation: the
  !! linear contraction of each layer (one GEMM over the slot axis) and
  !! the Bell-table part (activation derivatives plus the Faa di Bruno
  !! gather over fq_*/pair_*).  How the cost divides between them is
  !! what a CUDA port needs to know first: the GEMM maps onto cuBLAS
  !! directly, the irregular gather does not.  Measurement builds only
  !! (-DPHASE_TIMING): the default build compiles this out, and the
  !! accumulators are not thread safe.
  PUBLIC :: phase_t_gemm, phase_t_bell, phase_reset
  real(8),save :: phase_t_gemm = 0.d0
  real(8),save :: phase_t_bell = 0.d0
#endif

  !> Mutable intermediates of one gradient evaluation.  One per thread.
  !> Only the quantities the reverse pass rereads are stored per layer:
  !! T (for the gradient contraction) and the fused cache h (for the
  !! adjoint).  S, Tbar and Sbar live for one layer at a time, so the
  !! training memory is 2 per-layer arrays plus 3 single planes instead
  !! of 5 per-layer arrays.
  TYPE :: twork_t
     real(8),allocatable :: T(:,:,:)  ! (0:ndmax, na, nlayer)
     real(8),allocatable :: h(:,:,:)  ! (ndmax, na, nlayer) fused cache
     real(8),allocatable :: S(:,:)  ! (ndmax, na) current layer
     real(8),allocatable :: Tb(:,:)  ! (ndmax, na) adjoint, current
     real(8),allocatable :: Sb(:,:)  ! (ndmax, na) adjoint, current
     ! per-call scratch, persistent for the same reasons as in work_t
     real(8),allocatable :: wl(:,:)  ! (ndmax, 0:ndmax) weight slice
     real(8),allocatable :: acc(:,:)  ! (ndmax, 0:ndmax) gradient block
     ! neuron-axis vectorized scratch: j is the fast dimension
     real(8),allocatable :: bqv(:,:,:)  ! (ndmax, na, kmax) Bell values
     real(8),allocatable :: dt(:,:)  ! (ndmax, 0:kmax+1) tanh derivs
     real(8),allocatable :: tv(:)
     real(8),allocatable :: ttv(:)
     real(8),allocatable :: htv(:)
     real(8),allocatable :: bsv(:)
  END TYPE twork_t

  !> Work space of the batched value-fitting pass.  Separate from
  !! twork_t because it carries a batch axis instead of a derivative
  !! axis: the plain regression needs no multi-index tables, so the
  !! whole minibatch becomes three matrix products per layer.
  TYPE :: bwork_t
     real(8),allocatable :: Z(:,:,:)   ! (0:ndmax, nb, nlayer) activations
     real(8),allocatable :: Zd(:,:,:)  ! (ndmax, nb, nlayer) sigma'(pre)
     real(8),allocatable :: D(:,:,:)   ! (ndmax, nb, nlayer) deltas
     real(8),allocatable :: wl(:,:)    ! contiguous layer weights
     real(8),allocatable :: acc(:,:)   ! gradient block
     real(8),allocatable :: Zp(:,:)    ! staged activations of l-1
     real(8),allocatable :: Ac(:,:)    ! staged pre-activations
     real(8),allocatable :: Dc(:,:)    ! staged deltas
     real(8),allocatable :: Dn(:,:)    ! staged deltas of layer l
     integer :: nb = 0
  END TYPE bwork_t

  !> Gradient accumulator and optimizer moments of one network.
  TYPE :: grad_t
     real(8),allocatable :: nabla(:,:,:)  ! (nlayer, ndmax, 0:ndmax)
     real(8),allocatable :: m(:,:,:)  ! Adam first moment
     real(8),allocatable :: v(:,:,:)  ! Adam second moment
  END TYPE grad_t

CONTAINS

#ifdef PHASE_TIMING
  !> Wall clock in seconds, for the phase timers.
  REAL(8) FUNCTION pt_now()
    implicit none
    integer(8) :: c, r
    call system_clock( c, r )
    pt_now = dble(c)/dble(r)
  END FUNCTION pt_now

  SUBROUTINE phase_reset()
    implicit none
    phase_t_gemm = 0.d0
    phase_t_bell = 0.d0
  END SUBROUTINE phase_reset
#endif

  SUBROUTINE twork_init( tw, nt )
    implicit none
    type(twork_t),intent(OUT) :: tw
    type(net_t),intent(IN) :: nt
    allocate( tw%T(0:nt%ndmax,nt%tab%na,nt%nlayer) )
    allocate( tw%h(nt%ndmax,nt%tab%na,nt%nlayer) )
    allocate( tw%S(nt%ndmax,nt%tab%na) )
    allocate( tw%Tb(nt%ndmax,nt%tab%na) )
    allocate( tw%Sb(nt%ndmax,nt%tab%na) )
    allocate( tw%wl(nt%ndmax,0:nt%ndmax), tw%acc(nt%ndmax,0:nt%ndmax) )
    allocate( tw%bqv(nt%ndmax,nt%tab%na,max(nt%tab%kmax,1)) )
    allocate( tw%dt(nt%ndmax,0:nt%tab%kmax+1) )
    allocate( tw%tv(nt%ndmax), tw%ttv(nt%ndmax), tw%htv(nt%ndmax), tw%bsv(nt%ndmax) )
    tw%T = 0.d0; tw%S = 0.d0; tw%h = 0.d0; tw%Tb = 0.d0; tw%Sb = 0.d0
    tw%T(0,1,1:nt%nlayer) = 1.d0        ! constant bias channel
  END SUBROUTINE twork_init

  SUBROUTINE twork_free( tw )
    implicit none
    type(twork_t),intent(INOUT) :: tw
    if ( allocated(  tw%T) )  deallocate( tw%T )
    if ( allocated(  tw%S) )  deallocate( tw%S )
    if ( allocated(  tw%h) )  deallocate( tw%h )
    if ( allocated(  tw%Tb) ) deallocate( tw%Tb )
    if ( allocated(  tw%Sb) ) deallocate( tw%Sb )
    if ( allocated(  tw%wl) ) deallocate( tw%wl, tw%acc, tw%bqv, tw%dt, &
                                        tw%tv, tw%ttv, tw%htv, tw%bsv )
  END SUBROUTINE twork_free

  SUBROUTINE grad_init( g, nt )
    implicit none
    type(grad_t),intent(OUT) :: g
    type(net_t),intent(IN) :: nt
    ! The Adam moments are optimizer state, not part of an accumulator.
    ! Allocating them here cost two extra copies of the weight cube in
    ! every per-thread accumulator (two thirds of the gradient memory,
    ! never touched by plain descent), so they are created on first use
    ! by opt_adam_step instead.
    allocate( g%nabla(nt%nlayer,nt%ndmax,0:nt%ndmax) )
    g%nabla = 0.d0
  END SUBROUTINE grad_init

  SUBROUTINE grad_free( g )
    implicit none
    type(grad_t),intent(INOUT) :: g
    if ( allocated(  g%nabla) ) deallocate( g%nabla )
    if ( allocated(  g%m) )     deallocate( g%m )
    if ( allocated(  g%v) )     deallocate( g%v )
  END SUBROUTINE grad_free

  SUBROUTINE grad_zero( g )
    implicit none
    type(grad_t),intent(INOUT) :: g
    g%nabla = 0.d0
  END SUBROUTINE grad_zero

  !> gsum <- gsum + gadd (reduction of per-thread accumulators)
  SUBROUTINE grad_add( gsum, gadd )
    implicit none
    type(grad_t),intent(INOUT) :: gsum
    type(grad_t),intent(IN) :: gadd
    gsum%nabla = gsum%nabla + gadd%nabla
  END SUBROUTINE grad_add

  !> Forward pass, adjoint pass and gradient accumulation for one point.
  !!
  !! seed(1:nt%tab%na) is dL/dT at the output neuron, that is the loss
  !! specific seeding.  The routine adds this point's contribution to
  !! g%nabla.  It touches only its arguments and the shared read-only
  !! tables, so it may be called concurrently with one twork_t and one
  !! grad_t per thread.
  !> Forward half of net_grad_point: fills the work space with the
  !! carried derivatives of every layer and returns the output values
  !! T_alpha, so the caller can form the loss seed dL/dT_alpha before
  !! calling net_backward_point.  net_grad_point remains the fused
  !! convenience for callers whose seed does not need the outputs.
  SUBROUTINE net_forward_point( nt, tw, x, tout )
    implicit none
    type(net_t),intent(IN) :: nt
    type(twork_t),intent(INOUT) :: tw
    real(8),intent(IN) :: x(:)
    real(8),intent(OUT) :: tout(:)
    call forward_core( nt, tw, x )
    tout(1:nt%tab%na) = tw%T(1,1:nt%tab%na,nt%nlayer)
  END SUBROUTINE net_forward_point

  !> Forward pass returning every carried derivative of every output
  !! component, leaving the work planes ready for the adjoint.
  !!
  !! net_forward_point is the one-component case; the propagation is
  !! identical, since it already runs over every neuron of the output
  !! layer.
  SUBROUTINE net_forward_point_multi( nt, tw, x, tm )
    implicit none
    type(net_t),intent(IN) :: nt
    type(twork_t),intent(INOUT) :: tw
    real(8),intent(IN) :: x(:)
    real(8),intent(OUT) :: tm(:,:)
    real(8) :: tdum(nt%tab%na)
    integer :: nout
    nout = nt%ndim(nt%nlayer)
    call net_forward_point( nt, tw, x, tdum )
    tm(1:nout,1:nt%tab%na) = tw%T(1:nout,1:nt%tab%na,nt%nlayer)
  END SUBROUTINE net_forward_point_multi

  !> The forward sweep itself.  Writes only into the work space, which is
  !! what lets net_grad_point run it without a buffer of its own.
  SUBROUTINE forward_core( nt, tw, x )
    implicit none
    type(net_t),intent(IN) :: nt
    type(twork_t),intent(INOUT) :: tw
    real(8),intent(IN) :: x(:)
    integer :: l, i, ia, q, it, p, nd, ndm, j, jj
    real(8) :: dsc(0:nt%tab%kmax+1)
#ifdef PHASE_TIMING
    real(8) :: tp0, tp1, tp2
#endif

    ! ---------------- forward ----------------
    do j=1,nt%ndim(1)
       tw%T(j,1,1) = x(j)
       tw%T(j,2:nt%tab%na,1) = 0.d0
       if ( nt%tab%ind_e1(j) > 0 ) tw%T(j,nt%tab%ind_e1(j),1) = 1.d0
    end do

    do l=2,nt%nlayer
       nd  = nt%ndim(l)
       ndm = nt%ndim(l-1)
       tw%wl(1:nd,0:ndm) = nt%w(l,1:nd,0:ndm)

#ifdef PHASE_TIMING
       tp0 = pt_now()
#endif
#ifdef USE_BLAS
       call bgemm( 'N', 'N', nd, nt%tab%na, ndm+1, &
                   tw%wl, nt%ndmax, tw%T(0:,1:,l-1), nt%ndmax+1, &
                   tw%S, nt%ndmax )
#else
       do ia=1,nt%tab%na
          tw%S(1:nd,ia) = tw%wl(1:nd,0)*tw%T(0,ia,l-1)
          do i=1,ndm
             tw%S(1:nd,ia) = tw%S(1:nd,ia) + tw%wl(1:nd,i)*tw%T(i,ia,l-1)
          end do
       end do
#endif
#ifdef PHASE_TIMING
       tp1 = pt_now();  phase_t_gemm = phase_t_gemm + (tp1-tp0)
#endif

       if ( l < nt%nlayer ) then
          ! Neuron-axis vectorization: the neuron index j is the batch
          ! axis of the Bell recurrence.  The loops are interchanged so
          ! j runs innermost over the contiguous first dimension of S,
          ! and the table lookups become scalar broadcasts over the j
          ! vector.  Per (j, ia) the arithmetic and its order are exactly
          ! those of the per-neuron reference, so results are bitwise
          ! identical while the kernel runs at SIMD width.
          ! sigma^(q)(S(j,1)) per neuron via the scalar routine: the
          ! vector tanh of the math library is allowed to differ from the
          ! scalar one by an ulp, which broke bitwise identity with the
          ! reference kernel.  This part is O(width) per layer, so the
          ! scalar calls cost nothing; the O(width * terms) Bell part
          ! below stays vectorized.
          do j=1,nd
             call act_derivs_ts( nt%tab, tw%S(j,1), nt%tab%kmax+1, dsc )
             tw%dt(j,0:nt%tab%kmax+1) = dsc(0:nt%tab%kmax+1)
          end do
          tw%T(1:nd,1,l) = tw%dt(1:nd,0)
          tw%h(1:nd,1,l) = tw%dt(1:nd,1)
          do ia=2,nt%tab%na
             p = nt%tab%alpha_deg(ia)
             tw%bqv(1:nd,ia,1) = tw%S(1:nd,ia)
             tw%ttv(1:nd) = tw%dt(1:nd,1)*tw%S(1:nd,ia)
             tw%htv(1:nd) = tw%dt(1:nd,2)*tw%S(1:nd,ia)
             do q=2,p
                tw%bsv(1:nd) = 0.d0
                do it=nt%tab%fq_start(ia,q),nt%tab%fq_start(ia,q)+nt%tab%fq_num(ia,q)-1
                   ! Scalar on purpose: vectorizing this table-driven
                   ! accumulation was measured slower (see work_t note).
!GCC$ novector
                   do jj=1,nd
                      tw%bsv(jj) = tw%bsv(jj) + nt%tab%fq_c(it) &
                           *tw%S(jj,nt%tab%fq_ib(it))*tw%bqv(jj,nt%tab%fq_id(it),q-1)
                   end do
                end do
                tw%bqv(1:nd,ia,q) = tw%bsv(1:nd)
                tw%ttv(1:nd) = tw%ttv(1:nd) + tw%dt(1:nd,q)*tw%bsv(1:nd)
                tw%htv(1:nd) = tw%htv(1:nd) + tw%dt(1:nd,q+1)*tw%bsv(1:nd)
             end do
             tw%T(1:nd,ia,l) = tw%ttv(1:nd)
             tw%h(1:nd,ia,l) = tw%htv(1:nd)
          end do
       else
          tw%T(1:nd,1:nt%tab%na,l) = tw%S(1:nd,1:nt%tab%na)
       end if
#ifdef PHASE_TIMING
       tp2 = pt_now();  phase_t_bell = phase_t_bell + (tp2-tp1)
#endif
    end do


  END SUBROUTINE forward_core

  !> Adjoint half: consumes the forward state left in the work space by
  !! net_forward_point and accumulates the weight gradient of the loss
  !! whose seed dL/dT_alpha is given.  Must follow a matching forward
  !! on the same work space and weights.
  SUBROUTINE net_backward_point( nt, tw, seed, g )
    implicit none
    type(net_t),intent(IN) :: nt
    type(twork_t),intent(INOUT) :: tw
    real(8),intent(IN) :: seed(:)
    type(grad_t),intent(INOUT) :: g
    integer :: l, i, ia, ib, ip, it, nd, ndm, j

    ! ---------------- adjoint with fused gradient accumulation ----------------
    ! Tbar and Sbar live one layer at a time.  The gradient of layer l is
    ! accumulated as soon as Sbar(l) is complete; it depends only on
    tw%Tb(1:nt%ndim(nt%nlayer),1:nt%tab%na) = 0.d0
    tw%Tb(1,1:nt%tab%na) = seed(1:nt%tab%na)
    call backward_core( nt, tw, g )
  END SUBROUTINE net_backward_point

  !> The reverse sweep itself, once the adjoint of the output layer
  !! has been seeded.  Both seeding routines call it, so the two
  !! share one implementation and one verification.
  SUBROUTINE backward_core( nt, tw, g )
    implicit none
    type(net_t),intent(IN) :: nt
    type(twork_t),intent(INOUT) :: tw
    type(grad_t),intent(INOUT) :: g
    integer :: l, i, ia, ib, ip, it, nd, ndm, j, jj
#ifdef PHASE_TIMING
    real(8) :: tp0, tp1, tp2
#endif

    ! ---------------- adjoint with fused gradient accumulation ----------------
    ! Tbar and Sbar live one layer at a time.  The gradient of layer l is
    ! accumulated as soon as Sbar(l) is complete; it depends only on

    do l=nt%nlayer,2,-1
       nd  = nt%ndim(l)
       ndm = nt%ndim(l-1)

#ifdef PHASE_TIMING
       tp0 = pt_now()
#endif
       if ( l == nt%nlayer ) then
          tw%Sb(1:nd,1:nt%tab%na) = tw%Tb(1:nd,1:nt%tab%na)
       else
          ! Adjoint pair sums on j vectors; per (j, ib) the accumulation
          ! order over ia and ip is exactly that of the reference.
          tw%bsv(1:nd) = tw%Tb(1:nd,1)*tw%h(1:nd,1,l)
          do ia=2,nt%tab%na
             tw%bsv(1:nd) = tw%bsv(1:nd) + tw%Tb(1:nd,ia)*tw%h(1:nd,ia,l)
          end do
          tw%Sb(1:nd,1) = tw%bsv(1:nd)
          do ib=2,nt%tab%na
             tw%bsv(1:nd) = 0.d0
             do ip=nt%tab%pair_start(ib),nt%tab%pair_start(ib)+nt%tab%pair_num(ib)-1
                ! Scalar on purpose, as in forward_core.
!GCC$ novector
                do jj=1,nd
                   tw%bsv(jj) = tw%bsv(jj) + nt%tab%pair_binom(ip) &
                        *tw%Tb(jj,nt%tab%pair_ia(ip))*tw%h(jj,nt%tab%pair_id(ip),l)
                end do
             end do
             tw%Sb(1:nd,ib) = tw%bsv(1:nd)
          end do
       end if
#ifdef PHASE_TIMING
       tp1 = pt_now();  phase_t_bell = phase_t_bell + (tp1-tp0)
#endif

       ! gradient of layer l (reads only Sbar(l) and T(l-1))
#ifdef USE_BLAS
       ! acc(1:nd, 0:ndm) = Sb(1:nd, 1:na) T(0:ndm, 1:na)^T
       call bgemm( 'N', 'T', nd, ndm+1, nt%tab%na, &
                   tw%Sb, nt%ndmax, tw%T(0:,1:,l-1), nt%ndmax+1, &
                   tw%acc, nt%ndmax )
#else
       tw%acc(1:nd,0:ndm) = 0.d0
       do ia=1,nt%tab%na
          do i=0,ndm
             tw%acc(1:nd,i) = tw%acc(1:nd,i) + tw%Sb(1:nd,ia)*tw%T(i,ia,l-1)
          end do
       end do
#endif
       g%nabla(l,1:nd,0:ndm) = g%nabla(l,1:nd,0:ndm) + tw%acc(1:nd,0:ndm)

       ! Tbar of layer l-1 (overwrites the single plane after Sbar is done).
       ! The weight slice is copied once per layer: w has the layer index
       ! first, so the direct column nt%w(l,1:nd,i) is strided by the
       ! layer count, which at large widths turns this contraction into
       ! one cache line per element.  The copy restores unit stride and
       ! leaves the summation order unchanged.
       if ( l > 2 ) then
          tw%wl(1:nd,1:ndm) = nt%w(l,1:nd,1:ndm)
#ifdef USE_BLAS
          ! Tb(1:ndm, 1:na) = wl(1:nd, 1:ndm)^T Sb(1:nd, 1:na)
          call bgemm( 'T', 'N', ndm, nt%tab%na, nd, &
                      tw%wl(1:,1:), nt%ndmax, tw%Sb, nt%ndmax, &
                      tw%Tb, nt%ndmax )
#else
          do ia=1,nt%tab%na
             do i=1,ndm
                tw%Tb(i,ia) = dot_product( tw%wl(1:nd,i), tw%Sb(1:nd,ia) )
             end do
          end do
#endif
       end if
#ifdef PHASE_TIMING
       tp2 = pt_now();  phase_t_gemm = phase_t_gemm + (tp2-tp1)
#endif
    end do

  END SUBROUTINE backward_core

  !> Adjoint pass seeded on every output component.
  !!
  !! seedm(i,ia) is dL/dT for multi-index ia of output i, so a loss that
  !! couples the components of a vector field is expressed directly.  The
  !! scalar routine is the case of one output; the recursion below it is
  !! the same, since the reverse sweep already runs over all neurons of
  !! the output layer.
  SUBROUTINE net_backward_point_multi( nt, tw, seedm, g )
    implicit none
    type(net_t),intent(IN) :: nt
    type(twork_t),intent(INOUT) :: tw
    real(8),intent(IN) :: seedm(:,:)
    type(grad_t),intent(INOUT) :: g
    integer :: nout
    nout = nt%ndim(nt%nlayer)
    if ( size(seedm,1) < nout .or. size(seedm,2) < nt%tab%na ) then
       write(*,*) "net_backward_point_multi: seedm must be at least", &
            nout, " by", nt%tab%na
       stop
    end if
    tw%Tb(1:nt%ndim(nt%nlayer),1:nt%tab%na) = 0.d0
    tw%Tb(1:nout,1:nt%tab%na) = seedm(1:nout,1:nt%tab%na)
    call backward_core( nt, tw, g )
  END SUBROUTINE net_backward_point_multi

  !> Forward pass, adjoint pass and gradient accumulation for one point.
  SUBROUTINE net_grad_point( nt, tw, x, seed, g )
    implicit none
    type(net_t),intent(IN) :: nt
    type(twork_t),intent(INOUT) :: tw
    real(8),intent(IN) :: x(:)
    real(8),intent(IN) :: seed(:)
    type(grad_t),intent(INOUT) :: g
    ! The forward result is not needed here, so the sweep is run through
    ! its core and nothing is published.
    call forward_core( nt, tw, x )
    call net_backward_point( nt, tw, seed, g )
  END SUBROUTINE net_grad_point


  SUBROUTINE bwork_init( bw, nt, nb )
    implicit none
    type(bwork_t),intent(OUT) :: bw
    type(net_t),intent(IN) :: nt
    integer,intent(IN) :: nb
    allocate( bw%Z(0:nt%ndmax,nb,nt%nlayer) )
    allocate( bw%Zd(nt%ndmax,nb,nt%nlayer) );  bw%Zd = 0.d0
    allocate( bw%D(nt%ndmax,nb,nt%nlayer) )
    allocate( bw%wl(nt%ndmax,0:nt%ndmax), bw%acc(nt%ndmax,0:nt%ndmax) )
    allocate( bw%Zp(0:nt%ndmax,nb), bw%Ac(nt%ndmax,nb) )
    allocate( bw%Dc(nt%ndmax,nb), bw%Dn(nt%ndmax,nb) )
    bw%Zp = 0.d0; bw%Ac = 0.d0; bw%Dc = 0.d0; bw%Dn = 0.d0
    bw%Z = 0.d0; bw%D = 0.d0
    bw%nb = nb
  END SUBROUTINE bwork_init

  SUBROUTINE bwork_free( bw )
    implicit none
    type(bwork_t),intent(INOUT) :: bw
    if ( allocated(  bw%Z) ) deallocate( bw%Z, bw%D, bw%wl, bw%acc, &
                                        bw%Zp, bw%Ac, bw%Dc, bw%Dn )
    bw%nb = 0
  END SUBROUTINE bwork_free

  !> Weight gradient of the plain squared-error fit over a whole batch:
  !!   L = (coef/2) sum_n ( N(x_n) - y_n )^2
  !! Three matrix products per layer (forward, delta backpropagation,
  !! gradient block) replace the per-point loop; the result is added to
  !! g%nabla.  This is the value-only path, so it uses no derivative
  !! tables.  Every matrix argument is a whole two-dimensional buffer:
  !! the layer planes are staged through Zp, Ac and Dc so that each
  !! product sees a plain contiguous matrix rather than a strided
  !! section of the rank-3 work arrays.
  SUBROUTINE net_grad_batch( nt, bw, X, Y, coef, g )
    implicit none
    integer :: jb, jz
    real(8) :: dz1(0:1)
    type(net_t),intent(IN) :: nt
    type(bwork_t),intent(INOUT) :: bw
    real(8),intent(IN) :: X(:,:)      ! (ndim(1), nb) descriptors
    real(8),intent(IN) :: Y(:)        ! (nb) targets of the single output
    real(8),intent(IN) :: coef
    type(grad_t),intent(INOUT) :: g
    integer :: ib, l, nd, ndm, nb, i
    nb = bw%nb
    if ( nt%ndim(nt%nlayer) /= 1 ) then
       write(*,*) "net_grad_batch: the batched value path is written for"
       write(*,*) "  a single output, but the network has", nt%ndim(nt%nlayer)
       stop
    end if

    ! ---- load the batch; row 0 is the constant bias channel ----
    bw%Z(0,1:nb,1) = 1.d0
    do ib=1,nb
       bw%Z(1:nt%ndim(1),ib,1) = X(1:nt%ndim(1),ib)
    end do

    ! ---- forward ----
    do l=2,nt%nlayer
       nd  = nt%ndim(l)
       ndm = nt%ndim(l-1)
       bw%wl(1:nd,0:ndm) = nt%w(l,1:nd,0:ndm)
       bw%Zp(0:ndm,1:nb) = bw%Z(0:ndm,1:nb,l-1)
#ifdef USE_BLAS
       call bgemm( 'N','N', nd, nb, ndm+1, &
                   bw%wl, nt%ndmax, bw%Zp, nt%ndmax+1, bw%Ac, nt%ndmax )
#else
       do ib=1,nb
          bw%Ac(1:nd,ib) = bw%wl(1:nd,0)*bw%Zp(0,ib)
          do i=1,ndm
             bw%Ac(1:nd,ib) = bw%Ac(1:nd,ib) + bw%wl(1:nd,i)*bw%Zp(i,ib)
          end do
       end do
#endif
       bw%Z(0,1:nb,l) = 1.d0
       if ( l < nt%nlayer ) then
          ! the batched value path applies sigma itself; the derivative
       ! orders are not needed here, only sigma^(0)
       ! sigma and sigma' from the one interface every activation
       ! implements, so that this path is not tied to tanh
       if ( nt%tab%iact == 0 ) then
          bw%Z(1:nd,1:nb,l)  = tanh( bw%Ac(1:nd,1:nb) )
          bw%Zd(1:nd,1:nb,l) = 1.d0 - bw%Z(1:nd,1:nb,l)**2
       else
          do jb=1,nb
             do jz=1,nd
                call act_derivs_ts( nt%tab, bw%Ac(jz,jb), 1, dz1 )
                bw%Z(jz,jb,l)  = dz1(0)
                bw%Zd(jz,jb,l) = dz1(1)
             end do
          end do
       end if
       else
          bw%Z(1:nd,1:nb,l) = bw%Ac(1:nd,1:nb)    ! linear read-out
       end if
    end do

    ! ---- output seed ----
    bw%D(1,1:nb,nt%nlayer) = coef*( bw%Z(1,1:nb,nt%nlayer) - Y(1:nb) )

    ! ---- backpropagate the deltas ----
    do l=nt%nlayer-1,2,-1
       nd  = nt%ndim(l)
       ndm = nt%ndim(l+1)
       bw%wl(1:ndm,1:nd) = nt%w(l+1,1:ndm,1:nd)
       bw%Dc(1:ndm,1:nb) = bw%D(1:ndm,1:nb,l+1)
#ifdef USE_BLAS
       call bgemm( 'T','N', nd, nb, ndm, &
                   bw%wl(1:,1:), nt%ndmax, bw%Dc, nt%ndmax, bw%Dn, nt%ndmax )
#else
       do ib=1,nb
          do i=1,nd
             bw%Dn(i,ib) = dot_product( bw%wl(1:ndm,i), bw%Dc(1:ndm,ib) )
          end do
       end do
#endif
       bw%D(1:nd,1:nb,l) = bw%Dn(1:nd,1:nb)*bw%Zd(1:nd,1:nb,l)
    end do

    ! ---- gradient blocks: nabla_l += D_l Z_{l-1}^T ----
    do l=nt%nlayer,2,-1
       nd  = nt%ndim(l)
       ndm = nt%ndim(l-1)
       bw%Dc(1:nd,1:nb) = bw%D(1:nd,1:nb,l)
       bw%Zp(0:ndm,1:nb) = bw%Z(0:ndm,1:nb,l-1)
#ifdef USE_BLAS
       call bgemm( 'N','T', nd, ndm+1, nb, &
                   bw%Dc, nt%ndmax, bw%Zp, nt%ndmax+1, bw%acc, nt%ndmax )
#else
       bw%acc(1:nd,0:ndm) = 0.d0
       do ib=1,nb
          do i=0,ndm
             bw%acc(1:nd,i) = bw%acc(1:nd,i) + bw%Dc(1:nd,ib)*bw%Zp(i,ib)
          end do
       end do
#endif
       g%nabla(l,1:nd,0:ndm) = g%nabla(l,1:nd,0:ndm) + bw%acc(1:nd,0:ndm)
    end do

  END SUBROUTINE net_grad_batch


  !> Plain gradient descent step, w <- w - eta*nabla/nbatch.
  SUBROUTINE opt_sgd_step( nt, g, eta, nbatch )
    implicit none
    type(net_t),intent(INOUT) :: nt
    type(grad_t),intent(IN) :: g
    real(8),intent(IN) :: eta
    integer,intent(IN) :: nbatch
    integer :: l, j, i
    ! Single fused contiguous pass: the layer index is the fastest
    ! dimension, so per-layer slices would be strided.  No temporary
    ! cube is formed.
    do i=0,nt%ndmax
       do j=1,nt%ndmax
          do l=2,nt%nlayer
             nt%w(l,j,i) = nt%w(l,j,i) - eta*( g%nabla(l,j,i)/dble(nbatch) )
          end do
       end do
    end do
  END SUBROUTINE opt_sgd_step

  !> Adam step, with the same formulas and bias correction as
  !! optimizer_module.  istep is the running epoch counter (>=1).
  SUBROUTINE opt_adam_step( nt, g, eta, beta1, beta2, eps, nbatch, istep )
    implicit none
    type(net_t),intent(INOUT) :: nt
    type(grad_t),intent(INOUT) :: g
    real(8),intent(IN) :: eta, beta1, beta2, eps
    integer,intent(IN) :: nbatch, istep
    real(8) :: bc1, bc2, gval
    integer :: l, j, i
    if ( .not. allocated(  g%m) ) then
       allocate( g%m(nt%nlayer,nt%ndmax,0:nt%ndmax) );  g%m = 0.d0
       allocate( g%v(nt%nlayer,nt%ndmax,0:nt%ndmax) );  g%v = 0.d0
    end if
    bc1 = 1.d0 - beta1**max(istep,1)
    bc2 = 1.d0 - beta2**max(istep,1)
    ! One fused contiguous pass over the cubes: m, v and w are updated
    ! per element in one sweep (four streams, no temporary).
    do i=0,nt%ndmax
       do j=1,nt%ndmax
          do l=2,nt%nlayer
             gval = g%nabla(l,j,i)/dble(nbatch)
             g%m(l,j,i) = beta1*g%m(l,j,i) + (1.d0-beta1)*gval
             g%v(l,j,i) = beta2*g%v(l,j,i) + (1.d0-beta2)*gval**2
             nt%w(l,j,i) = nt%w(l,j,i) &
                  - eta*( g%m(l,j,i)/bc1 )/( sqrt( g%v(l,j,i)/bc2 ) + eps )
          end do
       end do
    end do
  END SUBROUTINE opt_adam_step

END MODULE train_module
