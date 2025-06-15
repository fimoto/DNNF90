! -----------------------------------------------------------------------
! This file (lib_net_module.f90) is part of DNNF90.
! (MIT License; see LICENSE at the repository root.)
!
! The trainer's library network.
!
! Every forward and backward pass of the trainer runs on the library
! kernels; this module is the only place where the two meet.  It holds
! the library objects (tables, network, work spaces, gradient
! accumulator), mirrors the trainer's weight array into the library
! network, and exposes the four operations the trainer needs:
!
!   lnet_forward_value  value of one point
!   lnet_forward_hod    value and all carried derivatives of one point,
!                       returned in the caller's buffer
!   lnet_seed_grad      adjoint of an arbitrary seed dL/dT, accumulated
!   lnet_seed_row       adjoint of an arbitrary seed, written to a row
!   lnet_batch_grad     batched value fit of a whole minibatch
!
! The trainer keeps its own weight array because the weights are also
! policy: checkpoints, restarts, MPI averaging and the optimizer states
! are written in terms of it.  The mirror costs one copy per batch
! iteration and keeps the optimizer and the file formats independent of
! the library's internal layout.
! -----------------------------------------------------------------------
MODULE lib_net_module
  use global_variables, only: kalman_gate, kalman_q, kalman_decoupled, sys_wcomp,  Nlayer, ndim, ndim_max, Activation_type, &
       iswitch_out_deriv, weight_gen, weight_best_gen
  use multi_index_bell_module, only: tabset_t, tabset_init, tabset_free, &
       tabset_from_current, hod_tables_ready, NUM_alpha
  use net_module, only:  net_t, net_init, net_free
  use kalman_module, only: kalman_t, kf_init, kf_free, kf_update, &
       kf_update_resid_multi, kf_iekf_begin, kf_iekf_iter, &
       kf_update_resid, kf_num_weights
  use train_module, only: net_backward_point_multi, &
       net_forward_point_multi, grad_zero, &
       twork_t, grad_t, bwork_t, twork_init, twork_free, &
       grad_init, grad_free, bwork_init, bwork_free, &
       net_forward_point, net_backward_point, net_grad_batch
  implicit none
  private
  public :: lnet_forward_hod_multi, lnet_seed_grad_multi
  public :: lnet_value_grad_multi, lnet_seed_row_multi
  public :: lnet_kalman_resid_multi
  public :: lnet_kalman_iekf_begin, lnet_kalman_iekf_iter
  public :: lnet_sync_weights, lnet_sync_best, lnet_free
  public :: lnet_forward_value, lnet_forward_hod, lnet_nalpha
  public :: lnet_seed_grad, lnet_seed_row, lnet_value_row, lnet_value_grad
  public :: lnet_batch_grad
  public :: lnet_kalman_init, lnet_kalman_slot, lnet_kalman_resid
  public :: lnet_kalman_active, lnet_export_weights, lnet_nweights
  public :: lnet_kalman_ngated

  type(tabset_t),save :: ts_l
  type(net_t),save    :: nt_l
  type(twork_t),save  :: tw_l
  type(grad_t),save   :: g_l
  !> Buffers of the multi-component path: one row per field component.
  real(8),allocatable,save :: tmul_l(:,:), smul_l(:,:)
  type(bwork_t),save  :: bw_l
  type(kalman_t),save :: kf_l
  logical,save :: kf_ready = .false.
  real(8),allocatable,save :: seed_l(:), tout_l(:)
  logical,save :: net_ready = .false.
  integer,save :: synced_gen = -1
  integer,save :: synced_src = 0   ! 1 = weight, 2 = weight_best

CONTAINS

  !> Build the library network matching the trainer.  When the trainer
  !! has already built the shared high-order tables (a high-order or
  !! collocation run), the network snapshots them instead of building
  !! its own: a second init_hod_tables would rebuild the shared tables
  !! underneath the residual evaluator.  For a plain value fit no such
  !! tables exist and a minimal one-index set is built, which carries
  !! the value alone.
  SUBROUTINE lnet_init
    implicit none
    integer :: seed0(ndim(1),1)
    if ( hod_tables_ready ) then
       call tabset_from_current( ts_l )
    else if ( iswitch_out_deriv /= 0 ) then
       ! A plain value fit that is asked for its derivatives: the dense
       ! first-order set carries the value and every dN/dx_i.
       seed0 = 0
       call tabset_init( ts_l, ndim(1), 1, 0, seed0 )
    else
       ! Pure value fitting: the closure of the zero multi-index is one
       ! index, so the tables carry the value alone.
       seed0 = 0
       call tabset_init( ts_l, ndim(1), 1, 1, seed0 )
    end if
    ! hand the activation to the table set: the engine reads it there
    select case ( trim(Activation_type) )
    case ( "SIN" );  ts_l%iact = 1
    case ( "ERF" );  ts_l%iact = 2
    case ( "BESSEL" ); ts_l%iact = 3
    case ( "BESSEL1" ); ts_l%iact = 4
    case default;    ts_l%iact = 0
    end select
    call net_init( nt_l, Nlayer, ndim(1:Nlayer), ts_l )
    call twork_init( tw_l, nt_l )
    call grad_init( g_l, nt_l )
    allocate( seed_l(ts_l%na), tout_l(ts_l%na) )
    allocate( tmul_l(nt_l%ndim(nt_l%nlayer),ts_l%na) )
    allocate( smul_l(nt_l%ndim(nt_l%nlayer),ts_l%na) )
    tmul_l = 0.d0;  smul_l = 0.d0
    seed_l = 0.d0
    net_ready = .true.
  END SUBROUTINE lnet_init

  !> Mirror the trainer weights into the library network.
  SUBROUTINE lnet_sync_weights( w )
    implicit none
    real(8),intent(IN) :: w(:,:,0:)
    integer :: l,nd,ndm
    if ( .not. net_ready ) call lnet_init
    do l=2,Nlayer
       nd  = ndim(l)
       ndm = ndim(l-1)
       nt_l%w(l,1:nd,0:ndm) = w(l,1:nd,0:ndm)
    end do
    synced_gen = weight_gen
    synced_src = 1
  END SUBROUTINE lnet_sync_weights

  !> Same, for the best-weight array, which has its own generation.
  SUBROUTINE lnet_sync_best( w )
    implicit none
    real(8),intent(IN) :: w(:,:,0:)
    integer :: l,nd,ndm
    if ( .not. net_ready ) call lnet_init
    do l=2,Nlayer
       nd  = ndim(l)
       ndm = ndim(l-1)
       nt_l%w(l,1:nd,0:ndm) = w(l,1:nd,0:ndm)
    end do
    synced_gen = weight_best_gen
    synced_src = 2
  END SUBROUTINE lnet_sync_best

  !> Guard of the one invariant this module carries: the network must
  !! hold the weights as they are now.  The trainer owns the master
  !! array, so a write to it without a following synchronization would
  !! leave every evaluation below silently one step behind.  The check
  !! is one integer comparison, so it stays on in production builds.
  SUBROUTINE fresh_check( cname )
    implicit none
    character(*),intent(IN) :: cname
    if ( synced_src == 0 ) then
       write(*,*) cname, ": the library network has never been given"
       write(*,*) "  any weights.  Call lnet_sync_weights first."
       stop
    end if
    if ( ( synced_src == 1 .and. synced_gen /= weight_gen ) .or. &
         ( synced_src == 2 .and. synced_gen /= weight_best_gen ) ) then
       write(*,*) cname, ": the weights have been written since the"
       if ( synced_src == 1 ) then
          write(*,*) "  library network was last synchronized with weight", &
               " (generation", synced_gen, "vs", weight_gen, ")."
       else
          write(*,*) "  library network was last synchronized with", &
               " weight_best (generation", synced_gen, "vs", weight_best_gen, ")."
       end if
       write(*,*) "  Call lnet_sync_weights with the array you intend to"
       write(*,*) "  evaluate before asking for a value or a gradient."
       stop
    end if
  END SUBROUTINE fresh_check

  !> Value of one point.
  SUBROUTINE lnet_forward_value( x, val )
    implicit none
    real(8),intent(IN) :: x(:)
    real(8),intent(OUT) :: val
    if ( .not. net_ready ) call lnet_init
    call fresh_check( "lnet_forward_value" )
    call net_forward_point( nt_l, tw_l, x, tout_l )
    val = tout_l(1)
  END SUBROUTINE lnet_forward_value

  !> Forward pass of one point, publishing the carried derivatives where
  !! the seed builders and the residual evaluator read them.
  !> Every carried derivative of every field component at one point.
  !!
  !! The multi-component counterpart of lnet_forward_hod: tm(i,ia) is
  !! multi-index ia of component i, which is what a system of residuals
  !! consumes.  The forward pass is the same one; only the extraction
  !! differs, since the recursion already runs over every neuron of the
  !! output layer.
  SUBROUTINE lnet_forward_hod_multi( x, tm )
    implicit none
    real(8),intent(IN) :: x(:)
    real(8),intent(OUT) :: tm(:,:)
    integer :: nout
    if ( .not. net_ready ) call lnet_init
    call fresh_check( "lnet_forward_hod_multi" )
    nout = nt_l%ndim(nt_l%nlayer)
    if ( size(tm,1) < nout .or. size(tm,2) < ts_l%na ) then
       write(*,*) "lnet_forward_hod_multi: the buffer is", size(tm,1), &
            " by", size(tm,2), " but", nout, " by", ts_l%na, " is needed"
       stop
    end if
    call net_forward_point_multi( nt_l, tw_l, x, tm )
  END SUBROUTINE lnet_forward_hod_multi

  !> Accumulate the weight gradient from a seed given per component.
  !!
  !! The multi-component counterpart of lnet_seed_grad.  A cross term of
  !! a system residual seeds two entries, the derivative slot of the
  !! component it differentiates and the value slot of the component
  !! multiplying it, so the seed cannot be collapsed into one vector.
  SUBROUTINE lnet_seed_grad_multi( sm, nabla )
    implicit none
    real(8),intent(IN) :: sm(:,:)
    real(8),intent(INOUT) :: nabla(:,:,0:)
    integer :: lslice, ndslice, ndmslice
    if ( .not. net_ready ) call lnet_init
    call fresh_check( "lnet_seed_grad_multi" )
    call grad_zero( g_l )
    call net_backward_point_multi( nt_l, tw_l, sm, g_l )
    ! Accumulate the live slice only, as the scalar routine does: the
    ! cube is (Nlayer, ndim_max, 0:ndim_max) and the weights of layer l
    ! occupy (l, 1:ndim(l), 0:ndim(l-1)).  Adding the whole cube carries
    ! the dead entries into the gradient.
    do lslice = 2, Nlayer
       ndslice  = ndim(lslice)
       ndmslice = ndim(lslice-1)
       nabla(lslice,1:ndslice,0:ndmslice) = &
            nabla(lslice,1:ndslice,0:ndmslice) &
            + g_l%nabla(lslice,1:ndslice,0:ndmslice)
    end do
  END SUBROUTINE lnet_seed_grad_multi

  SUBROUTINE lnet_forward_hod( x, t )
    implicit none
    real(8),intent(IN) :: x(:)
    real(8),intent(OUT) :: t(:)
    if ( .not. net_ready ) call lnet_init
    call fresh_check( "lnet_forward_hod" )
    if ( size(t) < ts_l%na ) then
       write(*,*) "lnet_forward_hod: the buffer holds", size(t), &
            " values but the tables carry", ts_l%na
       stop
    end if
    call net_forward_point( nt_l, tw_l, x, tout_l )
    t(1:ts_l%na) = tout_l(1:ts_l%na)
  END SUBROUTINE lnet_forward_hod

  !> Number of carried multi-indices, so a caller can size its buffers.
  INTEGER FUNCTION lnet_nalpha()
    implicit none
    if ( .not. net_ready ) call lnet_init
    lnet_nalpha = ts_l%na
  END FUNCTION lnet_nalpha

  !> Adjoint of an arbitrary seed dL/dT, accumulated into the trainer's
  !! gradient cube.  The forward pass of the same point must have gone
  !! through this module first.
  SUBROUTINE lnet_seed_grad( seed, nabla )
    implicit none
    real(8),intent(IN) :: seed(:)
    real(8),intent(INOUT) :: nabla(:,:,0:)
    integer :: l,nd,ndm
    call fresh_check( "lnet_seed_grad" )
    call seed_check( size(seed), "lnet_seed_grad" )
    seed_l(1:ts_l%na) = seed(1:ts_l%na)
    g_l%nabla = 0.d0
    call net_backward_point( nt_l, tw_l, seed_l, g_l )
    do l=2,Nlayer
       nd  = ndim(l)
       ndm = ndim(l-1)
       nabla(l,1:nd,0:ndm) = nabla(l,1:nd,0:ndm) + g_l%nabla(l,1:nd,0:ndm)
    end do
  END SUBROUTINE lnet_seed_grad

  !> Adjoint of an arbitrary seed, written (not accumulated) into a
  !! per-point row, which is what the natural-gradient metric needs.
  !> Metric row of a system residual.
  !!
  !! The Gauss-Newton metric of the natural gradient is built from rows
  !! j_n = dR(x_n)/dw, and nothing in that construction cares how many
  !! field components R is assembled from: the row is the weight gradient
  !! of one residual, obtained by seeding the adjoint with dR/dT and
  !! reading the result.  The only difference from the scalar case is
  !! that the seed is a matrix, one row per component, because a cross
  !! term of the residual seeds two entries.
  !!
  !! A system has several residuals, so the caller supplies the seed of
  !! whichever one the row is for; summing the rows of all residuals
  !! would give the metric of their sum, not of the system.
  SUBROUTINE lnet_seed_row_multi( sm, row )
    implicit none
    real(8),intent(IN) :: sm(:,:)
    real(8),intent(OUT) :: row(:,:,0:)
    integer :: l, nd, ndm
    if ( .not. net_ready ) call lnet_init
    call fresh_check( "lnet_seed_row_multi" )
    g_l%nabla = 0.d0
    call net_backward_point_multi( nt_l, tw_l, sm, g_l )
    ! Copy only the live slice, as the scalar routine does.  The gradient
    ! cube is (Nlayer, ndim_max, 0:ndim_max) and the weights of a layer
    ! occupy (l, 1:ndim(l), 0:ndim(l-1)); the rest is dead.  Copying the
    ! whole cube carries whatever the dead entries hold into the metric,
    ! and the metric is then the one being inverted.
    row = 0.d0
    do l = 2, Nlayer
       nd  = ndim(l)
       ndm = ndim(l-1)
       row(l,1:nd,0:ndm) = g_l%nabla(l,1:nd,0:ndm)
    end do
  END SUBROUTINE lnet_seed_row_multi

  SUBROUTINE lnet_seed_row( seed, row )
    implicit none
    real(8),intent(IN) :: seed(:)
    real(8),intent(OUT) :: row(:,:,0:)
    integer :: l,nd,ndm
    call fresh_check( "lnet_seed_row" )
    call seed_check( size(seed), "lnet_seed_row" )
    seed_l(1:ts_l%na) = seed(1:ts_l%na)
    g_l%nabla = 0.d0
    call net_backward_point( nt_l, tw_l, seed_l, g_l )
    row = 0.d0
    do l=2,Nlayer
       nd  = ndim(l)
       ndm = ndim(l-1)
       row(l,1:nd,0:ndm) = g_l%nabla(l,1:nd,0:ndm)
    end do
  END SUBROUTINE lnet_seed_row

  !> Jacobian row of the value output: unit seed on the value component
  !! and zero on every carried derivative.  Owned here because the seed
  !! length is the number of carried multi-indices, not one.
  SUBROUTINE lnet_value_row( row )
    implicit none
    real(8),intent(OUT) :: row(:,:,0:)
    integer :: l,nd,ndm
    call fresh_check( "lnet_value_row" )
    seed_l(1:ts_l%na) = 0.d0
    seed_l(1) = 1.d0
    g_l%nabla = 0.d0
    call net_backward_point( nt_l, tw_l, seed_l, g_l )
    row = 0.d0
    do l=2,Nlayer
       nd  = ndim(l)
       ndm = ndim(l-1)
       row(l,1:nd,0:ndm) = g_l%nabla(l,1:nd,0:ndm)
    end do
  END SUBROUTINE lnet_value_row

  !> Weight gradient of one value-fitting point:
  !!   seed = coef*( N(x) - y ) on the value component, zero elsewhere.
  !! The seed buffer is shared with the high-order paths, so the whole
  !! vector is cleared: a previous collocation point would otherwise
  !! leak its derivative seed into this gradient.
  !> Value fit of every field component at one point.
  !!
  !! The multi-component counterpart of lnet_value_grad.  The loss is
  !!
  !!     L = (coef/2) sum_i ( u_i(x) - y_i )^2
  !!
  !! so the seed carries coef*(u_i - y_i) in the value slot of component
  !! i and nothing elsewhere.  The scalar routine is the case of one
  !! component; using it for a system fits the first component and leaves
  !! the others unconstrained, which is silent and looks like a failure
  !! to converge.
  SUBROUTINE lnet_value_grad_multi( x, y, coef, nabla )
    implicit none
    real(8),intent(IN) :: x(:)
    real(8),intent(IN) :: y(:)
    real(8),intent(IN) :: coef
    real(8),intent(INOUT) :: nabla(:,:,0:)
    integer :: nout, i, lslice, ndslice, ndmslice
    if ( .not. net_ready ) call lnet_init
    call fresh_check( "lnet_value_grad_multi" )
    nout = nt_l%ndim(nt_l%nlayer)
    if ( size(y) < nout ) then
       write(*,*) "lnet_value_grad_multi: ", size(y), " targets for ", &
            nout, " components"
       stop
    end if
    call net_forward_point_multi( nt_l, tw_l, x, tmul_l )
    smul_l = 0.d0
    do i = 1, nout
       ! Sys_wcomp(i) rescales component i.  Without it the loss is
       ! dominated by whichever component happens to be numerically
       ! largest: on the Taylor-Green vortex the pressure is half the
       ! size of the velocities and is left at a coefficient of
       ! determination near zero while they reach 0.95.  Setting
       ! sys_wcomp = 1/||y_i||^2 puts every component on the same footing.
       smul_l(i,1) = coef*sys_wcomp(i)*( tmul_l(i,1) - y(i) )
    end do
    g_l%nabla = 0.d0
    call net_backward_point_multi( nt_l, tw_l, smul_l, g_l )
    ! Accumulate the live slice only, as the scalar routine does: the
    ! cube is (Nlayer, ndim_max, 0:ndim_max) and the weights of layer l
    ! occupy (l, 1:ndim(l), 0:ndim(l-1)).  Adding the whole cube carries
    ! the dead entries into the gradient.
    do lslice = 2, Nlayer
       ndslice  = ndim(lslice)
       ndmslice = ndim(lslice-1)
       nabla(lslice,1:ndslice,0:ndmslice) = &
            nabla(lslice,1:ndslice,0:ndmslice) &
            + g_l%nabla(lslice,1:ndslice,0:ndmslice)
    end do
  END SUBROUTINE lnet_value_grad_multi

  SUBROUTINE lnet_value_grad( x, y, coef, nabla )
    implicit none
    real(8),intent(IN) :: x(:)
    real(8),intent(IN) :: y
    real(8),intent(IN) :: coef
    real(8),intent(INOUT) :: nabla(:,:,0:)
    integer :: l,nd,ndm
    if ( .not. net_ready ) call lnet_init
    call fresh_check( "lnet_value_grad" )
    call net_forward_point( nt_l, tw_l, x, tout_l )
    seed_l = 0.d0
    seed_l(1) = coef*( tout_l(1) - y )
    g_l%nabla = 0.d0
    call net_backward_point( nt_l, tw_l, seed_l, g_l )
    do l=2,Nlayer
       nd  = ndim(l)
       ndm = ndim(l-1)
       nabla(l,1:nd,0:ndm) = nabla(l,1:nd,0:ndm) + g_l%nabla(l,1:nd,0:ndm)
    end do
  END SUBROUTINE lnet_value_grad

  !> Batched value fit of a whole minibatch.
  SUBROUTINE lnet_batch_grad( X, Y, coef, nabla )
    implicit none
    real(8),intent(IN) :: X(:,:)
    real(8),intent(IN) :: Y(:)
    real(8),intent(IN) :: coef
    real(8),intent(INOUT) :: nabla(:,:,0:)
    integer :: l,nd,ndm
    if ( .not. net_ready ) call lnet_init
    call fresh_check( "lnet_batch_grad" )
    if ( bw_l%nb /= size(Y) ) then
       if ( kf_ready ) then
       call kf_free( kf_l );  kf_ready = .false.
    end if
    call bwork_free( bw_l )
       call bwork_init( bw_l, nt_l, size(Y) )
    end if
    g_l%nabla = 0.d0
    call net_grad_batch( nt_l, bw_l, X, Y, coef, g_l )
    do l=2,Nlayer
       nd  = ndim(l)
       ndm = ndim(l-1)
       nabla(l,1:nd,0:ndm) = nabla(l,1:nd,0:ndm) + g_l%nabla(l,1:nd,0:ndm)
    end do
  END SUBROUTINE lnet_batch_grad

  !> Start the extended Kalman filter on this network.  Unlike the
  !! gradient methods, the filter writes the weights itself, so the
  !! trainer reads them back with lnet_export_weights after each pass.
  SUBROUTINE lnet_kalman_init( p0, lam, lam0 )
    implicit none
    real(8),intent(IN) :: p0, lam, lam0
    if ( .not. net_ready ) call lnet_init
    if ( kf_ready ) call kf_free( kf_l )
    if ( .not. kalman_decoupled .and. &
         dble(kf_num_weights(nt_l))**2*8.d0 > 2.d9 ) then
       write(*,*) "Kalman: the covariance of", kf_num_weights(nt_l), &
            " weights would need", &
            dble(kf_num_weights(nt_l))**2*8.d-9, " GB."
       write(*,*) "  This method is for small networks updated few times"
       write(*,*) "  per pattern; use a gradient method at this size."
       stop
    end if
    call kf_init( kf_l, nt_l, p0, lam, lam0, kalman_decoupled )
    kf_l%gate = kalman_gate
    kf_l%q    = kalman_q
    if ( kalman_decoupled ) &
         write(*,'(a,i0,a)') "### Kalman: decoupled (node-wise) filter, ", &
         kf_l%nb, " blocks"
    if ( kalman_q > 0.d0 ) &
         write(*,'(a,es10.2)') "### Kalman process noise q = ", kalman_q
    if ( kalman_gate > 0.d0 ) &
         write(*,'(a,f8.3)') "### Kalman innovation gate: ", kalman_gate
    kf_ready = .true.
  END SUBROUTINE lnet_kalman_init

  !> Gated-update count of the running filter, for the end-of-run report.
  INTEGER FUNCTION lnet_kalman_ngated()
    implicit none
    lnet_kalman_ngated = 0
    if ( kf_ready ) lnet_kalman_ngated = kf_l%ngated
  END FUNCTION lnet_kalman_ngated

  LOGICAL FUNCTION lnet_kalman_active()
    implicit none
    lnet_kalman_active = kf_ready
  END FUNCTION lnet_kalman_active

  !> Number of weights, which is the side of the filter's covariance.
  INTEGER FUNCTION lnet_nweights()
    implicit none
    if ( .not. net_ready ) call lnet_init
    lnet_nweights = kf_num_weights( nt_l )
  END FUNCTION lnet_nweights

  !> Observe one carried derivative of one point: slot 1 is the value
  !! itself, so a plain value fit uses islot = 1.
  SUBROUTINE lnet_kalman_slot( x, islot, target )
    implicit none
    real(8),intent(IN) :: x(:)
    integer,intent(IN) :: islot
    real(8),intent(IN) :: target
    call fresh_check( "lnet_kalman_slot" )
    call kf_update( nt_l, tw_l, g_l, kf_l, x, islot, target )
  END SUBROUTINE lnet_kalman_slot

  !> Observe a collocation residual, whose target is zero.  The seed is
  !! dR/dT and resid is R, both built by the caller.
  !> Filter update from one residual of a system.
  !!
  !! The seed carries dR/dT for every field component; the caller visits
  !! the residuals of the system one at a time, which is what keeps the
  !! update rank one.
  SUBROUTINE lnet_kalman_resid_multi( x, sm, resid, rnoise )
    implicit none
    real(8),intent(IN) :: x(:)
    real(8),intent(IN) :: sm(:,:)
    real(8),intent(IN) :: resid
    real(8),intent(IN),optional :: rnoise
    call fresh_check( "lnet_kalman_resid_multi" )
    if ( present(rnoise) ) then
       call kf_update_resid_multi( nt_l, tw_l, g_l, kf_l, x, sm, resid, rnoise )
    else
       call kf_update_resid_multi( nt_l, tw_l, g_l, kf_l, x, sm, resid )
    end if
  END SUBROUTINE lnet_kalman_resid_multi

  !> Iterated update, system form: begin stores the prior, then one
  !! iter per relinearization.  The caller re-evaluates the fields and
  !! the seed at the current weights between calls; this wrapper only
  !! adds the adjoint row on top, exactly as the plain entry does.
  SUBROUTINE lnet_kalman_iekf_begin()
    implicit none
    if ( .not. kf_ready ) then
       write(*,*) "lnet_kalman_iekf_begin: filter not initialized"
       stop
    end if
    call kf_iekf_begin( nt_l, kf_l )
  END SUBROUTINE lnet_kalman_iekf_begin

  SUBROUTINE lnet_kalman_iekf_iter( x, sm, resid, rnoise, final )
    implicit none
    real(8),intent(IN) :: x(:)
    real(8),intent(IN) :: sm(:,:)
    real(8),intent(IN) :: resid
    real(8),intent(IN) :: rnoise
    logical,intent(IN) :: final
    call fresh_check( "lnet_kalman_iekf_iter" )
    call kf_iekf_iter( nt_l, tw_l, g_l, kf_l, x, sm, resid, rnoise, final )
  END SUBROUTINE lnet_kalman_iekf_iter

  SUBROUTINE lnet_kalman_resid( x, seed, resid )
    implicit none
    real(8),intent(IN) :: x(:)
    real(8),intent(IN) :: seed(:)
    real(8),intent(IN) :: resid
    call fresh_check( "lnet_kalman_resid" )
    call seed_check( size(seed), "lnet_kalman_resid" )
    call kf_update_resid( nt_l, tw_l, g_l, kf_l, x, seed, resid )
  END SUBROUTINE lnet_kalman_resid

  !> Copy the library network's weights back into the trainer's array.
  !! The filter is the one path that moves the weights on its own.
  SUBROUTINE lnet_export_weights( w )
    implicit none
    real(8),intent(OUT) :: w(:,:,0:)
    integer :: l,nd,ndm
    do l=2,Nlayer
       nd  = ndim(l)
       ndm = ndim(l-1)
       w(l,1:nd,0:ndm) = nt_l%w(l,1:nd,0:ndm)
    end do
  END SUBROUTINE lnet_export_weights

  SUBROUTINE seed_check( n, cname )
    implicit none
    integer,intent(IN) :: n
    character(*),intent(IN) :: cname
    if ( n /= ts_l%na ) then
       write(*,*) cname, ": seed length", n, " but the tables carry", ts_l%na
       stop
    end if
  END SUBROUTINE seed_check

  SUBROUTINE lnet_free
    implicit none
    if ( .not. net_ready ) return
    if ( kf_ready ) then
       call kf_free( kf_l )
       kf_ready = .false.
    end if
    call bwork_free( bw_l )
    call grad_free( g_l )
    call twork_free( tw_l )
    call net_free( nt_l )
    call tabset_free( ts_l )
    if ( allocated(seed_l) ) deallocate( seed_l )
    if ( allocated(tmul_l) ) deallocate( tmul_l )
    if ( allocated(smul_l) ) deallocate( smul_l )
    if ( allocated(tout_l) ) deallocate( tout_l )
    ! The guard must not survive the network it describes: after a free
    ! the next evaluation has to be preceded by a synchronization, and
    ! leaving the generation behind would let a zeroed network pass.
    synced_gen = -1
    synced_src = 0
    net_ready  = .false.
  END SUBROUTINE lnet_free

END MODULE lib_net_module
