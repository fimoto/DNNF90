!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (global_variables.f90) is part of DNNF90.
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
MODULE global_variables

  implicit none   ! applies to the whole module

! For input training data
  integer :: NUM_input
  integer :: Ntot_train_set !<=100
  !> One bound for every per-Loss_term array, so that adding a term
  !! cannot outrun one of them.  Every fixed array indexed by a term and
  !! every input check must use it.
  integer,parameter :: MAX_LOSS_TERMS = 100
  integer :: Ndata_train_set(MAX_LOSS_TERMS) ! # of data in each training set
  integer :: label_start(MAX_LOSS_TERMS),label_end(MAX_LOSS_TERMS)
  integer,allocatable :: ind_train(:),ind_validation(:)
  ! Positions in ind_train at which each Loss_term begins and ends, and
  ! how many training points it kept.  ind_train is built in ascending
  ! original index and the sets occupy consecutive blocks of that index,
  ! so each set is a contiguous run of positions.  The minibatch draw is
  ! stratified over these runs.
  integer,allocatable :: set_first(:), set_last(:), nset_train(:)
  !> Per-Loss_term minibatch size.  0 means "take this term's share of
  !! Num_batch, in proportion to its training points"; a positive value
  !! draws exactly that many points from the term every epoch, and the
  !! value may equal the term's point count, which is the usual choice
  !! for a small boundary set that should enter every gradient in full.
  integer :: nbatch_set(MAX_LOSS_TERMS) = 0
  real(8),allocatable :: descriptor_input(:,:)
  real(8),allocatable :: response_input(:,:)
  character(15) :: form_train(MAX_LOSS_TERMS)
  ! NEW
  integer :: NUM_train_special
  integer,allocatable :: ind_train_special(:)

! For output files
  !+++Cost function
  character(40) :: file_history   ! single training-history file
  !
  !
  !+++RMSE
  !
  !
  !+++MAE
  !
  !
  !+++Relative error of integration

! Network structure
  character(15) :: activation_type
  integer :: Nlayer
  integer,allocatable :: ndim(:)
  integer :: ndim_max
  ! Generation of the master weight arrays.  Every write to weight or
  ! weight_best increments it; the library network records the value it
  ! was synchronized with, and every evaluation checks that nothing has
  ! been written since.  A stale library network would otherwise return
  ! plausible numbers computed from the weights of the previous step.
  integer :: weight_gen = 0
  integer :: weight_best_gen = 0
  real(8),allocatable :: weight(:,:,:)
  !---fixed weights

  real(8),allocatable :: zmat(:,:)


! Training setups
  integer :: iswitch_fit
  !---Derivatives
  integer :: iswitch_out_deriv
  integer :: deriv_train_cyc
  integer :: deriv_train_len
  !---Cost function
  character(15) :: init_weight_method
  !> Frequency scale of the periodic initialisation (Init_w_omega)
  real(8) :: init_w_omega = 30.d0
  integer :: iswitch_restart
  integer :: istart_step
  ! Number of steps the Adam moments have actually accumulated.  This is
  ! not the same as istart_step when a run restarts without its optimizer
  ! log: the moments are then cold while the epoch counter is not, and
  ! correcting the bias with the absolute epoch shrinks the first steps
  ! after the restart by an order of magnitude.
  integer :: iadam_step = 0
  !---General setting
  integer :: NUM_loop
  real(8) :: conv_fit
  !NUM_input = NUM_train+NUM_validation
  integer :: NUM_train,NUM_validation
  integer :: NUM_batch !( < NUM_train )
  integer :: io_cyc
  integer :: validation_cyc
  integer :: patience_max !early stopping
  character(30) :: gd_method
  real(8) :: gd_param(0:6)   ! elements 1:5 are the input; 0 and 6 unused
  !---Natural gradient (schedules for learning rate eta and damping mu)
  integer :: NUM_weight
  character(15) :: NGD_schedule_eta,NGD_schedule_mu
  !> How the natural-gradient metric is damped: TRACE for mu*tr(G)*I,
  !! the historical form, or ABS for mu*I.  The trace form is relative
  !! and its meaning changes with the weight count; the absolute form
  !! does not, which matters on the larger networks a coupled system
  !! needs.
  character(15) :: NGD_damping = "TRACE"

  !> Trust-region control of the damping.  When NGD_trust is on, the
  !! damping is raised when a step fails to reduce the loss and lowered
  !! when it succeeds, which is the Levenberg-Marquardt rule: a large
  !! damping makes the step short and gradient-like, a small one makes it
  !! long and Newton-like, and the loss decides which is wanted.
  !!
  !! It matters when the curvature is uneven rather than the gradient.
  !! The adaptive first-order rules divide by the size of the gradient,
  !! so they lengthen the step exactly along the directions where the
  !! surface is steepest, and on a residual built from second derivatives
  !! that is the wrong way round.
  !> Limited-memory BFGS.
  !!
  !! The curvature of the loss is estimated from the last LBFGS_M pairs
  !! (s_k, y_k) = (w_{k+1}-w_k, g_{k+1}-g_k) rather than from a stored
  !! matrix, so the memory is O(m N) and not O(N^2).  That is what makes
  !! it usable on the networks a collocation problem needs, where the
  !! dense metric of a natural gradient does not fit.
  !!
  !! The direction is followed by a backtracking line search on the
  !! Armijo condition, which is what removes the step-size problem: a
  !! step that does not reduce the loss is rejected and shortened rather
  !! than taken.  Every first-order rule tried before this one had to
  !! guess the step in advance.
  integer :: LBFGS_M = 8
  real(8) :: LBFGS_step0 = 1.d0        ! first trial step of the search
  real(8) :: LBFGS_c1 = 1.d-4          ! Armijo constant
  !> Largest weight change the first trial step may make.
  real(8) :: LBFGS_dmax = 0.5d0
  integer :: LBFGS_maxls = 30
  !> Forward-tracking: when the first trial step satisfies Armijo at
  !! once, try doubling it (up to 2^LBFGS_expand times) and keep the
  !! longest step that still satisfies the condition.  A quasi-Newton
  !! direction is scaled for the quadratic model, but on the plateaus a
  !! stiff coupled system produces the model is too cautious and t = 1
  !! is accepted at the first trial epoch after epoch; the search can
  !! then be told to look farther instead of settling.  0 disables it,
  !! which reproduces the pure backtracking search exactly.
  integer :: LBFGS_expand = 0
  !> Self-scaled BFGS (SSBFGS): give every stored pair its own
  !! Oren-Luenberger scaling inside the two-loop recursion instead of
  !! applying the newest pair's scaling once to the initial Hessian.
  logical :: LBFGS_selfscale = .false.
  !> Curvature (second Wolfe) constant c2 of the line search; 0 = off,
  !! which leaves the pure Armijo backtracking search.
  real(8) :: LBFGS_wolfe = 0.d0
  !> Print each trial of the line search, for diagnosis.
  logical :: lbfgs_verbose = .false.
  !> Exhaustive scan of the step, for diagnosis.
  logical :: lbfgs_scan = .false.          ! backtracks before giving up

  logical :: NGD_trust = .false.
  !> Solve the natural-gradient step through the Gram (dual) matrix,
  !! set by Ngd_dual.  Identical step, N_batch-sized solve.
  logical :: NGD_dual = .false.
  !> Geodesic acceleration of the natural-gradient step (Transtrum &
  !! Sethna): a second-order correction along the step direction,
  !! obtained from one extra residual evaluation and a second solve
  !! against the SAME metric.  Ngd_geo h alpha /: h is the relative
  !! finite-difference step along delta1, alpha the acceptance bound on
  !! |delta2|/|delta1| (step rejected back to plain NGD above it).
  logical :: NGD_geo = .false.
  real(8) :: ngd_geo_h = 0.1d0
  real(8) :: ngd_geo_alpha = 0.75d0
  real(8) :: NGD_trust_mu = 1.d0     ! current damping, updated each step
  real(8) :: NGD_trust_lo = 1.d-6    ! floor
  real(8) :: NGD_trust_hi = 1.d8     ! ceiling
  real(8) :: NGD_prev_cost = -1.d0   ! loss before the previous step
  real(8) :: ngd_param_eta(2),ngd_param_mu(2)
  real(8) :: ngd_eta_bound,ngd_mu_bound
  real(8) :: gd_ratio(MAX_LOSS_TERMS)
  !---Biasing evaluation
  real(8),allocatable :: weight_best(:,:,:)
  integer :: epoch_best

  !---Regularizations
  !---selective regularization


! High-order derivative (HOD) training: T^{(l,alpha)}, S^{(l,alpha)} and adjoints
! (multi-index extension of the first-order adjoint planes; see multi_index_bell_module)
  integer :: hod_kmax_in            ! max derivative order K (0 -> HOD off)
  integer :: iswitch_hod_check      ! 1 -> run self-tests before fitting
  integer :: iswitch_hod_dense      ! 1 -> force the dense set |alpha|<=K (timing)
  integer :: rand_seed_in           ! >0 -> fixed RNG seed (reproducible runs)
  real(8) :: lambda_hod(0:15)       ! per-order loss weights lambda_p
  character(40) :: hod_alpha_file   ! 'NONE' or file with closure seeds
  real(8),allocatable :: hod_target_input(:,:)  ! (NUM_input,NUM_alpha) targets y_alpha
  ! HOD arrays with the layer index last, so that the (neuron, multi-index)
  ! plane of one layer is contiguous and the kernels index it in place.
                                           ! h(l,j,ia)=sum_q sigma^{(q+1)} B_{ia,q}
                                           ! fused adjoint cache

! PINN residual training: R = sum_k c_k * P_k with
!   P_k = d^alpha u            (pinn_nonlin=.false.)
!   P_k = u * d^alpha u        (pinn_nonlin=.true. ; advection-type)
! (terms parsed from the Residual block; DXLAP macros already expanded)
  ! Source term of the residual: a value per collocation point, read
  ! from the collocation file.  It enters R additively, so it changes
  ! the residual but not dR/dT, and therefore not the seed.
  ! Which input axis is time.  0 means "not given", resolved to the last
  ! axis, which is the convention the shipped cases follow.  DXLAP needs
  ! it: the Laplacian runs over the axes that are not time.
  ! Committee (ensemble) evaluation: the member weight files.  The
  ! members are independently trained networks, so the trainer does not
  ! produce them; it consumes them.
  integer :: n_committee = 0
  character(80),allocatable :: committee_file(:)
  integer :: pinn_time_axis = 0
  ! DXLAP requests, expanded after the whole input has been read so that
  ! Time_axis takes effect wherever in the file it appears.
  integer :: pinn_ndxl = 0
  real(8) :: pinn_dxl_coeff(16)
  integer :: pinn_dxl_pow(16), pinn_dxl_axis(16)
  logical :: pinn_has_src = .false.
  real(8) :: pinn_src_coeff = 0.d0
  real(8),allocatable :: pinn_src(:)
  integer :: pinn_nterm
  integer :: iswitch_pinn_exact  ! 1 -> collocation file has an exact-u column
  real(8) :: pinn_coeff(64)
  logical :: pinn_nonlin(64)

  !> A system of residuals over several field components.
  !!
  !! Term k belongs to residual sys_res(k) and reads
  !!   sys_coeff(k) * T(sys_cmp(k), sys_ind(k))                if sys_fac(k)=0
  !!   sys_coeff(k) * T(sys_fac(k),1) * T(sys_cmp(k),sys_ind(k)) otherwise
  !! so a term may multiply one component by a derivative of another,
  !! which is what an advection term or a body force needs and what the
  !! scalar form above cannot express.
  integer :: sys_nres = 0
  integer :: sys_nterm = 0
  integer :: sys_res(256)  = 0     ! which residual the term belongs to
  integer :: sys_cmp(256)  = 0     ! component that is differentiated
  integer :: sys_ind(256)  = 0     ! slot of that derivative
  !> A term is a product of up to SYS_MAXFAC derivative factors,
  !!
  !!     coeff * prod_m  d^(beta_m) u_(comp_m)
  !!
  !! with sys_nfac(k) factors in term k.  One factor is a linear term,
  !! two covers advection and a body force, three covers a compressible
  !! momentum flux and most multi-species transport, and four covers the
  !! quartic terms of the higher dispersive hierarchies: the seventh-order
  !! Lax equation carries 140 u^3 u_x, a product of four factors.  The
  !! old fields sys_cmp/sys_ind and sys_fac/sys_fac_ind are factors one
  !! and two, so every input written before this generalisation still
  !! means what it did.
  integer,parameter :: SYS_MAXFAC = 4
  integer :: sys_nfac(256) = 1
  integer :: sys_third(256) = 0    ! component of a third factor, 0 if none
  integer :: sys_fourth(256) = 0   ! component of a fourth factor, 0 if none
  integer :: sys_fcomp(SYS_MAXFAC,256) = 0   ! component of each factor
  integer :: sys_find(SYS_MAXFAC,256)  = 1   ! slot of each factor

  integer :: sys_fac(256)  = 0     ! component multiplying it, 0 if none
  !> Slot of the derivative taken of the multiplying component.  One is
  !! the value itself, which is the common case and what XUX writes; any
  !! other slot makes the term a product of two derivatives, which the
  !! divergence of a flux needs: div(rho grad phi) expands to
  !! grad(rho).grad(phi) + rho lap(phi), and the first of those is a
  !! product of two first derivatives of different components.
  integer :: sys_fac_ind(256) = 1
  real(8) :: sys_coeff(256) = 0.d0
  integer,allocatable :: sys_alpha(:,:), sys_beta(:,:)
  integer,allocatable :: sys_gamma(:,:)   ! multi-index of a third factor
  integer,allocatable :: sys_delta(:,:)   ! multi-index of a fourth factor
  logical :: sys_has_src(16) = .false.
  real(8) :: sys_src_coeff(16) = 0.d0

  !> Weight of each field component in the supervised loss, set by
  !! Sys_wcomp.  Defaults to one, which is the right choice when the
  !! components are of comparable size and the wrong one when they are
  !! not: a component ten times smaller than another contributes a
  !! hundred times less to an unweighted sum of squares.
  real(8) :: sys_wcomp(16) = 1.d0

  !> Weight of each residual in the collocation loss, set by Sys_wres.
  !! The residuals of a system are different equations and need not be
  !! of comparable size: one may be a Poisson equation of order one and
  !! another a transport equation carrying a mobility of 0.5 and a
  !! diffusivity of 0.1.  An unweighted sum of squares is then dominated
  !! by whichever happens to be largest, exactly as an unweighted sum
  !! over field components is.
  real(8) :: sys_wres(16) = 1.d0

  !> Automatic balancing of the residual weights.
  !!
  !! The terms of a composite loss compete: if one residual produces a
  !! gradient ten times larger than another, it sets the step size and
  !! the other makes no progress.  With Sys_balance on, the weights are
  !! rescaled every Sys_balance_cyc epochs so that every residual
  !! contributes a gradient of comparable size,
  !!
  !!     w_r <- (1-a) w_r + a * max_s |grad L_s| / |grad L_r| ,
  !!
  !! which is the learning-rate annealing of Wang, Teng and Perdikaris
  !! applied per residual rather than per loss term.  The running average
  !! keeps the weights from chasing the noise of one batch.
  logical :: sys_balance = .false.
  integer :: sys_balance_cyc = 100
  real(8) :: sys_balance_alpha = 0.1d0

  !> Whether the collocation file of a system carries a source column per
  !! residual.  A homogeneous system carries none.
  logical :: sys_use_src = .false.

  !> Observation noise of each residual, for the extended Kalman filter,
  !! set by Sys_rnoise.  A residual whose source is much larger than the
  !! others produces large innovations, and the filter, which trusts
  !! every observable equally, then spends its covariance on that one.
  !! Raising its noise says the observation is less reliable.
  real(8) :: sys_rnoise(16) = 1.d0

  !> Innovation gate of the extended Kalman filter, set by Kalman_gate.
  !! 0 disables it.  See kalman_module for the rule; the short version:
  !! an observation whose innovation exceeds gate standard deviations of
  !! the filter's own predicted spread has its noise inflated until it
  !! does not, which bounds every rank-1 step.  This is the classical
  !! robust-EKF remedy for divergence under large innovations, which is
  !! exactly what the source-carrying residuals of a system produce.
  real(8) :: kalman_gate = 0.d0

  !> Process noise of the extended Kalman filter, set by Kalman_q.
  !! Added to the diagonal of P after every update.  0 = off.
  real(8) :: kalman_q = 0.d0

  !> Whether the filter carries a dense covariance or the node-wise
  !! block-diagonal one (Kalman_mode DENSE|DECOUPLED).
  logical :: kalman_decoupled = .false.

  !> Relinearizations per observation (IEKF), set by Kalman_iter.
  !! 1 = the plain EKF path, byte for byte.
  integer :: kalman_iter = 1

  !> Source value of every residual at every collocation point, read
  !! from the columns that follow the coordinates.
  real(8),allocatable :: sys_src_input(:,:)
  integer,allocatable :: pinn_alpha(:,:)  ! (ndim(1),64)
  integer :: pinn_ind(64)                 ! multi-index table index (set in read_data)

! MPI
  integer :: average_cyc_mpi
  integer :: iswitch_shuffle
  integer :: i_epoch_now = 0     ! absolute epoch of the next checkpoint
  integer :: shuffle_cyc_mpi
  real(8),allocatable :: weight_recv(:,:,:)

CONTAINS

  !> Release everything this module allocates, so that a host can run
  !! init -> free -> init in one process.  Idempotent: every deallocate
  !! is guarded, so calling it twice, or before any init, is harmless.

  !> Put every scalar and fixed-size variable of this module back to the
  !! value it is declared with.  global_free releases the allocations;
  !! this restores the state that has no allocation, so that a host can
  !! run init -> free -> init with a DIFFERENT configuration.  Without
  !! it, sys_nterm, the NGD flags, the committee count and the rest
  !! survive into the next case and describe a problem that is no longer
  !! there.  Generated from the declarations above; keep it in step when
  !! adding state.
  SUBROUTINE reset_global_state
    implicit none
    nbatch_set = 0
    weight_gen = 0
    weight_best_gen = 0
    init_w_omega = 30.d0
    iadam_step = 0
    NGD_damping = "TRACE"
    LBFGS_M = 8
    LBFGS_step0 = 1.d0
    LBFGS_c1 = 1.d-4
    LBFGS_dmax = 0.5d0
    LBFGS_maxls = 30
    LBFGS_expand = 0
    LBFGS_selfscale = .false.
    LBFGS_wolfe = 0.d0
    lbfgs_verbose = .false.
    lbfgs_scan = .false.
    NGD_trust = .false.
    NGD_dual = .false.
    NGD_geo = .false.
    ngd_geo_h = 0.1d0
    ngd_geo_alpha = 0.75d0
    NGD_trust_mu = 1.d0
    NGD_trust_lo = 1.d-6
    NGD_trust_hi = 1.d8
    NGD_prev_cost = -1.d0
    n_committee = 0
    pinn_time_axis = 0
    pinn_ndxl = 0
    pinn_has_src = .false.
    pinn_src_coeff = 0.d0
    sys_nres = 0
    sys_nterm = 0
    sys_res = 0
    sys_cmp = 0
    sys_ind = 0
    sys_nfac = 1
    sys_third = 0
    sys_fourth = 0
    sys_fcomp = 0
    sys_find = 1
    sys_fac = 0
    sys_fac_ind = 1
    sys_coeff = 0.d0
    sys_has_src = .false.
    sys_src_coeff = 0.d0
    sys_wcomp = 1.d0
    sys_wres = 1.d0
    sys_balance = .false.
    sys_balance_cyc = 100
    sys_balance_alpha = 0.1d0
    sys_use_src = .false.
    sys_rnoise = 1.d0
    kalman_gate = 0.d0
    kalman_q = 0.d0
    kalman_decoupled = .false.
    kalman_iter = 1
    i_epoch_now = 0
  END SUBROUTINE reset_global_state

  SUBROUTINE global_free
    implicit none
    if ( allocated(ind_train) ) deallocate( ind_train )
    if ( allocated(set_first) ) deallocate( set_first )
    if ( allocated(set_last) ) deallocate( set_last )
    if ( allocated(nset_train) ) deallocate( nset_train )
    if ( allocated(ind_validation) ) deallocate( ind_validation )
    if ( allocated(descriptor_input) ) deallocate( descriptor_input )
    if ( allocated(response_input) ) deallocate( response_input )
    if ( allocated(ind_train_special) ) deallocate( ind_train_special )
    if ( allocated(ndim) ) deallocate( ndim )
    if ( allocated(weight) ) deallocate( weight )
    if ( allocated(zmat) ) deallocate( zmat )
    if ( allocated(weight_best) ) deallocate( weight_best )
    if ( allocated(hod_target_input) ) deallocate( hod_target_input )
    if ( allocated(committee_file) ) deallocate( committee_file )
    if ( allocated(pinn_src) ) deallocate( pinn_src )
    if ( allocated(sys_alpha) ) deallocate( sys_alpha )
    if ( allocated(sys_beta) ) deallocate( sys_beta )
    if ( allocated(sys_gamma) ) deallocate( sys_gamma )
    if ( allocated(sys_delta) ) deallocate( sys_delta )
    if ( allocated(sys_src_input) ) deallocate( sys_src_input )
    if ( allocated(pinn_alpha) ) deallocate( pinn_alpha )
    if ( allocated(weight_recv) ) deallocate( weight_recv )
    call reset_global_state
  END SUBROUTINE global_free

END MODULE global_variables

