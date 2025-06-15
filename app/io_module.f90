!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (io_module.f90) is part of DNNF90.
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
!read input/feedforward/output
MODULE io_module

#ifdef _MPI_
  use parallel_module
#endif
  use rand_module,only: pre_random,random_irange
  use global_variables
  use multi_index_bell_module

  use lib_net_module, only: lnet_forward_hod_multi,  lnet_forward_value, lnet_forward_hod, lnet_sync_best, lnet_nalpha
  implicit none

  PRIVATE
  PUBLIC :: read_parameters,read_data,write_data,write_nn_param,write_deriv
  PUBLIC :: check_weight_header

  character(40) :: file_train(10,100) !file_train(ndim(1)+ndim(L),100 materials)

#ifdef USE_BLAS
#endif

CONTAINS

  !> Read the header of a weight file and check it against the run.
  !!
  !! The header records the activation the weights were trained with and
  !! the shape of the network that produced them.  A reader that skips
  !! those lines evaluates a J_0 network with tanh analytics, or reads a
  !! 6-wide layer into a 48-wide one, without saying anything.  On
  !! return the file is positioned at the first layer block.
  SUBROUTINE check_weight_header( ur, fname, istart )
    implicit none
    integer,intent(IN) :: ur
    character(len=*),intent(IN) :: fname
    integer,intent(OUT) :: istart
    character(len=32) :: ckey
    integer :: ifunc, iout, nl, i, nd, iwant
    read(ur,*) istart
    read(ur,*) ckey, ifunc
    read(ur,*) ckey, iout
    read(ur,*) nl
    select case ( trim(Activation_type) )
    case ( "SIN" );     iwant = 1
    case ( "ERF" );     iwant = 2
    case ( "BESSEL" );  iwant = 3
    case ( "BESSEL1" ); iwant = 4
    case default;       iwant = 0
    end select
    if ( ifunc /= iwant ) then
       write(*,*) trim(fname), ": trained with activation code", ifunc, &
            " but this run uses ", trim(Activation_type), " (code", iwant, ")"
       write(*,*) "  codes: 0 TANH, 1 SIN, 2 ERF, 3 BESSEL, 4 BESSEL1"
       stop
    end if
    if ( nl /= Nlayer ) then
       write(*,*) trim(fname), ": the weights are for", nl, &
            " layers, this run has", Nlayer
       stop
    end if
    do i=1,Nlayer
       read(ur,*) nd
       if ( nd /= ndim(i) ) then
          write(*,*) trim(fname), ": layer", i, " is", nd, &
               " wide in the file and", ndim(i), " in this run"
          stop
       end if
    end do
  END SUBROUTINE check_weight_header


  SUBROUTINE read_parameters
    implicit none
    integer :: ui
    integer,parameter :: max_read=1000
    integer :: i, j, itmp, l, jj, ios
    real(8) :: rtmp
    character(30) :: cbuf, ckey, ctmp, ctmp2, ckey2
    character(15) :: cform
    integer :: jax
    !----read from nn.dat----!
    iswitch_fit=1     ! Task TRAIN unless Task PREDICT is given
    gd_param=0.d0              ! GD_param: the all-zero guard below relies
                               ! on this, and module data is undefined
                               ! until it is set
    iswitch_out_deriv=0        ! Output_deriv
    iswitch_restart=0          ! Restart
    Activation_type="TANH"     ! TANH, SIN or ERF
    init_weight_method="NONE"  ! Init_w has no sensible default: it is required
    Ntot_train_set=0
    iswitch_shuffle=0
    Average_cyc_mpi=1
    Shuffle_cyc_mpi=1000000
    gd_ratio=1.d0
    hod_kmax_in=0
    iswitch_hod_check=0
    iswitch_hod_dense=0
    rand_seed_in=0
    lambda_hod=1.d0
    hod_alpha_file='NONE'
    pinn_nterm=0
    iswitch_pinn_exact=0
    NGD_schedule_eta="NONE"
    NGD_schedule_mu="NONE"
    ! Safe defaults for keys a minimal input may omit.  Every shipped
    ! example sets these explicitly, so nothing changes for them; without
    ! defaults an omitted key left the variable uninitialized, which is
    ! undefined behavior (an unset Validation_cyc is a division by zero,
    ! an unset P_max is a garbage patience limit).
    NUM_validation = 0          ! all data trains; see the 0-val guard
    validation_cyc = 1
    patience_max = huge(0)/2    ! patience stopping off unless configured
    conv_fit = 0.d0             ! convergence stopping off unless configured
    io_cyc = 1000
    NUM_batch = -1              ! sentinel: resolved after NUM_input is known
    NUM_LOOP = 1000             ! Epoch
    gd_method = "SIMPLE"
    ngd_param_eta=0.d0
    ngd_param_mu=0.d0
    ngd_eta_bound=0.d0
    ngd_mu_bound=0.d0
    pinn_coeff=0.d0
    pinn_nonlin=.false.
    pinn_ind=0
    
    ui=10
    open(ui,file='input_nn.dat',status='old')
!    ui=5
    do i=1,max_read
       read(ui,*,END=999) cbuf
       call convert_capital(cbuf,ckey)
       if ( ckey == "FIT" ) then
          write(*,*) "input: Fit has been renamed.  Use"
          write(*,*) "    Task TRAIN   (default: fit the weights)"
          write(*,*) "    Task PREDICT (read nn_weight.dat, write the"
          write(*,*) "                  outputs, no training)"
          stop
       else if ( ckey == "TASK" ) then
          backspace(ui)
          read(ui,*) cbuf, ctmp
          call convert_capital(ctmp,ctmp2)
          select case ( trim(ctmp2) )
          case ( "TRAIN" )
             iswitch_fit = 1
          case ( "PREDICT" )
             iswitch_fit = 0
          case ( "COMMITTEE" )
             ! evaluate an ensemble of already trained networks
             iswitch_fit = 2
          case default
             write(*,*) "Task: unknown value ", trim(ctmp), &
                  " (TRAIN, PREDICT or COMMITTEE)"
             stop
          end select
       else if ( ckey == "WEIGHT_UPDATE_TYPE" ) then
          write(*,*) "input: Weight_update_type has been removed (it no"
          write(*,*) "  longer selected anything; the best weights are"
          write(*,*) "  always kept on the training cost)"
          stop
       else if ( ckey == "AVERAGE_CYC_MPI" ) then
          backspace(ui)
          read(ui,*) cbuf, Average_cyc_mpi
       else if ( ckey == "SHUFFLE" ) then
          backspace(ui)
          read(ui,*) cbuf, iswitch_shuffle
#ifdef _MPI_
          if ( iswitch_shuffle /= 0 ) then
             write(*,*) "Shuffle has no effect under MPI: the minibatch is"
             write(*,*) "  drawn per Loss_term within each rank's share, and"
             write(*,*) "  a global permutation would move points between"
             write(*,*) "  terms.  The setting is accepted and ignored."
          end if
#endif
       else if ( ckey == "SHUFFLE_CYC_MPI" ) then
          backspace(ui)
          read(ui,*) cbuf, Shuffle_cyc_mpi
       else if ( ckey == "OUTPUT_DERIV" ) then
          backspace(ui)
          read(ui,*) cbuf, iswitch_out_deriv !1->calc derivatives
!          backspace(ui)
       else if ( ckey == "ACTIVATION_OUT" ) then
          write(*,*) "input: Activation_out has been removed.  The output"
          write(*,*) "  layer is linear: the derivatives the library"
          write(*,*) "  propagates are those of a linear read-out of the"
          write(*,*) "  last hidden layer, and a nonlinear output would"
          write(*,*) "  not be covered by the derivative tables."
          stop
       else if ( ckey == "COST_TYPE" ) then
          write(*,*) "input: Cost_type has been removed.  The cost is the"
          write(*,*) "  plain weighted sum of squares; per-term emphasis"
          write(*,*) "  belongs in the Loss_term weights."
          stop
       else if ( ckey == "RESTART" ) then
          backspace(ui)
          read(ui,*) cbuf, iswitch_restart !1->restart from read (w,b)
       else if ( ckey == "NLAYER" ) then
          backspace(ui)
          read(ui,*) cbuf, Nlayer
          allocate( ndim(Nlayer) )
          do l=1,Nlayer
             read(ui,*) ndim(l)
          end do
!---------------------------------------------------------------!
       else if ( ckey == "LOSS_TERM" ) then
          ! One term of the composite loss, self-contained on one line:
          !   Loss_term  FORM  npoints  filename  weight /
          ! FORM: DATA (supervised values), HOD_DATA (values and all
          ! carried derivatives), COLLOCATION (a cloud of points where the
          ! PDE residual).  The number of terms is counted, not declared.
          Ntot_train_set = Ntot_train_set + 1
          if ( Ntot_train_set > size(form_train) ) then
             write(*,*) "input: more Loss_term lines than the ", &
                  size(form_train), " supported"
             stop
          end if
          backspace(ui)
          ! An optional sixth field gives this term its own minibatch
          ! size.  Without it the term contributes its share of the
          ! global Num_batch, in proportion to its training points.  The
          ! read is attempted with the field and retried without it, so
          ! old input files are unchanged.
          nbatch_set(Ntot_train_set) = 0
          read(ui,*,iostat=ios) cbuf, ctmp, Ndata_train_set(Ntot_train_set), &
               file_train(1,Ntot_train_set), gd_ratio(Ntot_train_set), &
               nbatch_set(Ntot_train_set)
          if ( ios /= 0 ) then
             nbatch_set(Ntot_train_set) = 0
             backspace(ui)
             read(ui,*) cbuf, ctmp, Ndata_train_set(Ntot_train_set), &
                  file_train(1,Ntot_train_set), gd_ratio(Ntot_train_set)
          end if
          if ( nbatch_set(Ntot_train_set) < 0 ) then
             write(*,*) "Loss_term: the batch size must be >= 0 (0 means"
             write(*,*) "  take this term's share of Num_batch)"
             stop
          end if
          if ( gd_ratio(Ntot_train_set) < 0.d0 ) then
             write(*,*) "Loss_term: the weight must be >= 0 (a negative"
             write(*,*) "  weight makes the objective unbounded below)"
             stop
          end if
          call convert_capital(ctmp,cform)
          select case ( trim(cform) )
          case ( "DATA" )
             form_train(Ntot_train_set) = "MATH"
          case ( "HOD_DATA" )
             form_train(Ntot_train_set) = "MATH_HOD"
          case ( "COLLOCATION" )
             form_train(Ntot_train_set) = "PINN"
          case default
             write(*,*) "Loss_term: unknown form ", trim(cform), &
                  " (DATA, HOD_DATA or COLLOCATION)"
             stop
          end select
       else if ( ckey == "DERIV_CHANNEL" ) then
          write(*,*) "input: Deriv_channel has been removed.  Fitting a"
          write(*,*) "  first derivative is the K = 1 case of high-order"
          write(*,*) "  fitting, so use a HOD_DATA term instead:"
          write(*,*) "    Hod_K  1 /"
          write(*,*) "    <lambda_0>"
          write(*,*) "    <lambda_1>"
          write(*,*) "    Loss_term  HOD_DATA  <n>  <file>  <weight> /"
          write(*,*) "  with one record per point holding x, y and dy/dx."
          stop
       else if ( ckey == "NTRAIN_SET" ) then
          write(*,*) "input: Ntrain_set has been replaced.  Each term of"
          write(*,*) "  the loss is now one self-contained line,"
          write(*,*) "    Loss_term  DATA|HOD_DATA|COLLOCATION  npoints  file  weight /"
          write(*,*) "  and the number of terms is counted, not declared."
          stop
       else if ( ckey == "DERIV_INFO" ) then
          write(*,*) "input: Deriv_info has been replaced by"
          write(*,*) "    Deriv_channel  term_index  lambda  file /"
          stop
       else if ( ckey == "SPECIAL_TRAIN_DATA" ) then
          write(*,*) "input: Special_train_data has been removed.  An"
          write(*,*) "  emphasis region is a separate Loss_term with its"
          write(*,*) "  own points and weight."
          stop
       else if ( ckey == "NUM_VALIDATION" ) then
          backspace(ui)
          read(ui,*) cbuf, NUM_validation
       else if ( ckey == "VALIDATION_CYC" ) then
          backspace(ui)
          read(ui,*) cbuf, validation_cyc
       else if ( ckey == "P_MAX" ) then
          backspace(ui)
          read(ui,*) cbuf, patience_max
!---------------------------------------------------------------!
       else if ( ckey == "EPOCH" ) then
          backspace(ui)
          read(ui,*) cbuf, NUM_LOOP
       else if ( ckey == "ACTIVATION" ) then
          backspace(ui)
          read(ui,*) cbuf, ctmp
          call convert_capital(ctmp,activation_type)
          if ( trim(activation_type) /= "TANH" .and. &
               trim(activation_type) /= "SIN"  .and. &
               trim(activation_type) /= "ERF" .and. &
               trim(activation_type) /= "BESSEL" .and. &
               trim(activation_type) /= "BESSEL1" ) then
             write(*,*) "Activation: unknown value ", trim(ctmp), &
                  "  (TANH, SIN, ERF, BESSEL or BESSEL1)"
             stop
          end if
       else if ( ckey == "CONV" ) then
          backspace(ui)
          read(ui,*) cbuf, conv_fit
       else if ( ckey == "NUM_BATCH" ) then
          backspace(ui)
          read(ui,*) cbuf, NUM_batch
       else if ( ckey == "OC" ) then
          write(*,*) "input: OC has been renamed Checkpoint_cyc (the"
          write(*,*) "  interval, in epochs, of the checkpoint files)"
          stop
       else if ( ckey == "CHECKPOINT_CYC" ) then
          backspace(ui)
          read(ui,*) cbuf, io_cyc
       else if ( ckey == "INIT_W" ) then
          backspace(ui)
          read(ui,*) cbuf, ctmp
          call convert_capital(ctmp,init_weight_method)
       else if ( ckey == "GD_METHOD" ) then
          backspace(ui)
          read(ui,*) cbuf, ctmp
          call convert_capital(ctmp,gd_method)
       else if ( ckey == "GD_PARAM" ) then
          backspace(ui)
          read(ui,*) cbuf, gd_param(1:5)
       else if ( ckey == "GD_RATIO" ) then
          write(*,*) "input: Gd_ratio has been replaced by the weight"
          write(*,*) "  column of each Loss_term line"
          stop
       else if ( ckey == "HOD_K" ) then
          backspace(ui)
          read(ui,*) cbuf, hod_kmax_in   ! max order K; K+1 lambda lines follow
          do j=0,hod_kmax_in
             read(ui,*) lambda_hod(j)
          end do
       else if ( ckey == "HOD_CHECK" ) then
          backspace(ui)
          read(ui,*) cbuf, iswitch_hod_check
       else if ( ckey == "HOD_DENSE" ) then
          backspace(ui)
          read(ui,*) cbuf, iswitch_hod_dense
       else if ( ckey == "RAND_SEED" ) then
          backspace(ui)
          read(ui,*) cbuf, rand_seed_in
       else if ( ckey == "HOD_ALPHA_FILE" ) then
          backspace(ui)
          read(ui,*) cbuf, hod_alpha_file
!--- PINN residual training
       else if ( ckey == "PINN_RESIDUAL" ) then
          write(*,*) "input: Pinn_residual has been renamed Residual"
          write(*,*) "  (the residual is the operator, not a method brand)"
          stop
       else if ( ckey == "LBFGS_M" ) then
          backspace(ui)
          read(ui,*) cbuf, LBFGS_M
          if ( LBFGS_M < 1 .or. LBFGS_M > 200 ) then
             write(*,*) "Lbfgs_m: memory must lie in [1,200], got", LBFGS_M
             stop
          end if
       else if ( ckey == "LBFGS_M" ) then
          backspace(ui)
          read(ui,*) cbuf, LBFGS_M
          if ( LBFGS_M < 1 .or. LBFGS_M > 200 ) then
             write(*,*) "Lbfgs_m: memory pairs must lie in [1,200], got", &
                  LBFGS_M
             stop
          end if
       else if ( ckey == "LBFGS_EXPAND" ) then
          backspace(ui)
          read(ui,*) cbuf, LBFGS_expand
          if ( LBFGS_expand < 0 .or. LBFGS_expand > 20 ) then
             write(*,*) "Lbfgs_expand: doublings must lie in [0,20], got", &
                  LBFGS_expand
             stop
          end if
       else if ( ckey == "LBFGS_WOLFE" ) then
          backspace(ui)
          read(ui,*) cbuf, LBFGS_wolfe
          if ( LBFGS_wolfe < 0.d0 .or. LBFGS_wolfe >= 1.d0 ) then
             write(*,*) "Lbfgs_wolfe: c2 must lie in [0,1), got", LBFGS_wolfe
             stop
          end if
          if ( LBFGS_wolfe > 0.d0 .and. LBFGS_wolfe <= LBFGS_c1 ) then
             write(*,*) "Lbfgs_wolfe: c2 must exceed c1 =", LBFGS_c1
             stop
          end if
       else if ( ckey == "LBFGS_SS" ) then
          LBFGS_selfscale = .true.
       else if ( ckey == "LBFGS_SCAN" ) then
          lbfgs_scan = .true.
       else if ( ckey == "LBFGS_VERBOSE" ) then
          lbfgs_verbose = .true.
       else if ( ckey == "NGD_GEO" ) then
          backspace(ui)
          read(ui,*) cbuf, ngd_geo_h, ngd_geo_alpha
          if ( ngd_geo_h <= 0.d0 .or. ngd_geo_alpha <= 0.d0 ) then
             write(*,*) "Ngd_geo: h and alpha must be positive, got", &
                  ngd_geo_h, ngd_geo_alpha
             stop
          end if
          NGD_geo = .true.
       else if ( ckey == "NGD_DUAL" ) then
          NGD_dual = .true.
       else if ( ckey == "NGD_TRUST" ) then
          ! adaptive damping: Ngd_trust mu0 [lo hi]
          backspace(ui)
          read(ui,*) cbuf, NGD_trust_mu
          NGD_trust = .true.
          NGD_damping = "ABS"
       else if ( ckey == "NGD_DAMPING" ) then
          backspace(ui)
          read(ui,*) cbuf, ctmp
          call convert_capital(ctmp,NGD_damping)
          if ( NGD_damping /= "TRACE" .and. NGD_damping /= "ABS" ) then
             write(*,*) "Ngd_damping: unknown value ", trim(ctmp), " (TRACE or ABS)"
             stop
          end if
       else if ( ckey == "SYS_SRC" ) then
          ! the collocation file carries one source column per residual,
          ! between the coordinates and the exact solution
          sys_use_src = .true.
       else if ( ckey == "KALMAN_ITER" ) then
          backspace(ui)
          read(ui,*) cbuf, kalman_iter
          if ( kalman_iter < 1 .or. kalman_iter > 20 ) then
             write(*,*) "Kalman_iter: must lie in [1,20], got", kalman_iter
             stop
          end if
       else if ( ckey == "KALMAN_Q" ) then
          backspace(ui)
          read(ui,*) cbuf, kalman_q
          if ( kalman_q < 0.d0 ) then
             write(*,*) "Kalman_q: must be >= 0 (0 disables), got", kalman_q
             stop
          end if
       else if ( ckey == "KALMAN_MODE" ) then
          backspace(ui)
          read(ui,*) cbuf, ctmp
          call convert_capital(ctmp,ckey2)
          if ( ckey2 == "DENSE" ) then
             kalman_decoupled = .false.
          else if ( ckey2 == "DECOUPLED" ) then
             kalman_decoupled = .true.
          else
             write(*,*) "Kalman_mode: DENSE or DECOUPLED, got ", trim(ctmp)
             stop
          end if
       else if ( ckey == "KALMAN_GATE" ) then
          backspace(ui)
          read(ui,*) cbuf, kalman_gate
          if ( kalman_gate < 0.d0 ) then
             write(*,*) "Kalman_gate: must be >= 0 (0 disables), got", &
                  kalman_gate
             stop
          end if
       else if ( ckey == "SYS_RNOISE" ) then
          ! one observation noise per residual, used by the filter
          backspace(ui)
          read(ui,*) cbuf, sys_rnoise(1:sys_nres)
       else if ( ckey == "SYS_BALANCE" ) then
          ! Sys_balance cyc alpha
          backspace(ui)
          read(ui,*) cbuf, sys_balance_cyc, sys_balance_alpha
          sys_balance = .true.
       else if ( ckey == "SYS_WRES" ) then
          ! one weight per residual of the system, in the collocation loss
          backspace(ui)
          read(ui,*) cbuf, sys_wres(1:sys_nres)
       else if ( ckey == "SYS_WCOMP" ) then
          ! one weight per field component in the supervised loss
          backspace(ui)
          read(ui,*) cbuf, sys_wcomp(1:ndim(Nlayer))
       else if ( ckey == "SYSTEM" ) then
          ! A system of residuals over several field components.
          !
          !   System nres nterm /
          !   TRM  ir  jc  a1..aD  c /          c * d^alpha u_jc
          !   XUX  ir  ic  jc  a1..aD  c /      c * u_ic * d^alpha u_jc
          !
          ! ir is which residual the term belongs to, jc the component
          ! that is differentiated, ic the component multiplying it.
          ! XUX is what a coupled system needs and the scalar Residual
          ! block cannot write: the advection u du/dx + v du/dy, the
          ! transport of a charge density by a flow, the body force
          ! rho E.  With one component and ic = jc = 1 the two forms
          ! reduce to LIN and UUX.
          backspace(ui)
          read(ui,*) cbuf, sys_nres, itmp
          if ( .not. allocated(sys_alpha) ) then
             allocate( sys_alpha(ndim(1),256) );  sys_alpha = 0
             allocate( sys_beta(ndim(1),256) );  sys_beta = 0
             allocate( sys_gamma(ndim(1),256) );  sys_gamma = 0
             allocate( sys_delta(ndim(1),256) );  sys_delta = 0
          end if
          if ( sys_nres < 1 .or. sys_nres > size(sys_has_src) ) then
             write(*,*) "System: nres must be between 1 and", &
                  size(sys_has_src)
             stop
          end if
          do j=1,itmp
             read(ui,*) ctmp
             call convert_capital(ctmp,ckey2)
             backspace(ui)
             if ( sys_nterm >= size(sys_coeff) ) then
                write(*,*) "System: at most", size(sys_coeff), " terms"
                stop
             end if
             sys_nterm = sys_nterm + 1
             if ( ckey2 == "TRM" ) then
                read(ui,*) ctmp, sys_res(sys_nterm), sys_cmp(sys_nterm), &
                     sys_alpha(1:ndim(1),sys_nterm), sys_coeff(sys_nterm)
                sys_fac(sys_nterm) = 0
             else if ( ckey2 == "XUX" ) then
                read(ui,*) ctmp, sys_res(sys_nterm), sys_fac(sys_nterm), &
                     sys_cmp(sys_nterm), sys_alpha(1:ndim(1),sys_nterm), &
                     sys_coeff(sys_nterm)
                sys_fac_ind(sys_nterm) = 1        ! the value itself
             else if ( ckey2 == "TRP" ) then
                ! c * d^gamma u_k * d^beta u_i * d^alpha u_j: three
                ! factors, which a compressible momentum flux rho u u_x
                ! or a field-dependent mobility needs.
                read(ui,*) ctmp, sys_res(sys_nterm), sys_third(sys_nterm), &
                     sys_gamma(1:ndim(1),sys_nterm), sys_fac(sys_nterm), &
                     sys_beta(1:ndim(1),sys_nterm), sys_cmp(sys_nterm), &
                     sys_alpha(1:ndim(1),sys_nterm), sys_coeff(sys_nterm)
             else if ( ckey2 == "QAD" ) then
                ! c * d^delta u_l * d^gamma u_k * d^beta u_i * d^alpha u_j:
                ! four factors.  The quartic terms of the higher
                ! dispersive hierarchies need it; the seventh-order Lax
                ! equation carries 140 u^3 u_x.
                read(ui,*) ctmp, sys_res(sys_nterm), sys_fourth(sys_nterm), &
                     sys_delta(1:ndim(1),sys_nterm), sys_third(sys_nterm), &
                     sys_gamma(1:ndim(1),sys_nterm), sys_fac(sys_nterm), &
                     sys_beta(1:ndim(1),sys_nterm), sys_cmp(sys_nterm), &
                     sys_alpha(1:ndim(1),sys_nterm), sys_coeff(sys_nterm)
             else if ( ckey2 == "DXD" ) then
                ! c * d^beta u_ic * d^alpha u_jc, a product of two
                ! derivatives.  The divergence of a flux needs it:
                ! div(rho grad phi) = grad(rho).grad(phi) + rho lap(phi),
                ! and the first term is a product of two first
                ! derivatives of different components.
                read(ui,*) ctmp, sys_res(sys_nterm), sys_fac(sys_nterm), &
                     sys_beta(1:ndim(1),sys_nterm), sys_cmp(sys_nterm), &
                     sys_alpha(1:ndim(1),sys_nterm), sys_coeff(sys_nterm)
             else
                write(*,*) "System: unknown term type ", trim(ctmp)
                write(*,*) "  (TRM, XUX, DXD, TRP or QAD)"
                stop
             end if
             if ( sys_res(sys_nterm) < 1 .or. &
                  sys_res(sys_nterm) > sys_nres ) then
                write(*,*) "System: term", j, " names residual", &
                     sys_res(sys_nterm), " of", sys_nres
                stop
             end if
          end do

       else if ( ckey == "RESIDUAL" ) then
          ! must appear AFTER Nlayer (needs ndim(1) for the alpha lists)
          if ( .not.allocated(ndim) ) then
             write(*,*) "Residual must appear after Nlayer in input_nn.dat"
             stop
          end if
          if ( .not.allocated(pinn_alpha) ) then
             allocate( pinn_alpha(ndim(1),64) ); pinn_alpha=0
          end if
          backspace(ui)
          read(ui,*) cbuf, itmp    ! number of term lines that follow
          do j=1,itmp
             read(ui,*) ctmp
             call convert_capital(ctmp,ckey2)
             backspace(ui)
             if ( ckey2 == "LIN" .or. ckey2 == "UUX" ) then
                if ( pinn_nterm >= size(pinn_coeff) ) then
                   write(*,*) "Residual: at most", size(pinn_coeff), " terms"
                   stop
                end if
                pinn_nterm = pinn_nterm+1
                read(ui,*) ctmp, pinn_coeff(pinn_nterm), pinn_alpha(1:ndim(1),pinn_nterm)
                pinn_nonlin(pinn_nterm) = ( ckey2 == "UUX" )
             else if ( ckey2 == "DXLAP" ) then
                ! c * d_ix Lap^jj u, where the Laplacian runs over every
                ! axis except the time axis.  Written as
                !     DXLAP c jj /        differentiated axis = 1
                !     DXLAP c jj ix /     differentiated axis = ix
                ! The expansion is deferred to the end of the input so
                ! that Time_axis works from anywhere in the file.
                ! The trailing slash of the input convention terminates
                ! list-directed input and leaves the remaining items
                ! untouched, so the optional axis simply keeps its default
                ! when the short form  DXLAP c k /  is written.
                jax = 1
                read(ui,*) ctmp, rtmp, jj, jax
                if ( ndim(1) < 2 ) then
                   write(*,*) "DXLAP requires ndim(1)>=2 (spatial and time)"
                   stop
                end if
                if ( jax < 1 .or. jax > ndim(1) ) then
                   write(*,*) "DXLAP: the differentiated axis must lie in", &
                        " [1, ndim(1)], got", jax
                   stop
                end if
                if ( pinn_ndxl >= 16 ) then
                   write(*,*) "DXLAP: at most 16 such terms"
                   stop
                end if
                pinn_ndxl = pinn_ndxl + 1
                pinn_dxl_coeff(pinn_ndxl) = rtmp
                pinn_dxl_pow(pinn_ndxl)   = jj
                pinn_dxl_axis(pinn_ndxl)  = jax
                pinn_nterm = pinn_nterm   ! the terms appear at expansion time
             else if ( ckey2 == "SRC" ) then
                ! c * f(x), with f read per collocation point.  Not a
                ! multi-index term: it carries no derivative of u.
                read(ui,*) ctmp, pinn_src_coeff
                if ( pinn_has_src ) then
                   write(*,*) "Residual: SRC may appear only once"
                   stop
                end if
                pinn_has_src = .true.
             else
                write(*,*) "Residual: unknown term type: ",trim(ctmp)
                write(*,*) "  (LIN, UUX, DXLAP or SRC)"
                stop
             end if
             if ( pinn_nterm > 64 ) then
                write(*,*) "Pinn_residual: too many terms (max 64)"
                stop
             end if
          end do
       else if ( ckey == "PINN_EXACT" ) then
          write(*,*) "input: Pinn_exact has been renamed Exact_solution"
          stop
       else if ( ckey == "COMMITTEE" ) then
          backspace(ui)
          read(ui,*) cbuf, n_committee
          if ( n_committee < 2 ) then
             write(*,*) "Committee needs at least 2 members, got", n_committee
             stop
          end if
          if ( allocated(committee_file) ) deallocate( committee_file )
          allocate( committee_file(n_committee) )
          do j=1,n_committee
             read(ui,*) committee_file(j)
          end do
       else if ( ckey == "INIT_W_OMEGA" ) then
          backspace(ui)
          read(ui,*) cbuf, init_w_omega
          if ( init_w_omega <= 0.d0 ) then
             write(*,*) "Init_w_omega must be positive, got", init_w_omega
             stop
          end if
       else if ( ckey == "TIME_AXIS" ) then
          if ( .not.allocated(ndim) ) then
             write(*,*) "Time_axis must appear after Nlayer in input_nn.dat"
             stop
          end if
          backspace(ui)
          read(ui,*) cbuf, pinn_time_axis
          if ( pinn_time_axis < 1 .or. pinn_time_axis > ndim(1) ) then
             write(*,*) "Time_axis must lie in [1, ndim(1)], got", pinn_time_axis
             stop
          end if
       else if ( ckey == "EXACT_SOLUTION" ) then
          backspace(ui)
          read(ui,*) cbuf, iswitch_pinn_exact
!--- Natural-gradient schedules (ported from the mGGA_subsys version)
       else if ( ckey == "NGD_SCHEDULE_ETA" ) then
          backspace(ui)
          read(ui,*) cbuf, ctmp, ngd_param_eta(1:2)
          call convert_capital(ctmp,NGD_schedule_eta)
       else if ( ckey == "NGD_ETA_BOUND" ) then
          backspace(ui)
          read(ui,*) cbuf, ngd_eta_bound
       else if ( ckey == "NGD_SCHEDULE_MU" ) then
          backspace(ui)
          read(ui,*) cbuf, ctmp, ngd_param_mu(1:2)
          call convert_capital(ctmp,NGD_schedule_mu)
       else if ( ckey == "NGD_MU_BOUND" ) then
          backspace(ui)
          read(ui,*) cbuf, ngd_mu_bound
       else if ( ckey == "TRM" .or. ckey == "XUX" .or. ckey == "DXD" .or. &
                 ckey == "TRP" .or. ckey == "QAD" .or. ckey == "LIN" .or. &
                 ckey == "UUX" .or. ckey == "DXLAP" .or. ckey == "SRC" ) then
          ! A term line outside its block means the block's declared term
          ! count is smaller than the number of term lines written, so
          ! the operator being solved is NOT the one on the page: the
          ! trailing terms are silently absent.  A warning is too weak
          ! for that, since the run then converges on a different
          ! equation and reports nothing wrong.
          write(*,*) "input: term line '"//trim(ckey)//"' appears outside "// &
               "its block."
          write(*,*) "  Raise the term count on the System or Residual "// &
               "line to cover every term line,"
          write(*,*) "  or remove the extra line.  The operator would "// &
               "otherwise be missing its last terms."
          stop
       else if ( ckey(1:1) /= "#" .and. ckey(1:1) /= "!" .and. &
                 len_trim(ckey) > 0 ) then
          ! An unrecognized token at the top level is almost always a
          ! misspelled keyword; reporting it prevents a silent fall back
          ! to the default value.  Lines starting with # or ! are
          ! comments and are not reported.
          write(*,*) "input WARNING: unrecognized keyword ignored: ", trim(cbuf)
       end if

    end do
999 continue

    ! The periodic initialisation places the pre-activation deliberately
    ! inside one oscillation.  That is meaningful for any oscillatory
    ! activation, not only for sin: J_0 is flat near the origin (1 - a^2/4)
    ! and only shows its structure past its first zero at 2.405, so with a
    ! scaling meant for a saturating activation it never leaves the flat
    ! region.  It is rejected for the smooth saturating ones, where it has
    ! no meaning.
    if ( trim(init_weight_method) == "BESSEL_INIT" .and. &
         trim(Activation_type) /= "BESSEL" ) then
       write(*,*) "Init_w BESSEL_INIT is derived from the statistics of"
       write(*,*) "  J_0 and means nothing for another activation."
       write(*,*) "  Use it with Activation BESSEL."
       stop
    end if
    if ( trim(init_weight_method) == "SIREN" .and. &
         trim(Activation_type) /= "SIN" .and. &
         trim(Activation_type) /= "BESSEL" ) then
       write(*,*) "Init_w SIREN places the pre-activation inside one"
       write(*,*) "  oscillation, which only means something for an"
       write(*,*) "  oscillatory activation.  Use it with Activation SIN"
       write(*,*) "  or BESSEL, or choose another Init_w."
       stop
    end if
    if ( .not. allocated(ndim) ) then
       write(*,*) "input: Nlayer is required.  Give"
       write(*,*) "    Nlayer  L /"
       write(*,*) "  followed by L lines, the width of each layer."
       stop
    end if
    close(ui)

  END SUBROUTINE read_parameters
  
  SUBROUTINE read_data
    implicit none
    real(8),allocatable :: x_tmp(:)
    integer,parameter :: ut=21, uw=22, ud=23
    ! Fixed line buffer for the record-checked reader below.  Records
    ! may wrap over several physical lines; each line is parsed on its
    ! own, so only single physical lines have to fit (64 kB here, with
    ! an explicit stop if a longer one is met).
    character(65536) :: crec
    integer :: ita, nsp
    integer,allocatable :: sax(:)
    integer :: ios_rec, ntok_line, ntok_acc
    integer :: i, j, n, itmp, i_flag, ibuf
!    integer,allocatable :: ind_val(:),ind_t(:)
    integer :: ndrawn, n0
    integer :: is
    integer :: i3
    character(30) :: cbuf
    integer,allocatable :: hod_seed_tmp(:, :)
    integer :: nexa, jj, nsrc, jz, nseed_file, nseed_cap
    real(8),allocatable :: recbuf(:)   ! one logical record, checked reader
    
    ndim_max=maxval(ndim(:))
    allocate( recbuf( ndim(1)+2*max(2,ndim(Nlayer))+16 ) );  recbuf = 0.d0
    NUM_weight = 0
    do i=1,Nlayer-1
       NUM_weight = NUM_weight + ndim(i+1)*( ndim(i)+1 )
    end do

! bias: weight(:,:,0)
    allocate( weight(Nlayer,ndim_max,0:ndim_max) ); weight=0.d0
    allocate( weight_best(Nlayer,ndim_max,0:ndim_max) ); weight_best=0.d0
    allocate( weight_recv(Nlayer,ndim_max,0:ndim_max) ); weight_recv=0.d0

!NUM_input: total # of data set defined by Ndata_train_set(1:Ntot_train_set)
    NUM_input=0
    do i=1,Ntot_train_set
       NUM_input = NUM_input+Ndata_train_set(i)
    end do
    ! The adaptive rules divide by sqrt(accumulator) + eps, or by
    ! sqrt(accumulator + eps).  With eps = 0 the first step of a weight
    ! whose gradient component is zero divides by zero and fills the
    ! network with NaN, so the slot each rule uses is required positive.
    if ( iswitch_fit == 1 ) then
       itmp = 0
       if ( trim(gd_method) == "ADAGRAD" )  itmp = 2
       if ( trim(gd_method) == "ADADELTA" ) itmp = 2
       if ( trim(gd_method) == "RMSPROP" )  itmp = 3
       if ( trim(gd_method) == "RMSPROP_NESTEROV" ) itmp = 3
       if ( trim(gd_method) == "ADAM" )     itmp = 4
       if ( itmp > 0 ) then
          if ( gd_param(itmp) <= 0.d0 ) then
             write(*,*) "GD_param: ", trim(gd_method), " divides by an"
             write(*,*) "  accumulator regularized by p", itmp, &
                  ", which must be positive; got", gd_param(itmp)
             stop
          end if
       end if
    end if
    if ( trim(gd_method) == "KALMAN" .and. iswitch_fit == 1 ) then
       if ( gd_param(1) <= 0.d0 ) then
          write(*,*) "GD_param: Kalman needs p1 > 0 (the initial diagonal"
          write(*,*) "  of the covariance), got", gd_param(1)
          stop
       end if
       if ( gd_param(2) <= 0.d0 .or. gd_param(2) > 1.d0 ) then
          write(*,*) "GD_param: Kalman needs 0 < p2 <= 1 (the initial"
          write(*,*) "  forgetting factor lambda), got", gd_param(2)
          stop
       end if
       if ( gd_param(3) <= 0.d0 .or. gd_param(3) > 1.d0 ) then
          write(*,*) "GD_param: Kalman needs 0 < p3 <= 1 (the forgetting"
          write(*,*) "  schedule lambda0), got", gd_param(3)
          stop
       end if
    end if
    ! L-BFGS is exempt from the all-zero guard: its step length comes
    ! from the line search, so it is the one method with no
    ! hyper-parameter in GD_param to leave nonzero.
    if ( maxval(abs(gd_param(1:5))) == 0.d0 .and. iswitch_fit == 1 &
         .and. trim(gd_method) /= "LBFGS" &
         .and. trim(gd_method) /= "LM" ) then
       write(*,*) "input: GD_param is unset (all five values are zero),"
       write(*,*) "  so no optimizer step can be taken.  Give"
       write(*,*) "    GD_param  p1 p2 p3 p4 p5 /"
       write(*,*) "  on its own line (any order relative to GD_method)."
       stop
    end if

    if ( NUM_validation < 0 .or. NUM_validation >= NUM_input ) then
       write(*,*) "input: Num_validation must lie in [0, NUM_input-1], got", &
            NUM_validation, " of", NUM_input
       stop
    end if
    if ( validation_cyc <= 0 ) then
       write(*,*) "input: Validation_cyc must be positive, got", validation_cyc
       stop
    end if
    ! Fortran does not guarantee short-circuit evaluation, so the epoch
    ! test mod(loop, Shuffle_cyc_mpi) may be evaluated even when the
    ! shuffle is off: a zero cycle would be a division by zero.
    if ( Average_cyc_mpi <= 0 ) then
       write(*,*) "input: Average_cyc_mpi must be positive, got", Average_cyc_mpi
       stop
    end if
    if ( Ntot_train_set == 0 ) then
       write(*,*) "input: at least one Loss_term line is required"
       stop
    end if
    ! Unimplemented legacy keys (Tikhonov_reg, Elasnet_reg, Select_reg,
    ! Fix_weight, Deriv_train_cyc/len, Set_zero, Global_ratio, Eval_ratio)
    ! have been removed from the parser entirely; an old input that still
    ! contains them is reported by the unrecognized-keyword warning.
    if ( Shuffle_cyc_mpi <= 0 ) then
       write(*,*) "input: Shuffle_cyc_mpi must be positive, got", Shuffle_cyc_mpi
       stop
    end if
    if ( iswitch_shuffle /= 0 .and. iswitch_shuffle /= 1 ) then
       write(*,*) "input: Shuffle must be 0 or 1, got", iswitch_shuffle
       stop
    end if
    NUM_train = NUM_input - NUM_validation
    ! If any Loss_term names its own batch size, every term must, and
    ! Num_batch is then their sum rather than an independent setting:
    ! the two cannot both decide how many points an epoch draws.
    itmp = 0
    do j = 1, Ntot_train_set
       if ( nbatch_set(j) > 0 ) itmp = itmp + 1
    end do
    if ( itmp > 0 ) then
       if ( itmp /= Ntot_train_set ) then
          write(*,*) "input:", itmp, " of", Ntot_train_set, " Loss_term"
          write(*,*) "  lines give a batch size.  Give one on every line,"
          write(*,*) "  or on none of them."
          stop
       end if
       ! The sum is recomputed after the sets are split, since a size
       ! that asks for a whole set is clamped to what that set kept.
       NUM_batch = sum( nbatch_set(1:Ntot_train_set) )
    end if
    if ( NUM_batch < 0 ) then
       NUM_batch = min( 10, NUM_train )
       write(*,*) "Num_batch not set; defaulting to", NUM_batch
    end if
    ! The draw is stratified, so every Loss_term must be able to place at
    ! least one point in the batch.  A smaller batch would silently drop
    ! the leading terms and train on an operator the input file does not
    ! describe.
    if ( NUM_batch < Ntot_train_set ) then
       write(*,*) "input: Num_batch =", NUM_batch, " but there are", &
            Ntot_train_set, " Loss_term lines"
       write(*,*) "  the minibatch is stratified, so it needs at least one"
       write(*,*) "  point per term; raise Num_batch, or give the terms"
       write(*,*) "  their own sizes on the Loss_term lines."
       stop
    end if
    if ( NUM_batch < 1 .or. NUM_batch > NUM_train ) then
       write(*,*) "input: Num_batch must lie in [1, NUM_train], got", &
            NUM_batch, " of", NUM_train
       stop
    end if
    ! L-BFGS is a full-batch method: the pairs (s,y) only mean anything
    ! if consecutive gradients are of the same function, and a resampled
    ! minibatch changes the function every step.  Refusing here is
    ! kinder than converging to nothing later.
    if ( ( trim(gd_method) == "LBFGS" .or. trim(gd_method) == "LM" ) &
         .and. iswitch_fit == 1 .and. NUM_batch /= NUM_train ) then
       write(*,'(a,a,a)') " input: GD_method ", trim(gd_method), &
            " is a full-batch method, so"
       write(*,*) "  Num_batch must equal the number of training points"
       write(*,*) "  (NUM_input - Num_validation).  Set"
       write(*,'(a,i0,a)') "     Num_batch  ", NUM_train, " /"
       stop
    end if
    ! The per-pattern filter observes one carried slot of one output at a
    ! time, through the single-output forward pass.  For a value or
    ! derivative fit of a multi-output network that would update the
    ! first component only and silently leave the others untrained; the
    ! system-residual path has its own multi-output entry and is fine.
    if ( trim(gd_method) == "KALMAN" .and. ndim(Nlayer) > 1 .and. &
         any( form_train(1:Ntot_train_set) == "MATH" .or. &
              form_train(1:Ntot_train_set) == "MATH_HOD" ) ) then
       write(*,*) "input: GD_method KALMAN fits DATA and HOD_DATA terms"
       write(*,*) "  through the single-output observation, but the network"
       write(*,*) "  has", ndim(Nlayer), " outputs.  Use one output, or"
       write(*,*) "  another method, or a COLLOCATION-only loss."
       stop
    end if
    ! HOD_DATA targets have no output-component axis, and the fitting
    ! path calls the single-output forward: with more than one output
    ! only the first would be trained, silently, while the reported cost
    ! and the validation figures came from a different quantity.  The
    ! System residual path is unaffected -- it has its own multi-output
    ! entry -- so this refuses only the combination that is not
    ! implemented.
    if ( ndim(Nlayer) > 1 .and. &
         any( form_train(1:Ntot_train_set) == "MATH_HOD" ) ) then
       write(*,*) "input: HOD_DATA fits derivative targets of one output,"
       write(*,*) "  but the network has", ndim(Nlayer), " of them."
       write(*,*) "  Use a single output for HOD_DATA, or express the"
       write(*,*) "  requirement as a System residual."
       stop
    end if
    ! The geodesic correction reads, for every metric row, the observable
    ! that row was built from.  The derivative-fitting path contributes
    ! metric rows but no such observable, so the correction would read an
    ! array that was never filled.  Refuse the combination rather than
    ! compute from uninitialized memory.
    if ( NGD_geo .and. trim(gd_method) == "NATURAL_GRAD" .and. &
         any( form_train(1:Ntot_train_set) == "MATH_HOD" ) ) then
       write(*,*) "input: Ngd_geo is not implemented for HOD_DATA terms."
       write(*,*) "  The geodesic correction needs one observable per"
       write(*,*) "  metric row, and the derivative rows do not carry one."
       write(*,*) "  Drop Ngd_geo, or train the HOD_DATA term with another"
       write(*,*) "  method."
       stop
    end if

    allocate( descriptor_input(NUM_input,ndim(1)) ); descriptor_input=0.d0
    allocate( response_input(NUM_input,ndim(Nlayer)) ); response_input=0.d0

    allocate( x_tmp(ndim(1)) )

!------------------------------------------------------------!
! High-order derivative / PINN training: build multi-index/Bell tables & storage
    if ( any( form_train(1:Ntot_train_set)=="MATH_HOD" ) .or. &
         any( form_train(1:Ntot_train_set)=="PINN" ) ) then
       ! Expand the deferred DXLAP terms.  Doing it here, after the whole
       ! file has been read, is what makes Time_axis independent of where
       ! it appears.
       ita = pinn_time_axis
       if ( ita == 0 ) ita = ndim(1)
       if ( pinn_ndxl > 0 ) then
          allocate( sax(ndim(1)) );  sax = 0
          nsp = 0
          do j=1,ndim(1)
             if ( j /= ita ) then
                nsp = nsp + 1
                sax(nsp) = j
             end if
          end do
          do j=1,pinn_ndxl
             if ( pinn_dxl_axis(j) == ita ) then
                write(*,*) "DXLAP: the differentiated axis", pinn_dxl_axis(j), &
                     " is the time axis"
                write(*,*) "  (Time_axis =", ita, "), which is not what this"
                write(*,*) "  term means.  Give DXLAP c k ix / with a spatial ix."
                stop
             end if
             call expand_dxlap( pinn_dxl_coeff(j), pinn_dxl_pow(j), &
                  pinn_dxl_axis(j), nsp, sax(1:nsp) )
          end do
          deallocate( sax )
       end if

       if ( any( form_train(1:Ntot_train_set)=="PINN" ) ) then
          ! The operator comes either from the scalar Residual block or
          ! from the System block; one of the two must be present, and
          ! not both, since they would describe different equations.
          if ( pinn_nterm <= 0 .and. sys_nterm <= 0 ) then
             write(*,*) "A collocation term needs an operator: give a"
             write(*,*) "  Residual block for a scalar field, or a"
             write(*,*) "  System block for several components."
             stop
          end if
          if ( pinn_nterm > 0 .and. sys_nterm > 0 ) then
             write(*,*) "Residual and System both given; they describe"
             write(*,*) "  different operators.  Keep one."
             stop
          end if
          ! A determined system has one residual per component.  More
          ! are allowed, because a component that enters the equations
          ! only through its gradient is fixed by them up to an additive
          ! constant, and a further residual is how that constant is
          ! pinned: the pressure of an incompressible flow is the case at
          ! hand, and a residual holding p alone drives its level to
          ! zero.  Fewer would leave a component undetermined.
          ! The extended Kalman filter presents one scalar observable
          ! at a time, so a system would need a component index threaded
          ! through its update; without it the filter would silently
          ! treat the system as its first component.  Refusing is the
          ! honest behaviour: a wrong answer that looks like a slow one
          ! is the worst outcome.  The natural gradient needs no such
          ! change, since its metric is built from the weight gradient
          ! of the loss and does not see the components at all.
          ! Both the natural gradient and the filter now reach the
          ! system path: the metric is built from the weight gradient of
          ! the loss, and the filter takes one residual at a time as its
          ! scalar observable.
          if ( sys_nterm > 0 .and. sys_nres < ndim(Nlayer) ) then
             write(*,*) "System: ", sys_nres, " residuals for ", &
                  ndim(Nlayer), " components leaves one undetermined."
             stop
          end if
          ! max derivative order appearing in the operator
          itmp = 0
          do j=1,pinn_nterm
             itmp = max( itmp, sum(pinn_alpha(1:ndim(1),j)) )
          end do
          do j=1,sys_nterm
             itmp = max( itmp, sum(sys_alpha(1:ndim(1),j)) )
             ! every factor of a System term carries its own multi-index,
             ! and any of them may be the one of highest order
             if ( sys_fac(j)    > 0 ) &
                  itmp = max( itmp, sum(sys_beta(1:ndim(1),j)) )
             if ( sys_third(j)  > 0 ) &
                  itmp = max( itmp, sum(sys_gamma(1:ndim(1),j)) )
             if ( sys_fourth(j) > 0 ) &
                  itmp = max( itmp, sum(sys_delta(1:ndim(1),j)) )
          end do
          if ( hod_kmax_in <= 0 ) then
             hod_kmax_in = max(itmp,1)
          else if ( hod_kmax_in < itmp ) then
             write(*,*) "Hod_k smaller than the residual order:",hod_kmax_in,itmp
             stop
          end if
       end if
       if ( hod_kmax_in <= 0 ) then
          write(*,*) "MATH_HOD requires the Hod_k keyword (K>=1)"
          stop
       end if
       ! The engine needs sigma to be C^(K+1): the adjoint reaches order
       ! K+1.  All three activations below satisfy that for every K.
       if ( activation_type /= "TANH" .and. activation_type /= "SIN" &
            .and. activation_type /= "ERF" .and. &
            activation_type /= "BESSEL" .and. &
            activation_type /= "BESSEL1" ) then
          write(*,*) "MATH_HOD/PINN needs a C^infinity activation:", &
               " TANH, SIN, ERF, BESSEL or BESSEL1, got ", trim(activation_type)
          stop
       end if
       ! Several outputs are allowed when a System block supplies one
       ! residual per component; the scalar forms carry one field.
       if ( ndim(Nlayer) /= 1 .and. sys_nterm <= 0 ) then
          write(*,*) "A scalar Residual or MATH_HOD target carries one"
          write(*,*) "  field, so ndim(Nlayer) must be 1.  For several"
          write(*,*) "  components use a System block."
          stop
       end if

       if ( any( lambda_hod(0:hod_kmax_in) < 0.d0 ) ) then
          write(*,*) "Hod_k: lambda weights must be >= 0 (a negative"
          write(*,*) "  weight makes the objective unbounded below)"
          stop
       end if
       ! closure seeds (downward closed multi-index set):
       !  - MATH_HOD without Hod_alpha_file -> dense (the file defines all columns)
       !  - otherwise: file seeds (if any) + all residual multi-indices,
       !    and the engine carries their downward closure
       ! One capacity for both branches: file seeds, one per scalar
       ! residual term, and up to four per System term (alpha, and beta,
       ! gamma, delta when the term has that many factors).  Sizing the
       ! file branch without the System share overran the buffer as soon
       ! as an explicit Hod_alpha_file was combined with a System block,
       ! which is the usual arrangement for a coupled problem.
       itmp = 0
       nseed_file = 0
       if ( trim(hod_alpha_file) /= 'NONE' .and. trim(hod_alpha_file) /= 'none' ) then
          open(ut,file=trim(hod_alpha_file),status='old')
          read(ut,*) nseed_file
          if ( nseed_file < 0 ) then
             write(*,*) "Hod_alpha_file: the seed count must be >= 0, got", &
                  nseed_file
             stop
          end if
       end if
       nseed_cap = max( 1, nseed_file + pinn_nterm + 4*sys_nterm )
       allocate( hod_seed_tmp(ndim(1),nseed_cap) )
       hod_seed_tmp = 0
       if ( nseed_file > 0 ) then
          do j=1,nseed_file
             read(ut,*) hod_seed_tmp(1:ndim(1),j)
          end do
          itmp = nseed_file
       end if
       if ( nseed_file >= 0 .and. trim(hod_alpha_file) /= 'NONE' .and. &
            trim(hod_alpha_file) /= 'none' ) close(ut)
       if ( iswitch_hod_dense == 1 ) then
          itmp = 0        ! dense set forced (Hod_dense 1; for cost measurements)
       else if ( any( form_train(1:Ntot_train_set)=="MATH_HOD" ) .and. &
            trim(hod_alpha_file)=='NONE' ) then
          itmp = 0        ! dense set
       else
          do j=1,pinn_nterm
             hod_seed_tmp(1:ndim(1),itmp+j) = pinn_alpha(1:ndim(1),j)
          end do
          itmp = itmp + pinn_nterm   ! 0 stays 0 (dense) only when no terms
          ! and every factor of every System term; the closure of the
          ! union is what the residual and its adjoint actually read
          do j=1,sys_nterm
             if ( itmp + 4 > nseed_cap ) then
                write(*,*) "closure seeds: capacity", nseed_cap, &
                     " exhausted at System term", j
                stop
             end if
             itmp = itmp + 1
             hod_seed_tmp(1:ndim(1),itmp) = sys_alpha(1:ndim(1),j)
             if ( sys_fac(j) > 0 ) then
                itmp = itmp + 1
                hod_seed_tmp(1:ndim(1),itmp) = sys_beta(1:ndim(1),j)
             end if
             if ( sys_third(j) > 0 ) then
                itmp = itmp + 1
                hod_seed_tmp(1:ndim(1),itmp) = sys_gamma(1:ndim(1),j)
             end if
             if ( sys_fourth(j) > 0 ) then
                itmp = itmp + 1
                hod_seed_tmp(1:ndim(1),itmp) = sys_delta(1:ndim(1),j)
             end if
          end do
       end if
       call init_hod_tables( ndim(1), hod_kmax_in, itmp, hod_seed_tmp )
       deallocate( hod_seed_tmp )
       ! map residual terms onto the carried multi-index set
       do j=1,pinn_nterm
          pinn_ind(j) = alpha_index( pinn_alpha(1:ndim(1),j) )
          if ( pinn_ind(j) == 0 ) then
             write(*,*) "PINN residual term not in the carried multi-index set:",j
             stop
          end if
       end do

       ! the same for the system terms, and a check that every component
       ! named by a term exists in the network
       do j = 1, sys_nterm
          sys_ind(j) = alpha_index( sys_alpha(1:ndim(1),j) )
          if ( sys_fac(j) > 0 ) &
               sys_fac_ind(j) = alpha_index( sys_beta(1:ndim(1),j) )
          ! Fill the factor list the residual reads.  The keyword forms
          ! name a differentiated component, optionally a multiplying
          ! one, and optionally a third; the list is what the product
          ! and its adjoint iterate over.
          sys_fcomp(1,j) = sys_cmp(j)
          sys_find(1,j)  = sys_ind(j)
          sys_nfac(j) = 1
          if ( sys_fac(j) > 0 ) then
             sys_nfac(j) = 2
             sys_fcomp(2,j) = sys_fac(j)
             sys_find(2,j)  = sys_fac_ind(j)
          end if
          if ( sys_third(j) > 0 ) then
             sys_nfac(j) = 3
             sys_fcomp(3,j) = sys_third(j)
             sys_find(3,j)  = alpha_index( sys_gamma(1:ndim(1),j) )
          end if
          if ( sys_fourth(j) > 0 ) then
             sys_nfac(j) = 4
             sys_fcomp(4,j) = sys_fourth(j)
             sys_find(4,j)  = alpha_index( sys_delta(1:ndim(1),j) )
             if ( sys_third(j) < 1 ) then
                write(*,*) "System: term", j, " names a fourth factor " // &
                     "without a third"
                stop
             end if
          end if
          ! every factor must name a multi-index the engine carries;
          ! a zero here would be read as tm(component,0) later
          do jz = 1, sys_nfac(j)
             if ( sys_find(jz,j) == 0 ) then
                write(*,*) "System: term", j, " factor", jz, &
                     " asks for a multi-index that is not carried."
                write(*,*) "  Raise Hod_k, or add the index to " // &
                     "Hod_alpha_file, or drop the term."
                stop
             end if
          end do
          if ( sys_fourth(j) < 0 .or. sys_fourth(j) > ndim(Nlayer) ) then
             write(*,*) "System: term", j, " multiplies by component", &
                  sys_fourth(j), " but the network has", ndim(Nlayer)
             stop
          end if
          if ( sys_cmp(j) < 1 .or. sys_cmp(j) > ndim(Nlayer) ) then
             write(*,*) "System: term", j, " differentiates component", &
                  sys_cmp(j), " but the network has", ndim(Nlayer)
             stop
          end if
          if ( sys_fac(j) < 0 .or. sys_fac(j) > ndim(Nlayer) ) then
             write(*,*) "System: term", j, " multiplies by component", &
                  sys_fac(j), " but the network has", ndim(Nlayer)
             stop
          end if
          ! the third factor's component was not checked here, so a TRP
          ! term naming a component outside 1..nout indexed tm past its
          ! first dimension
          if ( sys_third(j) < 0 .or. sys_third(j) > ndim(Nlayer) ) then
             write(*,*) "System: term", j, " multiplies by component", &
                  sys_third(j), " but the network has", ndim(Nlayer)
             stop
          end if
          ! and the components the factor list actually carries
          do jz = 1, sys_nfac(j)
             if ( sys_fcomp(jz,j) < 1 .or. sys_fcomp(jz,j) > ndim(Nlayer) ) then
                write(*,*) "System: term", j, " factor", jz, " names", &
                     " component", sys_fcomp(jz,j), " of", ndim(Nlayer)
                stop
             end if
          end do
       end do
       call echo_residual( ita )
       call write_alpha_order( 'hod_alpha_order.dat' )
       allocate( hod_target_input(NUM_input,NUM_alpha) ); hod_target_input=0.d0
       allocate( pinn_src(NUM_input) ); pinn_src=0.d0
       if ( sys_nterm > 0 ) then
          allocate( sys_src_input(NUM_input,max(sys_nres,1)) )
          sys_src_input = 0.d0
       end if
    end if
!------------------------------------------------------------!
    n0=1
    do i=1,Ntot_train_set
       if ( form_train(i)=="MATH" ) then
          label_start(i)=n0
          open(ut,file=file_train(1,i),status='old')
          ! each record: x(1:ndim(1)) then the target value
          do n=1,Ndata_train_set(i)
             call read_checked_record( ut, file_train(1,i), n, &
                  ndim(1)+ndim(Nlayer), recbuf )
             descriptor_input(n0,1:ndim(1)) = recbuf(1:ndim(1))
             response_input(n0,1:ndim(Nlayer)) = &
                  recbuf(ndim(1)+1:ndim(1)+ndim(Nlayer))
             n0=n0+1
          end do
          close(ut)
          label_end(i)=n0-1

       else if ( form_train(i)=="MATH_HOD" ) then
          label_start(i)=n0
          open(ut,file=file_train(1,i),status='old')
          ! each line: x(1:ndim(1)), then y_alpha for all NUM_alpha
          ! multi-indices in the canonical order (see hod_alpha_order.dat).
          ! Each record is token-counted before parsing: a short record
          ! would otherwise be completed silently from the next line by
          ! the list-directed read, frame-shifting every later record
          ! (the only symptom was an end-of-file crash with no location,
          ! or none at all if the file had spare lines).
          ! A logical record may wrap over several physical lines, so each
          ! line is parsed on its own into the next free positions and the
          ! token ndrawn is accumulated until it reaches the expected
          ! ndim(1)+NUM_alpha.  A short record is caught as an overshoot
          ! (its continuation pulls the next record in), a long line by
          ! the buffer check, and a premature end of file by iostat.
          do n=1,Ndata_train_set(i)
             ntok_acc = 0
             do while ( ntok_acc < ndim(1)+NUM_alpha )
                read(ut,'(a)',iostat=ios_rec) crec
                if ( ios_rec /= 0 ) then
                   write(*,*) "read_data: file ", trim(file_train(1,i)), &
                        " ends inside record", n, " (", ntok_acc, " of", &
                        ndim(1)+NUM_alpha, " values found)"
                   stop
                end if
                if ( len_trim(crec) >= len(crec) ) then
                   write(*,*) "read_data: a line of ", trim(file_train(1,i)), &
                        " exceeds the ", len(crec), " character buffer"
                   stop
                end if
                ntok_line = count_tokens( crec )
                if ( ntok_line == 0 ) cycle
                if ( ntok_acc+ntok_line > ndim(1)+NUM_alpha ) then
                   write(*,*) "read_data: record", n, " of ", trim(file_train(1,i))
                   write(*,*) "  accumulates", ntok_acc+ntok_line, &
                        " values where ndim(1)+NUM_alpha =", &
                        ndim(1)+NUM_alpha, " are required: a line is missing a value"
                   write(*,*) "  (x(1:ndim(1)) then y_alpha for every carried multi-index)"
                   stop
                end if
                call parse_record_part( crec, ntok_line, n0, ntok_acc )
                ntok_acc = ntok_acc + ntok_line
             end do
             response_input(n0,1) = hod_target_input(n0,1)   ! function value
             n0=n0+1
          end do
          close(ut)
          label_end(i)=n0-1

       else if ( form_train(i)=="PINN" ) then
          label_start(i)=n0
          open(ut,file=file_train(1,i),status='old')
          ! each line: collocation point x(1:ndim(1))
          !   + the source f(x) if the residual declares a SRC term
          !   + the exact u if Exact_solution=1 (reporting only, never trained on)
          ! The exact-solution column is one per field component, so a
          ! system carries ndim(Nlayer) of them and a scalar field one.
          ! A system carries one source per residual, since each residual
          ! is a separate equation and may be inhomogeneous on its own; a
          ! scalar field carries at most one.  The exact-solution columns
          ! are one per field component.
          nexa = 0
          if ( iswitch_pinn_exact == 1 ) nexa = ndim(Nlayer)
          ! Source columns are present only when the case says so.  A
          ! homogeneous system has none, and assuming one per residual
          ! would make its collocation file the wrong width; Sys_src
          ! declares them.
          nsrc = 0
          if ( sys_nterm > 0 ) then
             if ( sys_use_src ) nsrc = sys_nres
          else if ( pinn_has_src ) then
             nsrc = 1
          end if
          itmp = ndim(1) + nsrc + nexa
          do n=1,Ndata_train_set(i)
             call read_checked_record( ut, file_train(1,i), n, itmp, recbuf )
             descriptor_input(n0,1:ndim(1)) = recbuf(1:ndim(1))
             jj = ndim(1)
             if ( sys_nterm > 0 .and. sys_use_src ) then
                sys_src_input(n0,1:sys_nres) = recbuf(jj+1:jj+sys_nres)
                jj = jj + sys_nres
             else if ( pinn_has_src ) then
                jj = jj + 1
                pinn_src(n0) = recbuf(jj)
             end if
             if ( nexa > 0 ) then
                response_input(n0,1:nexa) = recbuf(jj+1:jj+nexa)
             end if
             n0=n0+1
          end do
          close(ut)
          label_end(i)=n0-1

       end if
    end do

!---
!------------------------------------------------------------!
!***** Prepare x_t,y_t & x_val,y_val from x_in,y_in for early stopping *****!
    allocate( ind_validation(NUM_validation)  ); ind_validation=1
    allocate( ind_train(NUM_train)  ); ind_train=1
    !choose validation data, 1:N_val
    call pre_random !Share same random seed if MPI is used
    do i=1,NUM_validation
       if (i==1) then
          call random_irange(ibuf,1,NUM_input)
       else
          i_flag=0
          do while ( i_flag == 0 )
             call random_irange(ibuf,1,NUM_input)
             i_flag=1
             do j=1,i-1
                if ( ibuf == ind_validation(j) ) then
                   i_flag=0
                end if
             end do
          enddo
       end if
       ind_validation(i)=ibuf
    end do

    ndrawn = 0
    do i=1,NUM_input
       i_flag=1
       do j=1,NUM_validation
          if ( i == ind_validation(j) ) i_flag=0
       end do

       if ( i_flag == 1 ) then
          ndrawn = ndrawn + 1
          ind_train(ndrawn) = i
       end if
    end do

    ! The validation points are drawn from the pooled index space of all
    ! sets, so a small set can lose every training point to the draw,
    ! which would make its per-set costs undefined.
    ! Positions of each set inside ind_train, for the stratified draw.
    ! ind_train is ascending in the original index and the sets are
    ! consecutive blocks of it, so each set is one run of positions.
    if ( .not. allocated(set_first) ) then
       allocate( set_first(Ntot_train_set), set_last(Ntot_train_set), &
                 nset_train(Ntot_train_set) )
    end if
    set_first = 0;  set_last = -1;  nset_train = 0
    do j=1,Ntot_train_set
       do i=1,ndrawn
          if ( (label_start(j)<=ind_train(i)) .and. &
               (ind_train(i)<=label_end(j)) ) then
             if ( set_first(j) == 0 ) set_first(j) = i
             set_last(j) = i
             nset_train(j) = nset_train(j) + 1
          end if
       end do
    end do
    do j=1,Ntot_train_set
       ! A term may ask for more points than it kept, because the
       ! validation draw is over the pooled index space and its split
       ! moves with the seed.  Asking for the whole set is the common
       ! case (a small boundary set that should enter every gradient in
       ! full), so it is clamped rather than refused; a request far
       ! above the set is a mistake and still stops.
       if ( nbatch_set(j) > nset_train(j) ) then
          if ( nbatch_set(j) <= Ndata_train_set(j) ) then
             write(*,'(a,i0,a,i0,a,i0,a)') " Loss_term ", j, ": batch size ", &
                  nbatch_set(j), " clamped to its ", nset_train(j), &
                  " training points (validation took the rest)"
             nbatch_set(j) = nset_train(j)
          else
             write(*,*) "Loss_term", j, ": batch size", nbatch_set(j), &
                  " exceeds its", Ndata_train_set(j), " points"
             stop
          end if
       end if
       itmp = nset_train(j)
       if ( itmp == 0 ) then
          write(*,*) "input: the validation draw left set", j, " with no"
          write(*,*) "  training points; reduce Num_validation (each set"
          write(*,*) "  keeps at least one training point)"
          stop
       end if
    end do
    if ( any( nbatch_set(1:Ntot_train_set) > 0 ) ) then
       NUM_batch = sum( nbatch_set(1:Ntot_train_set) )
       write(*,'(a,i0,a)') " Num_batch is the sum of the per-term sizes: ", &
            NUM_batch, " points per epoch"
    end if

#ifdef _MPI_
    if (myrank==0) then
       write(*,*) "myrank=",myrank,"ind_val(1:5)",ind_validation(1:5)
       write(*,*) "myrank=",myrank,"ind_t(1:5)",ind_train(1:5)
    else if (myrank==1) then
       write(*,*) "myrank=",myrank,"ind_val(1:5)",ind_validation(1:5)
       write(*,*) "myrank=",myrank,"ind_t(1:5)",ind_train(1:5)
    end if
#endif


     if ( iswitch_fit == 0 ) then
       open(uw,file='nn_weight.dat',status='old')
       read(uw,*) istart_step
       read(uw,'()') !i_func
       read(uw,'()') !Softplus_out
       read(uw,'()') !Nlayer
       do i=1,Nlayer
          read(uw,'()') !ndim
       end do
       do i=2,Nlayer
          read(uw,'()')
          do j=1,ndim(i)
             read(uw,*) weight_best(i,j,0:ndim(i-1))
          end do
       end do
       close(uw)
    end if
    
    deallocate( x_tmp )


    !--- check ind_val & ind_t ---!
#ifdef _MPI_
    if (myrank<=3) then
       write(cbuf,'("data_division.log",i3.3)') myrank
       cbuf=trim(cbuf)
       open(10,file=cbuf,status='replace')
       write(10,*) "#validation"
       do i=1,NUM_validation
          write(10,*) ind_validation(i)
       end do
       write(10,*) "#training"
       do i=1,NUM_train
          write(10,*) ind_train(i)
       end do
       close(10)
    end if
#else
    open(10,file="data_division.log",status='replace')
    write(10,*) "#validation"
    do i=1,NUM_validation
       write(10,*) ind_validation(i)
    end do
    write(10,*) "#training"
    do i=1,NUM_train
       write(10,*) ind_train(i)
    end do
    close(10)
#endif


    allocate( zmat(Nlayer,0:ndim_max) ); zmat=0.d0


  END SUBROUTINE read_data


  SUBROUTINE write_data(cbuf)
    implicit none
    character(*),intent(IN) :: cbuf
    integer,parameter :: uo=24,uw=25
    integer :: i,j,n,is,ie,itmp,i_func
    real(8) :: vbuf
    real(8),allocatable :: tmwrite(:,:)
    character(40) :: fname
    character(40) :: file


    ! The activation actually trained with is recorded, so that a reader
    ! cannot evaluate the file with the wrong analytics.  The codes are
    ! the ones the library uses internally: 0 TANH, 1 SIN, 2 ERF,
    ! 3 BESSEL (J_0), 4 BESSEL1 (J_1).  The output layer is linear by the
    ! HOD contract, so that field stays 0.
    select case ( trim(Activation_type) )
    case ( "SIN" );     i_func = 1
    case ( "ERF" );     i_func = 2
    case ( "BESSEL" );  i_func = 3
    case ( "BESSEL1" ); i_func = 4
    case default;       i_func = 0
    end select
    itmp   = 0

    if ( cbuf == "weight_best" ) then
       
       open(uw,file='nn_weight.dat',status='replace')
       write(uw,'(i0)') epoch_best
       write(uw,'(a4,1x,i0)') "func",i_func
       write(uw,'(a14,1x,i0)') "Activation_out",itmp

       write(uw,'(i0)') Nlayer
       do i=1,Nlayer
         write(uw,'(i0)') ndim(i)
       end do

       do i=2,Nlayer
          write(uw,*) "#l=",i
          do j=1,ndim(i)
             write(uw,'(1000(1x,es24.17))') weight_best(i,j,0:ndim(i-1))
          end do
       end do
       close(uw)
       
    else if ( cbuf == "weight_log" ) then

       ! Periodic checkpoint, one file per output epoch, in the same
       ! format as nn_weight.dat: copying a checkpoint to nn_weight.dat
       ! restarts from that epoch.
       write(fname,'("checkpoint_ep",i7.7,".dat")') i_epoch_now
       open(uw,file=trim(fname),status='replace')
       write(uw,'(i0)') i_epoch_now
       write(uw,'(a4,1x,i0)') "func",i_func
       write(uw,'(a14,1x,i0)') "Activation_out",itmp
       write(uw,'(i0)') Nlayer
       do i=1,Nlayer
         write(uw,'(i0)') ndim(i)
       end do

       do i=2,Nlayer
          write(uw,*) "#l=",i
          do j=1,ndim(i)
             write(uw,'(1000(1x,es24.17))') weight(i,j,0:ndim(i-1))
          end do
       end do
       close(uw)

    else if ( cbuf == "output" ) then
       ! outputs are written from the best weights, so the library
       ! network is pointed at that array first
       call lnet_sync_best( weight_best )
       
       ! Feedforward
       write(*,'(a,i0,a)') " writing output_set*.dat for ", NUM_input, " points"

! sequentially numbered output.dat1, ... , output.dat[Ntrain_set]
       do j=1,Ntot_train_set
          !label_start(j):label_end(j)
          is = label_start(j)
          ie = label_end(j)

          write(fname, '("output_set",i4.4,".dat")') j
          fname=trim(fname)
          open(uo,file=fname,status='replace')
          close(uo)

          open(uo,file=fname,position='append')
          do n=is,ie
             ! one prediction per field component, then the reference
             ! values the file carried, so a system writes 2*ndim(Nlayer)
             ! columns after the coordinates
             if ( ndim(Nlayer) > 1 ) then
                if ( .not. allocated(tmwrite) ) &
                     allocate( tmwrite(ndim(Nlayer),NUM_alpha) )
                call lnet_forward_hod_multi( descriptor_input(n,1:ndim(1)), &
                     tmwrite )
                write(uo,'(1000es20.10)') descriptor_input(n,1:ndim(1)), &
                     tmwrite(1:ndim(Nlayer),1), &
                     response_input(n,1:ndim(Nlayer))
             else
                call lnet_forward_value( descriptor_input(n,1:ndim(1)), vbuf )
                write(uo,'(1000es20.10)') descriptor_input(n,1:ndim(1)), &
                     vbuf, response_input(n,1:ndim(Nlayer))
             end if
          end do
          close(uo)
       end do
       
    end if

  END SUBROUTINE write_data

!write derivatives
  SUBROUTINE write_deriv
    implicit none
    integer :: n, j, is, ie
    integer,parameter :: ud=26
    integer :: i
    real(8),allocatable :: dbuf(:), tvec(:)
    character(40) :: fname

    allocate( dbuf(ndim(1)) );  dbuf = 0.d0
    ! Synchronize first: for a plain DATA case the module-level tables
    ! are empty until lnet_sync_best builds them, so sizing tvec from
    ! NUM_alpha beforehand gives a buffer of length one for a forward
    ! sweep that then writes lnet_nalpha() slots into it.
    call lnet_sync_best( weight_best )
    allocate( tvec(max(lnet_nalpha(),1)) );  tvec = 0.d0

    do j=1,Ntot_train_set

!label_start(j):label_end(j)
       is = label_start(j)
       ie = label_end(j)

       if ( form_train(j) == "MATH" ) then
          write(fname, '("output_deriv_set",i4.4,".dat")') j
          fname=trim(fname)
          open(ud,file=fname,status='replace')
          write(ud,*) "#x(1:ndim1), dN/dx(1:ndim1)"
          do n=is,ie
             call lnet_forward_hod( descriptor_input(n,1:ndim(1)), tvec )
             do i=1,ndim(1)
                dbuf(i) = tvec(ind_e1(i))
             end do
             write(ud,'(2000es22.12)') descriptor_input(n,1:ndim(1)), dbuf(1:ndim(1))
          end do
          close(ud)
       else if ( form_train(j) == "MATH_HOD" ) then
          ! all high-order derivatives, predicted vs target, at the best weight
          write(fname, '("output_hod_set",i4.4,".dat")') j
          fname=trim(fname)
          open(ud,file=fname,status='replace')
          write(ud,*) "#x(1:ndim1), T_pred(1:NUM_alpha), y_target(1:NUM_alpha)"
          do n=is,ie
             call lnet_forward_hod( descriptor_input(n,1:ndim(1)), tvec )
             write(ud,'(2000e22.12)') descriptor_input(n,1:ndim(1)), &
                  tvec(1:NUM_alpha), hod_target_input(n,1:NUM_alpha)
          end do
          close(ud)
       end if
!       end if
    end do

    deallocate( dbuf, tvec )
  END SUBROUTINE write_deriv

  SUBROUTINE write_nn_param
    implicit none
    integer :: i
    integer :: num_W !number of elements of weight matrix
    num_W=0
    if ( pinn_nterm > 0 ) then
       write(*,'(a)') "### PINN residual training enabled"
       write(*,'(a,2x,i0)') "Residual terms (after DXLAP expansion)", pinn_nterm
       write(*,'(a,2x,i0)') "Exact_solution", iswitch_pinn_exact
       write(*,'(a,2x,i0)') "Time_axis", merge(pinn_time_axis,ndim(1),pinn_time_axis>0)
    end if
    if ( hod_kmax_in > 0 ) then
       write(*,'(a)') "### High-order derivative (HOD) training enabled"
       write(*,'(a,2x,i0)') "Hod_K", hod_kmax_in
       write(*,'(a,2x,20e11.3)') "Hod_lambda(0:K)", lambda_hod(0:hod_kmax_in)
       write(*,'(a,2x,a)') "Hod_alpha_file", trim(hod_alpha_file)
       write(*,'(a,2x,i0)') "Hod_check", iswitch_hod_check
    end if
    if ( iswitch_fit == 1 ) then
       write(*,'(a,2x,a)') "Task", "TRAIN"
    else if ( iswitch_fit == 0 ) then
       write(*,'(a,2x,a)') "Task", "PREDICT"
    else
       write(*,'(a,2x,a)') "Task", "COMMITTEE"
    end if
    write(*,'(a,2x,i0)') "Average_cyc_mpi",Average_cyc_mpi
    write(*,'(a,2x,i0)') "Shuffle",iswitch_shuffle
    write(*,'(a,2x,i0)') "Shuffle_cyc_mpi",Shuffle_cyc_mpi
    write(*,'(a,2x,i0)') "Output_deriv",iswitch_out_deriv
    write(*,'(a,2x,i0)') "Restart",iswitch_restart
    write(*,'()')
    write(*,'(a)') "********************"
    write(*,'(a)') "Network Construction"
    write(*,'(a,2x,i0)') "Nlayer",Nlayer
    do i=1,Nlayer
       write(*,*) ndim(i)
    end do
    write(*,'(a)') "********************"

    do i=1,Nlayer-1
       num_W = num_W + ndim(i+1)*( ndim(i)+1 )
    end do
    write(*,'(a,2x,i0)') "Number of weight parameters",num_W !recently added

    write(*,'()')
    write(*,'(a,2x,i0)') "All input",NUM_input
    write(*,'(a,2x,i0)') "Training",NUM_train
    write(*,'(a,2x,i0)') "Validation",NUM_validation
    write(*,'(a)') "********************"
    write(*,'(a)') "Label_start & Label_end"
    do i=1,Ntot_train_set
       write(*,*) label_start(i),label_end(i)
    end do
    write(*,'(a)') "Ndata_train_set(i) & label_end(i)-label_start(i)+1"
    do i=1,Ntot_train_set
       write(*,*) Ndata_train_set(i),label_end(i)-label_start(i)+1
    end do


    write(*,'(a)') "********************"
    write(*,'(a)') "Early Stopping"
    write(*,'(a,2x,i0)') "NUM_validation",NUM_validation
    write(*,'(a,2x,i0)') "val_cyc",validation_cyc
    write(*,'(a,2x,i0)') "p_max: maximum patience for early stopping",patience_max
    write(*,'(a)') "********************"

    write(*,'(a,2x,i0)') "Epoch",Num_loop
    write(*,'(a,2x,E10.2)') "Conv",conv_fit
    write(*,'(a)') "********************"
    write(*,'(a)') "Regularization"

    do i=1,Ntot_train_set
    end do
    write(*,'(a)') "********************"
    write(*,'()')

!--- fixed weight

    write(*,'(a,2x,a)') "Activation",Activation_type
    write(*,'()')

    write(*,'(a,2x,i0)') "Sample number in a minibatch (Mbatch)",NUM_batch
    write(*,'()')

    if ( iswitch_fit == 1 ) then
       write(*,'(a,2x,i0)') "Output Cycle (OC)",io_cyc
       write(*,'(a,2x,a)') "Initialization", init_weight_method
       write(*,'(a,2x,a)') "GD_method", gd_method
       write(*,'(a,2x,7f15.10)') "GD_param", gd_param(1:)
       write(*,'(a)') "GD_ratio"
       do i=1,Ntot_train_set
          write(*,'(f15.10)') gd_ratio(i)
       end do
    end if

    write(*,'()')
  END SUBROUTINE write_nn_param

  ! append the multinomial expansion of  c * d_x Lap^j u  to the PINN term list
  ! (nondecreasing choice of j spatial axes; axis v chosen n_v times ->
  !  multi-index e_x + sum_v 2 n_v e_v with weight j!/prod n_v!)
  !> Expand c * d_{ixd} Lap^j u into multi-index terms.  The Laplacian
  !! runs over the nsp axes listed in sax, which are the axes that are
  !! not time.  With the default layout (time last, ixd = 1) sax is
  !! (1,...,ndim(1)-1) and the expansion is the one the shipped cases use.
  SUBROUTINE expand_dxlap( c, j, ixd, nsp, sax )
    implicit none
    real(8),intent(IN) :: c
    integer,intent(IN) :: j, ixd, nsp
    integer,intent(IN) :: sax(nsp)
    real(8) :: fact(0:12)
    integer :: spatial, axes(j), v, ip

    spatial = nsp
    fact(0)=1.d0
    do v=1,12
       fact(v)=fact(v-1)*dble(v)
    end do
    axes = 1
    call dxlap_rec( c, j, 1, 1, axes, spatial )

  CONTAINS

    RECURSIVE SUBROUTINE dxlap_rec( c, j, depth, vmin, axes, spatial )
      implicit none
      real(8),intent(IN) :: c
      integer,intent(IN) :: j, depth, vmin, spatial
      integer,intent(INOUT) :: axes(j)
      integer :: v, ip, m, wnum
      integer :: cnt(spatial)

      do v=vmin,spatial
         axes(depth) = v
         if ( depth == j ) then
            cnt = 0
            do ip=1,j
               cnt(axes(ip)) = cnt(axes(ip))+1
            end do
            wnum = 1
            do ip=2,j
               wnum = wnum*ip
            end do
            do ip=1,spatial
               do m=2,cnt(ip)
                  wnum = wnum/m
               end do
            end do
            pinn_nterm = pinn_nterm+1
            if ( pinn_nterm > 64 ) then
               write(*,*) "Pinn_residual: too many terms after DXLAP expansion"
               stop
            end if
            pinn_alpha(:,pinn_nterm) = 0
            do ip=1,spatial
               pinn_alpha(sax(ip),pinn_nterm) = 2*cnt(ip)
            end do
            pinn_alpha(ixd,pinn_nterm) = pinn_alpha(ixd,pinn_nterm)+1  ! leading d_ixd
            pinn_coeff(pinn_nterm) = c*dble(wnum)
            pinn_nonlin(pinn_nterm) = .false.
         else
            call dxlap_rec( c, j, depth+1, v, axes, spatial )
         end if
      end do
    END SUBROUTINE dxlap_rec

  END SUBROUTINE expand_dxlap

  SUBROUTINE convert_capital(cbuf,CKEY)
    implicit none
    character(*),intent(IN)  :: cbuf
    character(*),intent(OUT) :: CKEY
    integer :: j,k,n

    n=len_trim(cbuf)
    CKEY=""
    do j=1,n
       k=iachar( cbuf(j:j) )
       if ( k >= 97 ) k=k-32
       CKEY(j:j) = achar(k)
    end do
    
  END SUBROUTINE convert_capital



  !> Parse ntok values from one physical line into the record n0 at
  !! token offset noff (descriptor first, then the hod targets).
  !> Read one logical record of nexp numbers into val(1:nexp).
  !!
  !! A record may wrap over several physical lines, so lines are taken
  !! until the token count reaches nexp.  A record with too few values
  !! would otherwise be completed silently from the next line by a plain
  !! list-directed read, frame-shifting every later record: the symptom
  !! is an end-of-file crash with no location, or nothing at all when the
  !! file has spare lines.  Here the shortfall shows up as an overshoot,
  !! because the continuation pulls the next record in.
  SUBROUTINE read_checked_record( ut, fname, nrec, nexp, val )
    implicit none
    integer,intent(IN) :: ut, nrec, nexp
    character(*),intent(IN) :: fname
    real(8),intent(OUT) :: val(nexp)
    character(65536) :: crec
    integer :: ios_rec, ntok_line, ntok_acc

    ntok_acc = 0
    do while ( ntok_acc < nexp )
       read(ut,'(a)',iostat=ios_rec) crec
       if ( ios_rec /= 0 ) then
          write(*,*) "read_data: file ", trim(fname), " ends inside record", &
               nrec, " (", ntok_acc, " of", nexp, " values found)"
          stop
       end if
       if ( len_trim(crec) >= len(crec) ) then
          write(*,*) "read_data: a line of ", trim(fname), " exceeds the ", &
               len(crec), " character buffer"
          stop
       end if
       ntok_line = count_tokens( crec )
       if ( ntok_line == 0 ) cycle
       if ( ntok_acc+ntok_line > nexp ) then
          write(*,*) "read_data: record", nrec, " of ", trim(fname)
          write(*,*) "  accumulates", ntok_acc+ntok_line, " values where", &
               nexp, " are required: a line is missing a value"
          stop
       end if
       read(crec,*) val(ntok_acc+1:ntok_acc+ntok_line)
       ntok_acc = ntok_acc + ntok_line
    end do
  END SUBROUTINE read_checked_record

  SUBROUTINE parse_record_part( c, ntok, n0, noff )
    implicit none
    character(*),intent(IN) :: c
    integer,intent(IN) :: ntok, n0, noff
    real(8) :: vals(ntok)
    integer :: k, kk
    read(c,*) vals(1:ntok)
    do k=1,ntok
       kk = noff + k
       if ( kk <= ndim(1) ) then
          descriptor_input(n0,kk) = vals(k)
       else
          hod_target_input(n0,kk-ndim(1)) = vals(k)
       end if
    end do
  END SUBROUTINE parse_record_part


  !> number of blank-separated tokens in a record
  !> Write the residual back in readable form.  The operator is the one
  !! thing in the input that cannot be checked against anything else: a
  !! mistyped multi-index gives a different equation, solved correctly.
  !! Echoing the parsed form lets the reader compare it with what they
  !! meant, and the carried slot lets them cross-check
  !! hod_alpha_order.dat.
  SUBROUTINE echo_residual( ita )
    implicit none
    integer,intent(IN) :: ita
    integer :: k, i, m
    character(160) :: expr
    character(8)   :: lab
    if ( pinn_nterm == 0 .and. .not. pinn_has_src ) return
    write(*,'(a)') "### Residual as parsed (imposed as R = 0 at every collocation point)"
    write(*,'(a,i0,a,i0,a)') "###   axes are written x1..x", ndim(1), &
         ", with axis ", ita, " shown as t (Time_axis)"
    write(*,'(a)') "###   term      coefficient  expression                    slot  multi-index"
    do k=1,pinn_nterm
       expr = "u"
       if ( pinn_nonlin(k) ) expr = "u * u"
       do i=1,ndim(1)
          if ( pinn_alpha(i,k) == 0 ) cycle
          if ( i == ita ) then
             lab = "t"
          else
             write(lab,'("x",i0)') i
          end if
          if ( index(trim(expr),"_") == 0 ) expr = trim(expr)//"_"
          do m=1,pinn_alpha(i,k)
             expr = trim(expr)//trim(lab)
          end do
       end do
       write(*,'(a,i5,3x,es15.6,2x,a30,i5,2x,10i3)') "###", k, pinn_coeff(k), &
            adjustl(expr(1:30)), pinn_ind(k), pinn_alpha(1:ndim(1),k)
    end do
    if ( pinn_has_src ) then
       write(*,'(a,a5,3x,es15.6,2x,a)') "###", "  src", pinn_src_coeff, &
            "f(x)  (a column of the collocation file)"
    end if
    write(*,'(a)') "###"
  END SUBROUTINE echo_residual

  INTEGER FUNCTION count_tokens( c )
    implicit none
    character(*),intent(IN) :: c
    integer :: i
    logical :: inword
    count_tokens = 0;  inword = .false.
    do i=1,len_trim(c)
       if ( c(i:i) /= " " .and. c(i:i) /= char(9) ) then
          if ( .not. inword ) count_tokens = count_tokens + 1
          inword = .true.
       else
          inword = .false.
       end if
    end do
  END FUNCTION count_tokens

END MODULE io_module

