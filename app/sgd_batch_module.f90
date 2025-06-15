!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (sgd_batch_module.f90) is part of DNNF90.
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
!Stochastic gradient descent with batch data
MODULE sgd_batch_module

  ! The parallel module carries serial defaults (one rank, whole range,
  ! no reduction), so the whole-set sweeps below need no conditionals.
  use parallel_module
  use rand_module,only: random_irange
  use global_variables
  use multi_index_bell_module
  use io_module
  use lib_net_module, only: lnet_sync_weights, lnet_value_grad, &
       lnet_forward_hod, lnet_seed_grad, lnet_seed_row, lnet_value_row, &
       lnet_forward_hod_multi, lnet_seed_grad_multi, &
       lnet_value_grad_multi, lnet_seed_row_multi, &
       lnet_kalman_resid_multi, lnet_kalman_iekf_begin, lnet_kalman_iekf_iter, &
       lnet_batch_grad, lnet_kalman_init, lnet_kalman_slot, &
       lnet_kalman_resid, lnet_kalman_active, lnet_export_weights, &
       lnet_nweights
  use init_weight_module
  use optimizer_module
  use blas_wrap_module, only: bgemm, bposv
  use validation_module

  use pinn_module, only: calc_pinn_residual, set_pinn_seed, &
       calc_sys_residual, set_sys_seed
  implicit none

  PRIVATE
  PUBLIC :: sgd_minibatch

  ! Membership marks for drawing distinct indices.  The draw itself is
  ! unchanged (same random calls, same accept and reject decisions, so
  ! the selected sets are identical); only the duplicate test becomes
  ! O(1) instead of a scan over everything drawn so far.  The scan made
  ! one epoch cost O(NUM_BATCH^2), and O(N^2 log N) for full batches,
  ! which dominates the run once the training set reaches force field
  ! sizes.  A draw is marked with the current tag, so no array has to be
  ! cleared between epochs.
  integer,allocatable :: seen_stamp(:)
  integer :: seen_tag = 0

#ifdef USE_BLAS
  ! scratch of the batched plain pass (see batched_math_pass)
#endif

CONTAINS


  SUBROUTINE sgd_minibatch
    implicit none
    integer :: loop, i, ib, ibuf, i_flag, n, j, nbdrawn, nbset
    integer :: nloc, lo, hi, myrank_l, nprocs_l, ncand, itmp2
    integer :: nbset_g, nbdrawn_g, nrem, nb_use
    integer,allocatable,save :: cand(:)
    integer,allocatable :: ind_batch(:)
    integer :: init_step
    !
    real(8),allocatable :: nabla(:, :, :)
    !
    real(8),allocatable :: dweight(:, :, :)
    real(8),allocatable :: dweight_zero(:, :, :) ! 0 at origin
    real(8),allocatable :: weight_old(:, :, :) !for Nesterov
    real(8),allocatable :: gd_param_r(:, :, :) !for AdaGrad
    real(8),allocatable :: gd_param_u(:, :, :)
    real(8),allocatable :: gd_param_v(:, :, :)
    real(8),allocatable :: gd_param_m(:, :, :)
!---
    real(8) :: Etrain_best !training error
    real(8) :: Eval_old !validation error
    integer :: patience, it
!---
    real(8) :: Etrain_update, Eval_update
!---
    real(8),allocatable :: dweight_reg(:, :, :)
!---
    integer :: ib_ntrain, itmp, is, ir
    real(8) :: rbuf
    integer :: Nbatch_s, Nbatch_e, ierr, Ndata
    integer,allocatable :: ista(:)
    integer,allocatable :: ind_tmp(:)
!---
    integer :: i0
!----
    logical :: status_deriv_train
    logical :: gdlog_ok
    integer :: istart_weight
    integer :: iadam_base
    integer :: loop_save !for deriv_train_len
    integer :: istop_local, istop_global
    real(8),allocatable :: jrow(:, :, :, :) ! per-point rows for NATURAL_GRAD
    real(8),allocatable :: Xbat(:, :), Ybat(:) ! staging of one minibatch
    real(8),allocatable :: tvec(:), svec(:)
    !> Working arrays of the multi-component path: the carried
    !! derivatives of every field component, the seed per component, the
    !! residuals of the system and its source values.
    real(8),allocatable :: tmat(:,:), smat(:,:), rsys(:), syssrc(:)
    real(8),allocatable :: rsel(:)   ! selects one residual of the system
    real(8) :: lbfgs_cost
    integer :: iit
    integer :: nrow_pt, irow
    logical :: lbfgs_ok
    ! geodesic acceleration (Ngd_geo)
    real(8),allocatable,save :: rowval(:), avec(:), jdotu(:)
    real(8) :: u1(Nlayer,ndim_max,0:ndim_max), u2(Nlayer,ndim_max,0:ndim_max)
    real(8) :: nabla2(Nlayer,ndim_max,0:ndim_max), wsave(Nlayer,ndim_max,0:ndim_max)
    real(8) :: phih, n1sq, n2sq, hga, etag, wsq ! one point's outputs and seed
    real(8),allocatable :: nabla_save(:, :, :)
    ! Weight-shaped cubes, on the heap rather than the stack: thirteen
    ! of them at once is hundreds of megabytes at force-field widths,
    ! which killed the trainer before the first epoch under the default
    ! stack limit.
    allocate( nabla(Nlayer,ndim_max,0:ndim_max) )
    allocate( dweight(Nlayer,ndim_max,0:ndim_max) )
    allocate( dweight_zero(Nlayer,ndim_max,0:ndim_max) )
    allocate( weight_old(Nlayer,ndim_max,0:ndim_max) )
    allocate( gd_param_r(Nlayer,ndim_max,0:ndim_max) )
    allocate( gd_param_u(Nlayer,ndim_max,0:ndim_max) )
    allocate( gd_param_v(Nlayer,ndim_max,0:ndim_max) )
    allocate( gd_param_m(Nlayer,ndim_max,0:ndim_max) )
    allocate( dweight_reg(Nlayer,ndim_max,0:ndim_max) )
    allocate( nabla_save(Nlayer,ndim_max,0:ndim_max) )

    loop_save=-deriv_train_len

    allocate( ind_batch(NUM_BATCH) ); ind_batch=1
    allocate( ind_tmp(NUM_BATCH) ); ind_tmp=1

    call get_initial_weight !know restart or not
    ! Restarting from a weight file alone is the normal case, and
    ! read_gdlog opens its files with status='old', so it aborted the run
    ! with a bare "Cannot open file 'gd_dw.dat'".  The optimizer state is
    ! restored when its log is there and started cold otherwise.
    gdlog_ok = .false.
    if ( iswitch_restart==1 ) then
       inquire( file='gd_dw.dat', exist=gdlog_ok )
       if ( gdlog_ok ) then
          istart_weight = istart_step
          call read_gdlog( dweight,gd_param_r,gd_param_u,gd_param_v,gd_param_m,istart_step )
          ! The optimizer log carries its own epoch, and reading it above
          ! has just overwritten the one that came with the weights.  If
          ! the two disagree the directory holds a mismatched pair, for
          ! example a weight file copied back from an older checkpoint
          ! next to the optimizer state of a later run.  Continuing would
          ! silently combine the weights of one epoch with the moments
          ! and the counter of another.
          if ( istart_weight /= istart_step ) then
             write(*,*) "restart: nn_weight.dat is at epoch", istart_weight, &
                  " but gd_dw.dat is at epoch", istart_step
             write(*,*) "  These belong to different points of the run."
             write(*,*) "  To resume from this weight file, delete the"
             write(*,*) "  optimizer log (gd_*.dat) and the state will"
             write(*,*) "  start cold; to resume the logged run, restore"
             write(*,*) "  the matching nn_weight.dat."
             stop
          end if
          write(*,'(a,2x,i0)') &
               "### restart: optimizer state restored at epoch", istart_step
       else
          write(*,'(a)') "### restart: no gd_dw.dat, the optimizer state starts cold"
          write(*,'(a)') "###   weights and the epoch counter come from nn_weight.dat"
       end if
    end if
    call alloc_for_validation

#ifdef _MPI_
if (myrank==0) then
    call prep_outfiles
end if
#else
  call prep_outfiles
#endif


    patience = 0

    ! The whole-set methods take one step from one state, so their
    ! sweeps may be split across ranks and summed (see replicated_step
    ! in parallel_module).  The minibatch rules may not: between two
    ! averaging points their weights differ per rank.
    replicated_step = ( gd_method == "LBFGS" .or. gd_method == "LM" .or. &
                        gd_method == "NATURAL_GRAD" )

    ! One rank unless the MPI build says otherwise; the per-term draw
    ! below uses these, and with one rank it takes the whole term.
    myrank_l = 0
    nprocs_l = 1
#ifdef _MPI_
    allocate(ista(MPI_STATUS_SIZE))
    call init_parallel(NUM_train)
    myrank_l = myrank
    nprocs_l = nprocs
    Nbatch_s = JSTA(myrank)
    Nbatch_e = JEND(myrank)
!    Nbatch_s = 1
!    Nbatch_e = NUM_train
    if (myrank==0) then
       do i=0,nprocs-1
          write(*,*) "rank",i,"JSTA",JSTA(i),"JEND",JEND(i)
       end do
    end if
    Ndata = Nlayer*ndim_max*(1+ndim_max)

    if (myrank==0) then

       do i=1,nprocs-1
          call mpi_send(weight(1,1,0),Ndata,mpi_real8,i,0,mpi_comm_world,ierr)
       end do
    else
       call mpi_recv(weight(1,1,0),Ndata,mpi_real8,0,0,mpi_comm_world,ista,ierr)
    end if
#else
    Nbatch_s = 1
    Nbatch_e = NUM_train
#endif

    init_step = istart_step

    ! allocate global variables: amat,zmat,delmat,gmmat,epsmat,zetamat
    ! ********** !
    ! However, the layer index is not needed for the staging buffers.
    ! ********** !
    if ( gd_method=="NATURAL_GRAD" ) then
       ! One Gauss-Newton row per RESIDUAL, not per point.  The metric
       ! of a least-squares objective is J^T J with J the Jacobian of
       ! the residual vector; collapsing a point's residuals into the
       ! single row d(sum_r R_r^2/2)/dw builds the outer product of a
       ! sum instead of the sum of outer products, which is a rank-one
       ! caricature of the true metric (the empirical Fisher).  With
       ! sys_nres residuals, or ndim(Nlayer) fitted components, the
       ! point contributes that many rows.  Under Ngd_dual the cost of
       ! the extra rows is the Gram solve, which stays small at
       ! minibatch sizes, so the better metric is nearly free.
       nrow_pt = 1
       if ( sys_nterm > 0 ) then
          nrow_pt = sys_nres
       else if ( ndim(Nlayer) > 1 ) then
          nrow_pt = ndim(Nlayer)
       end if
       allocate( jrow(NUM_batch*nrow_pt,Nlayer,ndim_max,0:ndim_max) )
       jrow=0.d0
       if ( nrow_pt > 1 ) write(*,'(a,i0,a,i0,a)') &
            "### NATURAL_GRAD: ", nrow_pt, " Gauss-Newton rows per point (", &
            NUM_batch*nrow_pt, " rows)"
    else
       ! Not a natural-gradient run: the driver receives this dummy and
       ! never reads it.  nrow_pt is set here as well, because the call
       ! below forms a section from it on every path.
       nrow_pt = 0
       allocate( jrow(1,1,1,0:0) ); jrow=0.d0   ! dummy (unused)
    end if
    if ( .not. allocated(Xbat) ) then
       allocate( Xbat(ndim(1),NUM_batch), Ybat(NUM_batch) )
       Xbat = 0.d0;  Ybat = 0.d0
       allocate( tvec(max(NUM_alpha,1)), svec(max(NUM_alpha,1)) )
       ! the multi-component path needs one row per field component
       allocate( tmat(max(ndim(Nlayer),1),max(NUM_alpha,1)) )
       allocate( smat(max(ndim(Nlayer),1),max(NUM_alpha,1)) )
       allocate( rsys(max(sys_nres,1)), syssrc(max(sys_nres,1)) )
       allocate( rsel(max(sys_nres,1)) );  rsel = 0.d0
       tmat = 0.d0;  smat = 0.d0;  rsys = 0.d0;  syssrc = 0.d0
       tvec = 0.d0;  svec = 0.d0
    end if

    ib_ntrain=0

    Etrain_best = 1.d15
    Eval_old = 1.d15
    
    dweight_zero = 0.d0
    dweight_reg=0.d0
    ! Zeroing these unconditionally discarded whatever read_gdlog had just
    ! restored, which made the restore dead code while the epoch counter
    ! survived: Adam then bias-corrected cold moments with a large step
    ! index.  The moment age is tracked separately so the correction
    ! always matches the state actually carried.
    if ( gdlog_ok ) then
       iadam_base = init_step
    else
       iadam_base = 0
       gd_param_r=0.d0; gd_param_u=0.d0; gd_param_v=0.d0; gd_param_m=0.d0
       dweight(:,:,:) = 0.d0
    end if

    ! The L-BFGS line-search log is appended to, so that a restart
    ! continues one series.  A fresh run must not continue the previous
    ! one, though: leaving the file in place mixes generations, and the
    ! shipped reference then holds a series from a run that no longer
    ! matches the case.
    if ( gd_method == "LBFGS" .and. iswitch_restart /= 1 ) then
       open(203,file="lbfgs_history.dat",status='replace')
       write(203,'(a)') "# epoch  line-search objective  step  its  ok"
       close(203)
    end if

    epoch:  do loop=1,NUM_LOOP
       istart_step = init_step + loop
       iadam_step  = iadam_base + loop
       status_deriv_train=.false.

       if ( gd_method=="NESTEROV" ) then
          weight_old = weight
          weight = weight + gd_param(2)*dweight !gd_param(2)=alpha (lookahead)
       else if ( gd_method=="RMSPROP_NESTEROV" ) then
          weight_old = weight
          weight = weight + gd_param(4)*dweight !gd_param(4)=alpha (lookahead)
       end if

       ! Mirror the weights into the library once per batch iteration:
       ! the bridged per-point gradients below run on the library kernels.
       call lnet_sync_weights( weight )

       !choose batch, 1:NUM_BATCH
       if ( .not. allocated(seen_stamp) ) then
          allocate( seen_stamp(NUM_train) );  seen_stamp = 0
       end if
       if ( seen_tag > huge(0)-4 ) then       ! tag wrap (not reachable in practice)
          seen_stamp = 0;  seen_tag = 0
       end if
       seen_tag = seen_tag + 1
       ! Capacity is checked per Loss_term inside the draw below, since
       ! that is the granularity the draw works at.  A global check
       ! against this rank's contiguous span would be the wrong
       ! quantity: the span is not what the stratified draw reads.
       ! The draw is stratified by Loss_term.  Drawing from the pooled
       ! training set instead leaves the number of points of each set in
       ! the batch random: with 93 boundary points and 177 collocation
       ! points and a batch of twenty, the boundary count is
       ! hypergeometric with mean 6.9 and standard deviation 2.1, and it
       ! reaches zero often enough to matter.  The batch objective is
       ! then a different weighting of the terms at every step -- on the
       ! steps with no boundary point, one with no boundary condition at
       ! all -- and the iteration wanders even though the pooled gradient
       ! is still unbiased.  Drawing n_j = round(B |T_j| / |T|) from each
       ! set keeps the proportions fixed and the draw uniform inside each
       ! set, which is what the objective actually is.
       if ( .not. allocated(cand) ) allocate( cand(NUM_train) )
       ! With a global step the batch is one draw shared out among the
       ! ranks.  The stratification itself is the serial one -- every
       ! rank computes the same per-term counts from Num_batch -- and
       ! only the drawing is divided, so the union of the ranks' draws is
       ! exactly the batch a serial run would take.  Splitting the target
       ! instead and stratifying the smaller number does not: the
       ! rounding of one term moves, and the last term inherits it.
       ! With the local-SGD rules the count is per rank, as before.
       nbdrawn = 0
       nbdrawn_g = 0
       do j = 1, Ntot_train_set
          if ( nbatch_set(j) > 0 ) then
             nbset_g = nbatch_set(j)           ! this term's own size
          else if ( j < Ntot_train_set ) then
             nbset_g = nint( dble(NUM_BATCH)*dble(nset_train(j))/dble(NUM_train) )
             nbset_g = max( 1, min( nbset_g, nset_train(j) ) )
             nbset_g = min( nbset_g, NUM_BATCH - nbdrawn_g - (Ntot_train_set-j) )
          else
             nbset_g = NUM_BATCH - nbdrawn_g   ! the remainder, exactly
          end if
          nbset = nbset_g
          nbdrawn_g = nbdrawn_g + nbset_g
          nrem = mod( nset_train(j), nprocs_l )
          if ( replicated_step .and. nprocs_l > 1 ) then
             ! the same balanced rule as the candidate shares below, so
             ! that the counts sum to nbset_g and no rank is over-asked
             nbset = nbset_g/nprocs_l
             if ( myrank_l < mod(nbset_g,nprocs_l) ) nbset = nbset + 1
          end if
          if ( nbset_g > nset_train(j) ) then
             write(*,*) "batch: set", j, " holds", nset_train(j), &
                  " training points but the stratified draw needs", nbset_g
             write(*,*) "  lower Num_batch, or give that Loss_term more points"
             stop
          end if
          ! Build this term's candidate list explicitly, intersected with
          ! the rank's own partition, and take the first nbset of a
          ! partial Fisher-Yates shuffle.  A rejection loop over the
          ! whole term cannot terminate when the intersection is empty
          ! or smaller than nbset, which is reachable under MPI because
          ! the partition cuts across the terms.
          ! This rank's share of THIS term.  Splitting the pooled
          ! training set instead cuts across the terms: with the terms
          ! stored in consecutive blocks, a rank can end up holding none
          ! of one of them, and no draw can then be stratified.  Slicing
          ! inside each term gives every rank a share of every term, and
          ! reduces to the whole term when there is one rank.
          ! Balanced block distribution: the first mod(n,P) ranks hold
          ! one point more.  The draw below shares out its own remainder
          ! by the same rule, which is what keeps a rank from being asked
          ! for more points than it holds; giving the whole remainder to
          ! the last rank, as this did, breaks that agreement as soon as
          ! the remainder exceeds one.
          nloc = nset_train(j)/nprocs_l
          lo = set_first(j) + myrank_l*nloc + min( myrank_l, nrem )
          if ( myrank_l < nrem ) nloc = nloc + 1
          hi = lo + nloc - 1
          ncand = 0
          do i = lo, hi
             ncand = ncand + 1
             cand(ncand) = i
          end do
          if ( ncand < nbset ) then
             write(*,*) "batch: Loss_term", j, " needs", nbset, &
                  " points but this rank holds", ncand, " of it"
             write(*,*) "  (rank partition", Nbatch_s, "to", Nbatch_e, &
                  ", term positions", set_first(j), "to", set_last(j), ")"
             write(*,*) "  lower the batch size, use fewer ranks, or give"
             write(*,*) "  the term more points"
             stop
          end if
          do ib = 1, nbset
             call random_irange( ibuf, ib, ncand )
             itmp2 = cand(ib);  cand(ib) = cand(ibuf);  cand(ibuf) = itmp2
             nbdrawn = nbdrawn + 1
             ind_batch(nbdrawn) = cand(ib)
          end do
       end do
 


       ! How many points this rank actually holds this epoch.  With one
       ! rank, or with the local-SGD rules, it is Num_batch; with a
       ! global step it is this rank's share of it, and the loops below
       ! must use it rather than Num_batch, or they would read entries of
       ! ind_batch that no draw has written.  The NORMALIZATION stays
       ! Num_batch: the objective is the average over the whole batch,
       ! which the ranks hold between them.
       nb_use = nbdrawn

#ifdef _MPI_
       ! The draw above is already stratified over this rank's share of
       ! each Loss_term.  A global permutation of the drawn positions
       ! would send them into other terms and undo that, so none is
       ! applied: `Shuffle` has no effect under MPI, and the input
       ! reader says so.
#endif

       ! The layer index is the FASTEST dimension of these cubes, so a
       ! per-layer slice is a stride-Nlayer walk that still touches every
       ! cache line of the cube; the padded but contiguous full-cube
       ! sweep is faster.  The winning move is fewer sweeps, not smaller
       ! ones: work on arrays that stay zero is skipped outright, which
       ! is bitwise neutral (adding or zeroing zeros).
       ! Zero the geodesic row arrays before any sweep fills them.  The
       ! sums below run over every slot of every point, so a term that
       ! contributes fewer than nrow_pt rows would otherwise leave values
       ! from an earlier epoch in the slots it does not touch.  This is
       ! outside the method branches because NATURAL_GRAD is the method
       ! that reads them.
       if ( allocated(rowval) ) then
          rowval = 0.d0;  avec = 0.d0;  jdotu = 0.d0
       end if

       if ( gd_method=="KALMAN" ) then
          ! The filter presents one scalar observable at a time and moves
          ! the weights itself, so this branch replaces both the gradient
          ! accumulation and the optimizer step.
          if ( .not. lnet_kalman_active() ) then
             call lnet_kalman_init( gd_param(1), gd_param(2), gd_param(3) )
             write(*,'(a,i0,a,f8.1,a)') "### Kalman filter: ", lnet_nweights(), &
                  " weights, covariance ", &
                  dble(lnet_nweights())**2*8.d0/1.d6, " MB"
          end if
          do ib=1,nb_use
             itmp = ind_batch(ib)
             zmat(1,1:ndim(1)) = descriptor_input(ind_train(itmp),1:ndim(1))
             call get_batch2Ntrain_set( ib,ind_batch,ind_train,ib_ntrain )
             ! The filter balances observations through its own noise and
             ! forgetting factor, not through the loss weights: the set
             ! weight of a Loss_term and the HOD lambda select WHICH
             ! observations are presented, but do not scale them.  A run
             ! that needs weighted balancing should use a gradient method
             ! or the residual entry, which takes an explicit noise.
             if ( form_train(ib_ntrain)=="MATH" ) then
                call lnet_kalman_slot( zmat(1,1:ndim(1)), 1, &
                     response_input(ind_train(itmp),1) )
             else if ( form_train(ib_ntrain)=="MATH_HOD" ) then
                do ir=1,NUM_alpha
                   if ( lambda_hod(alpha_deg(ir)) == 0.d0 ) cycle
                   call lnet_kalman_slot( zmat(1,1:ndim(1)), ir, &
                        hod_target_input(ind_train(itmp),ir) )
                end do
             else if ( form_train(ib_ntrain)=="PINN" ) then
                if ( sys_nterm > 0 ) then
                   ! One observable per residual of the system, presented
                   ! in turn: the seed of residual ir is dR_ir/dT, which
                   ! set_sys_seed gives when the residual vector holds a
                   ! one in that slot and zeros elsewhere.
                   syssrc(1:sys_nres) = &
                        sys_src_input(ind_train(itmp),1:sys_nres)
                   do ir = 1, sys_nres
                      ! The filter moves the weights on every update, so
                      ! the fields must be re-evaluated before each one.
                      ! Computing them once outside this loop leaves the
                      ! second and later residuals seeded with values
                      ! from weights that no longer exist, and the filter
                      ! then corrects an innovation it did not measure.
                      if ( kalman_iter <= 1 ) then
                         call lnet_forward_hod_multi( zmat(1,1:ndim(1)), tmat )
                         call calc_sys_residual( tmat, syssrc, rsys )
                         rsel = 0.d0
                         rsel(ir) = 1.d0
                         ! unweighted: the observation row is dR/dw itself
                         call set_sys_seed( tmat, rsel, 1.d0, smat, .false. )
                         ! sys_rnoise(ir) is the observation noise of this
                         ! residual; the default of one leaves the filter
                         ! as it was
                         call lnet_kalman_resid_multi( zmat(1,1:ndim(1)), &
                              smat, rsys(ir), sys_rnoise(ir) )
                      else
                         ! Iterated EKF (Kalman_iter > 1): the same
                         ! observation, relinearized at the running
                         ! iterate.  Residual and seed are rebuilt at
                         ! the current weights before every iteration --
                         ! that re-evaluation is the whole content of
                         ! the method -- and only the last call touches
                         ! the covariance.
                         call lnet_kalman_iekf_begin()
                         do iit = 1, kalman_iter
                            call lnet_forward_hod_multi( &
                                 zmat(1,1:ndim(1)), tmat )
                            call calc_sys_residual( tmat, syssrc, rsys )
                            rsel = 0.d0
                            rsel(ir) = 1.d0
                            call set_sys_seed( tmat, rsel, 1.d0, smat, &
                                 .false. )
                            call lnet_kalman_iekf_iter( zmat(1,1:ndim(1)), &
                                 smat, rsys(ir), sys_rnoise(ir), &
                                 iit == kalman_iter )
                         end do
                      end if
                   end do
                else
                   call lnet_forward_hod( zmat(1,1:ndim(1)), tvec )
                   call calc_pinn_residual( tvec, pinn_src(ind_train(itmp)), rbuf )
                   call set_pinn_seed( tvec, 1.d0, 1.d0, svec )
                   call lnet_kalman_resid( zmat(1,1:ndim(1)), svec, rbuf )
                end if
             end if
             ! the filter has just written the weights
             call lnet_export_weights( weight )
             weight_gen = weight_gen + 1
             call lnet_sync_weights( weight )
          end do
          go to 700
       end if

       nabla(:,:,:)=0.d0
       ! dweight is NOT zeroed here: it carries the previous step (velocity)
       ! for the momentum-family optimizers
       !
       ! jrow holds one metric row per point of the batch and is written
       ! only on the paths the natural gradient takes.  A point whose
       ! path writes no row would otherwise keep the row of whatever
       ! point occupied that slot in an earlier epoch, and the metric
       ! would mix two samples.  Zeroing costs a pass over the buffer, so
       ! it is done only for the rule that reads it.
       if ( gd_method == "NATURAL_GRAD" ) jrow(:,:,:,:) = 0.d0

#ifdef USE_BLAS
       ! Batched fast path for the plain regression: the whole minibatch
       ! becomes three matrix products per layer inside the library,
       ! which is what a framework does with its batch axis.  Applies
       ! when the batch is homogeneous plain value fitting; every other
       ! configuration uses the per-point loop below.
       ! net_grad_batch fits one scalar output; a multi-output network
       ! must take the per-point path, which handles the components.
       if ( ( Ntot_train_set==1 ) .and. ( trim(form_train(1))=="MATH" ) .and. &
            ( ndim(Nlayer)==1 ) .and. &
            ( trim(gd_method)/="NATURAL_GRAD" ) ) then
          do ib=1,nb_use
             itmp = ind_batch(ib)
             Xbat(1:ndim(1),ib) = descriptor_input(ind_train(itmp),1:ndim(1))
             Ybat(ib) = response_input(ind_train(itmp),1)
          end do
          call lnet_batch_grad( Xbat(1:ndim(1),1:NUM_BATCH), &
               Ybat(1:NUM_BATCH), gd_ratio(1), nabla )
       else
#endif
       do ib=1,nb_use
          !################  Gradient decent for "ib"-batch  ####################!
          itmp = ind_batch(ib)
          zmat(1,1:ndim(1)) = descriptor_input(ind_train(itmp),1:ndim(1))
          call get_batch2Ntrain_set( ib,ind_batch,ind_train,ib_ntrain )

          if ( form_train(ib_ntrain)=="MATH" ) then
             ! ---- value fitting ----
             if ( ndim(Nlayer) > 1 ) then
                ! L = (w/2) sum_i ( u_i(x) - y_i )^2 over the components.
                ! The scalar call below fits the first one only, which
                ! for a system leaves the rest unconstrained.
                call lnet_value_grad_multi( zmat(1,1:ndim(1)), &
                     response_input(ind_train(itmp),1:ndim(Nlayer)), &
                     gd_ratio(ib_ntrain), nabla )
             else
                ! L = (w/2) ( N(x) - y )^2
                call lnet_value_grad( zmat(1,1:ndim(1)), &
                     response_input(ind_train(itmp),1), &
                     gd_ratio(ib_ntrain), nabla )
             end if
             if ( gd_method=="NATURAL_GRAD" ) then
                if ( ndim(Nlayer) > 1 ) then
                   ! The row of a supervised point is dL_n/dw with
                   ! L_n = (1/2) sum_i w_i ( u_i - y_i )^2.  Taking it
                   ! from the first component alone, as the scalar
                   ! routine does, leaves the other components out of the
                   ! metric: the rows then span fewer directions than
                   ! there are weights and the metric is singular, which
                   ! is what the solver reports.
                   ! One row per component: for
                   !   L_n = (1/2) sum_i w_i ( u_i - y_i )^2
                   ! the residual vector is r_i = sqrt(w_i)(u_i - y_i)
                   ! and its Jacobian rows are sqrt(w_i) du_i/dw, which
                   ! a unit seed on component i produces.
                   call lnet_forward_hod_multi( zmat(1,1:ndim(1)), tmat )
                   do ir = 1, ndim(Nlayer)
                      smat = 0.d0
                      smat(ir,1) = sqrt( gd_ratio(ib_ntrain)*sys_wcomp(ir) )
                      call lnet_seed_row_multi( smat, nabla_save )
                      irow = (ib-1)*nrow_pt + ir
                      jrow(irow,:,:,:) = nabla_save
                      if ( NGD_geo ) then
                         if ( .not. allocated(rowval) ) &
                              allocate( rowval(NUM_batch*nrow_pt), &
                                        avec(NUM_batch*nrow_pt), &
                                        jdotu(NUM_batch*nrow_pt) )
                         rowval(irow) = sqrt( gd_ratio(ib_ntrain)*sys_wcomp(ir) ) &
                              *( tmat(ir,1) - response_input(ind_train(itmp),ir) )
                      end if
                   end do
                else
                   ! row j_n = sqrt(w) du(x_n)/dw, so that the metric
                   ! belongs to the same weighted objective as the
                   ! gradient above
                   call lnet_value_row( nabla_save )
                   nabla_save = sqrt( gd_ratio(ib_ntrain) )*nabla_save
                   if ( NGD_geo ) then
                      ! the observable of this row, for the geodesic
                      ! correction: phi_n = u(x_n) - y_n
                      call lnet_forward_hod( zmat(1,1:ndim(1)), tvec )
                      if ( .not. allocated(rowval) ) &
                           allocate( rowval(NUM_batch*nrow_pt), &
                                     avec(NUM_batch*nrow_pt), &
                                     jdotu(NUM_batch*nrow_pt) )
                      rowval((ib-1)*nrow_pt+1) = sqrt( gd_ratio(ib_ntrain) ) &
                           *( tvec(1) - response_input(ind_train(itmp),1) )
                   end if
                   jrow((ib-1)*nrow_pt+1,:,:,:) = nabla_save
                end if
             end if

          else if ( form_train(ib_ntrain)=="MATH_HOD" ) then
             ! ---- high-order fitting: unified loss
             !   L = (w/2) sum_alpha lambda_{|alpha|} ( T^alpha - y_alpha )^2
             ! Only the seed is built here; the sweep is the library's.
             call lnet_forward_hod( zmat(1,1:ndim(1)), tvec )
             do ir=1,NUM_alpha
                svec(ir) = gd_ratio(ib_ntrain)* &
                     lambda_hod(alpha_deg(ir))* &
                     ( tvec(ir) - hod_target_input(ind_train(itmp),ir) )
             end do
             if ( gd_method=="NATURAL_GRAD" ) then
                ! Row = the per-point loss gradient: the empirical Fisher,
                ! not the Gauss-Newton metric the other paths build.  The
                ! set weight and the HOD lambda are already inside svec,
                ! so the row scales as w rather than sqrt(w); that is what
                ! this metric is, and it is the one rank-one row per point
                ! whose limits Section 5 of the paper discusses.
                call lnet_seed_row( svec, nabla_save )
                jrow((ib-1)*nrow_pt+1,:,:,:) = nabla_save
                nabla = nabla + nabla_save
             else
                call lnet_seed_grad( svec, nabla )
             end if

          else if ( form_train(ib_ntrain)=="PINN" ) then
             if ( sys_nterm > 0 ) then
                ! ---- a system of residuals over several components:
                !   L = (w/2) sum_r R_r(x)^2
                ! The seed is a matrix, one row per component, because a
                ! cross term contributes both to the derivative slot of
                ! the component it differentiates and to the value slot
                ! of the component multiplying it.
                call lnet_forward_hod_multi( zmat(1,1:ndim(1)), tmat )
                syssrc(1:sys_nres) = &
                     sys_src_input(ind_train(itmp),1:sys_nres)
                call calc_sys_residual( tmat, syssrc, rsys )
                call set_sys_seed( tmat, rsys, gd_ratio(ib_ntrain), smat )
                call lnet_seed_grad_multi( smat, nabla )
                if ( gd_method=="NATURAL_GRAD" ) then
                   ! One Gauss-Newton row per residual.  For
                   !   L_n = (1/2) sum_r w_r R_r(x_n)^2
                   ! the residual vector is sqrt(w_r) R_r and its
                   ! Jacobian rows are sqrt(w_r) dR_r/dw, which
                   ! set_sys_seed produces from the selector rsel (a one
                   ! in slot r) with the loss weights switched off and
                   ! the square root applied here.  The earlier single
                   ! row per point was d(sum_r w_r R_r^2/2)/dw, whose
                   ! outer product is the empirical Fisher: rank one per
                   ! point, and blind to how the residuals trade against
                   ! each other -- which on a coupled system is the
                   ! whole difficulty.
                   do ir = 1, sys_nres
                      rsel = 0.d0
                      rsel(ir) = sqrt( gd_ratio(ib_ntrain)*sys_wres(ir) )
                      call set_sys_seed( tmat, rsel, 1.d0, smat, .false. )
                      call lnet_seed_row_multi( smat, nabla_save )
                      irow = (ib-1)*nrow_pt + ir
                      jrow(irow,:,:,:) = nabla_save
                      if ( NGD_geo ) then
                         if ( .not. allocated(rowval) ) &
                              allocate( rowval(NUM_batch*nrow_pt), &
                                        avec(NUM_batch*nrow_pt), &
                                        jdotu(NUM_batch*nrow_pt) )
                         rowval(irow) = &
                              sqrt( gd_ratio(ib_ntrain)*sys_wres(ir) )*rsys(ir)
                      end if
                   end do
                end if
             else
                ! ---- collocation: L = (w/2) R(x)^2, with
                !   dL/dT_alpha = w * R * dF/dT_alpha built by set_pinn_seed
                call lnet_forward_hod( zmat(1,1:ndim(1)), tvec )
                call calc_pinn_residual( tvec, pinn_src(ind_train(itmp)), rbuf )
                call set_pinn_seed( tvec, rbuf, gd_ratio(ib_ntrain), svec )
                call lnet_seed_grad( svec, nabla )
                if ( gd_method=="NATURAL_GRAD" ) then
                   ! row j_n = dR(x_n)/dw  (Gauss-Newton metric)
                   call set_pinn_seed( tvec, 1.d0, 1.d0, svec )
                   call lnet_seed_row( svec, nabla_save )
                   nabla_save = sqrt( gd_ratio(ib_ntrain) )*nabla_save
                   jrow((ib-1)*nrow_pt+1,:,:,:) = nabla_save
                   if ( NGD_geo ) then
                      if ( .not. allocated(rowval) ) &
                           allocate( rowval(NUM_batch*nrow_pt), &
                                     avec(NUM_batch*nrow_pt), &
                                     jdotu(NUM_batch*nrow_pt) )
                      rowval((ib-1)*nrow_pt+1) = &
                           sqrt( gd_ratio(ib_ntrain) )*rbuf
                   end if
                end if
             end if
          end if
       end do !ib=1,NUM_BATCH->ib_s,ib_e
#ifdef USE_BLAS
       end if
#endif


!****** optimizer step (all loss forms accumulate into the same nabla) ******!
       ! L-BFGS replaces the update rule rather than parameterising it:
       ! the direction comes from the stored curvature pairs and the step
       ! length from a line search, so there is no learning rate to pass.
       ! The gradient accumulated above is over the whole training set
       ! when Num_batch covers it, which is what the pairs require.
       if ( gd_method == "LBFGS" ) then
          call lbfgs_iteration( nabla, lbfgs_cost, lbfgs_ok )
          if ( .not. lbfgs_ok ) then
             write(*,'(a,i0,a)') "### L-BFGS: no step accepted at epoch ", &
                  istart_step, "; the memory was reset"
          end if
       else if ( gd_method == "LM" ) then
          ! Levenberg-Marquardt: like L-BFGS it replaces the update rule,
          ! moves the weights itself, and measures its own objective.
          call lm_iteration( lbfgs_cost, lbfgs_ok )
          if ( .not. lbfgs_ok ) then
             write(*,'(a,i0,a)') "### LM: no damping accepted at epoch ", &
                  istart_step, "; the weights were not moved"
          end if
       else if ( gd_method == "NATURAL_GRAD" .and. NGD_geo ) then
          ! Geodesic acceleration (Transtrum & Sethna): the plain step
          ! solves (G+cI) delta1 = -g; the correction solves, against
          ! the SAME metric, (G+cI) delta2 = -(1/N) J^T a with
          ! a_n the second directional derivative of the observable
          ! phi_n along delta1, measured by one finite difference:
          !   a_n = (2/h^2) ( phi_n(w+h*delta1) - phi_n - h (J delta1)_n ).
          ! The step is delta1 + delta2/2 when |delta2| <= 2 alpha
          ! |delta1| and falls back to delta1 otherwise, which is the
          ! standard acceptance rule.  Scalar observables only: the
          ! multi-component rows are per-point loss gradients, whose
          ! phi is not a residual, and pretending otherwise would
          ! accelerate along a curvature that was never measured.
          ! Every metric row now carries its own observable in rowval
          ! (one per residual for a system, one per component for a
          ! multi-component fit, one per point for the scalar cases),
          ! so the correction applies to all of them uniformly and the
          ! scalar-only restriction is gone.
          call ngd_apply_inv( nabla, jrow(1:nb_use*nrow_pt,:,:,:), u1, .true. )
          ! The correction must be measured along the step actually
          ! taken, eta*delta1, not along the unscaled direction: the
          ! second directional derivative scales quadratically, so
          ! measuring on delta1 and applying at eta*delta1 misstates
          ! the correction by 1/eta.  Folding eta into u1 here makes
          ! every line below exact for any eta.
          etag = ngd_eta_now()
          u1 = etag*u1
          do ib=1,nb_use*nrow_pt
             jdotu(ib) = sum( jrow(ib,:,:,:)*u1(:,:,:) )
          end do
          hga = ngd_geo_h
          wsave = weight
          weight = weight - hga*u1          ! w + h*delta1, delta1 = -u1
          weight_gen = weight_gen + 1
          call lnet_sync_weights( weight )
          do ib=1,nb_use
             itmp = ind_batch(ib)
             zmat(1,1:ndim(1)) = descriptor_input(ind_train(itmp),1:ndim(1))
             call get_batch2Ntrain_set( ib,ind_batch,ind_train,ib_ntrain )
             ! The perturbed observable must be the SAME function of the
             ! weights that built the row and rowval, set weight and
             ! component weight included.  Leaving sqrt(gd_ratio) out of
             ! phih while rowval carries it makes the finite difference a
             ! difference of two different functions, and the curvature
             ! term picks up an O(1)/h^2 artefact whenever the set weight
             ! is not one.
             wsq = sqrt( gd_ratio(ib_ntrain) )
             if ( form_train(ib_ntrain)=="PINN" .and. sys_nterm > 0 ) then
                call lnet_forward_hod_multi( zmat(1,1:ndim(1)), tmat )
                syssrc(1:sys_nres) = &
                     sys_src_input(ind_train(itmp),1:sys_nres)
                call calc_sys_residual( tmat, syssrc, rsys )
                do ir = 1, sys_nres
                   irow = (ib-1)*nrow_pt + ir
                   phih = wsq*sqrt( sys_wres(ir) )*rsys(ir)
                   avec(irow) = 2.d0/hga**2 &
                        *( phih - rowval(irow) + hga*jdotu(irow) )
                end do
             else if ( form_train(ib_ntrain)=="PINN" ) then
                call lnet_forward_hod( zmat(1,1:ndim(1)), tvec )
                call calc_pinn_residual( tvec, pinn_src(ind_train(itmp)), phih )
                phih = wsq*phih
                irow = (ib-1)*nrow_pt + 1
                avec(irow) = 2.d0/hga**2 &
                     *( phih - rowval(irow) + hga*jdotu(irow) )
             else if ( ndim(Nlayer) > 1 ) then
                call lnet_forward_hod_multi( zmat(1,1:ndim(1)), tmat )
                do ir = 1, ndim(Nlayer)
                   irow = (ib-1)*nrow_pt + ir
                   phih = wsq*sqrt( sys_wcomp(ir) ) &
                        *( tmat(ir,1) - response_input(ind_train(itmp),ir) )
                   avec(irow) = 2.d0/hga**2 &
                        *( phih - rowval(irow) + hga*jdotu(irow) )
                end do
             else
                call lnet_forward_hod( zmat(1,1:ndim(1)), tvec )
                phih = wsq*( tvec(1) - response_input(ind_train(itmp),1) )
                irow = (ib-1)*nrow_pt + 1
                avec(irow) = 2.d0/hga**2 &
                     *( phih - rowval(irow) + hga*jdotu(irow) )
             end if
          end do
          weight = wsave
          weight_gen = weight_gen + 1
          call lnet_sync_weights( weight )
          nabla2 = 0.d0
          do ib=1,nb_use*nrow_pt
             nabla2(:,:,:) = nabla2(:,:,:) + jrow(ib,:,:,:)*avec(ib)
          end do
          call ngd_apply_inv( nabla2, jrow(1:nb_use*nrow_pt,:,:,:), u2, .false. )
          n1sq = sum( u1*u1 );  n2sq = 0.25d0*sum( u2*u2 )
          if ( n2sq <= ngd_geo_alpha**2 * n1sq ) then
             dweight = -( u1 + 0.5d0*u2 )
          else
             dweight = -u1                  ! correction rejected
          end if
       else
       ! The natural gradient receives the rows this epoch filled; every
       ! other method receives the dummy unchanged (a section of the
       ! dummy with a bound built from nrow_pt would be out of range).
       if ( gd_method == "NATURAL_GRAD" ) then
          call optimization_driver( nabla, jrow(1:nb_use*nrow_pt,:,:,:), &
               gd_param_r, gd_param_u, gd_param_v, gd_param_m, dweight )
       else
          call optimization_driver( nabla, jrow, &
               gd_param_r, gd_param_u, gd_param_v, gd_param_m, dweight )
       end if
       end if

       ! L-BFGS and LM move the weights themselves, inside their own line
       ! search or damping loop, and the step they accepted is already in
       ! `weight`.  The generic update below must not run for them: on a
       ! restart, or after a change of method, `dweight` still holds the
       ! previous rule's step and would be added to an accepted one.
       if ( gd_method == "LBFGS" .or. gd_method == "LM" ) then
          dweight(:,:,:) = 0.d0
       else if ( (gd_method=="NESTEROV").or.(gd_method=="RMSPROP_NESTEROV") ) then
          weight(:,:,:) = weight_old(:,:,:) + dweight(:,:,:)
       else
          weight(:,:,:) = weight(:,:,:) + dweight(:,:,:)
       end if
       weight_gen = weight_gen + 1

700    continue
       weight_gen = weight_gen + 1

! Average nn_weight.dat
#ifdef _MPI_
       Ndata = Nlayer*ndim_max*(1+ndim_max)
       ! Nothing to average when every rank has taken the same step from
       ! the same state: the reduction would only add rounding.
       if ( mod(loop,Average_cyc_mpi)==0 .and. .not. replicated_step ) then
          call mpi_allreduce(weight(1,1,0),weight_recv(1,1,0),Ndata,mpi_real8,mpi_sum,mpi_comm_world,ierr)
          weight = weight_recv/nprocs
          weight_gen = weight_gen + 1
       end if
#endif


       if (mod(loop,io_cyc)==0) then
#ifdef _MPI_
          if (myrank==0) then
             i_epoch_now = istart_step
             call write_data("weight_log")
             call write_gdlog(dweight,gd_param_r,gd_param_u,gd_param_v,gd_param_m,istart_step)
          end if
#else
          i_epoch_now = istart_step
          call write_data("weight_log")
          call write_gdlog(dweight,gd_param_r,gd_param_u,gd_param_v,gd_param_m,istart_step)
#endif
       end if

!********** get new w_best **********! START
!************************************!
       if ( loop==1 .or. mod(loop,validation_cyc)==0 ) then
          call perform_validation

          Etrain_update = Cost_train(0)/NUM_train
          ! For L-BFGS report the quantity the line search actually
          ! minimized.  Cost_train is a different sum -- unweighted by
          ! the loss-term coefficients -- so reporting it would show a
          ! trajectory that rises while the search is lowering its own
          ! objective at every step.
          if ( gd_method == "LBFGS" .or. gd_method == "LM" ) &
               Etrain_update = lbfgs_cost/dble(max(NUM_train,1))

          ! Automatic balance of the residual weights.  The norm of the
          ! gradient of each residual is measured on a sample of the
          ! collocation points, and the weights are moved towards the
          ! values that equalise them.  Measuring costs one adjoint pass
          ! per residual per sampled point, so it is done every
          ! sys_balance_cyc epochs rather than every one.
          if ( sys_balance .and. sys_nterm > 0 .and. &
               mod( istart_step, sys_balance_cyc ) == 0 ) then
             call balance_residual_weights
          end if

          ! Trust region on the natural-gradient damping.  A step that
          ! lowered the loss says the quadratic model was trusted too
          ! little, so the damping comes down and the next step is
          ! longer; a step that raised it says the opposite.  The factors
          ! are the usual ones, and the bounds keep the rule from running
          ! away in either direction.
          if ( NGD_trust .and. gd_method == "NATURAL_GRAD" ) then
             if ( NGD_prev_cost > 0.d0 ) then
                if ( Etrain_update < NGD_prev_cost ) then
                   NGD_trust_mu = max( NGD_trust_mu/3.d0, NGD_trust_lo )
                else
                   NGD_trust_mu = min( NGD_trust_mu*5.d0, NGD_trust_hi )
                end if
             end if
             NGD_prev_cost = Etrain_update
          end if
          if ( NUM_validation > 0 ) then
             Eval_update = Cost_val(0)/NUM_validation
          else
             ! No validation set: Cost_val/0 is undefined and would make
             ! every patience comparison false, silently stopping the run
             ! after patience_max cycles.  The patience test falls back to
             ! the training cost instead.
             Eval_update = Etrain_update
          end if
          if ( Etrain_update < Etrain_best ) then
             weight_best=weight
             weight_best_gen = weight_best_gen + 1
             epoch_best=istart_step
             Etrain_best = Etrain_update
          end if

          ! patience is the number of CONSECUTIVE validation cycles
          ! without improvement, the usual meaning; an improvement
          ! resets it.  Counting them cumulatively would stop a run
          ! that is still improving, only more slowly than every cycle.
          if ( Eval_update <= Eval_old ) then
             Eval_old = Eval_update
             patience = 0
          else
             patience = patience+1
          end if

#ifdef _MPI_
          if (myrank==0) then
             call write_outfile( loop,patience )
          end if
#else
          call write_outfile( loop,patience )
#endif
       end if
!************************************!
!********** get new w_best **********! END

       istop_local = 0
       if ( (Etrain_update<conv_fit).or.(patience>patience_max) ) istop_local = 1
#ifdef _MPI_
       ! all ranks must agree on stopping, otherwise the next allreduce deadlocks
       call mpi_allreduce(istop_local,istop_global,1,mpi_integer,mpi_max,mpi_comm_world,ierr)
       istop_local = istop_global
#endif
       if ( istop_local == 1 ) then
#ifdef _MPI_
          if (myrank==0) then
             if (patience>patience_max) write(*,*) "patience exceeded"
             write(*,*) "exit epoch#=",istart_step,"err=",Etrain_update,conv_fit
          end if
#else
          if (patience>patience_max) write(*,*) "patience exceeded"
          write(*,*) "exit epoch#=",istart_step,"err=",Etrain_update,conv_fit
#endif
          exit epoch
       end if

       
    enddo epoch !loop
    
#ifdef _MPI_
    if (myrank==0) then
       write(*,*) "best epoch=",epoch_best,Etrain_best
       call write_data("weight_best")
       call write_gdlog(dweight,gd_param_r,gd_param_u,gd_param_v,gd_param_m,istart_step)
    end if
#else
    write(*,*) "best epoch=",epoch_best,Etrain_best
    call write_data("weight_best")
    call write_gdlog(dweight,gd_param_r,gd_param_u,gd_param_v,gd_param_m,istart_step)
#endif
    
    deallocate( ind_batch )

  END SUBROUTINE sgd_minibatch


#ifdef USE_BLAS
  ! One minibatch of the plain MATH fit as three GEMMs per layer.
  ! Point ib of the batch is column ib.  The arithmetic per point is the
  ! same as calc_feedforward + err_prop + grad_descent; only the
  ! summation order inside each contraction changes (dgemm), so results
  ! agree with the reference loop to roundoff.

#endif

  SUBROUTINE get_batch2Ntrain_set( i_batch,index_b,index_t,i_batch2train ) !use label_start,label_end
    implicit none
    integer,intent(IN) :: i_batch,index_b(NUM_BATCH),index_t(NUM_train)
    integer,intent(OUT) :: i_batch2train
    integer :: i0,i1,j
    ! original data: x_in(index_t( index_b(i_batch)) ) )
    i0 = index_b(i_batch)
    i1 = index_t(i0)

    ! label_start(j) <= i1 <= label_end(j)
    ! i_batch2train is intent(OUT) and was left undefined when no range
    ! matched, so a malformed label range produced a silent out of range
    ! index in the caller.  It is now initialised and checked.
    i_batch2train = 0
    do j=1,Ntot_train_set
       if ( (label_start(j)<=i1).and.(i1<=label_end(j)) ) then
          i_batch2train = j
          exit
       end if
    end do
    if ( i_batch2train == 0 ) then
       write(*,*) "get_batch2Ntrain_set: point ",i1," lies outside every training set"
       stop
    end if

  END SUBROUTINE get_batch2Ntrain_set


  !> Rescale the residual weights so that each contributes a gradient of
  !! comparable size.  See sys_balance in global_variables for what the
  !! rule is and why it exists.
  SUBROUTINE balance_residual_weights
    implicit none
    real(8),allocatable :: gnorm(:), tmb(:,:), smb(:,:), rb(:), sb(:)
    real(8),allocatable :: nab(:,:,:), tdum(:)
    real(8) :: gmax
    integer :: ir, n, nsample, l, j, k

    nsample = min( 40, NUM_train )
    if ( nsample < 1 ) return
    allocate( gnorm(sys_nres), rb(sys_nres), sb(sys_nres) )
    allocate( tmb(ndim(Nlayer),NUM_alpha), smb(ndim(Nlayer),NUM_alpha) )
    allocate( nab(Nlayer,ndim_max,0:ndim_max), tdum(NUM_alpha) )
    gnorm = 0.d0

    do ir = 1, sys_nres
       nab = 0.d0
       do n = 1, nsample
          call lnet_forward_hod_multi( &
               descriptor_input(ind_train(n),1:ndim(1)), tmb )
          sb(1:sys_nres) = 0.d0
          if ( sys_use_src ) &
               sb(1:sys_nres) = sys_src_input(ind_train(n),1:sys_nres)
          call calc_sys_residual( tmb, sb, rb )
          ! the gradient of this residual's own term of the loss
          sb = 0.d0;  sb(ir) = rb(ir)
          call set_sys_seed( tmb, sb, 1.d0, smb, .false. )
          call lnet_seed_grad_multi( smb, nab )
       end do
       do l = 2, Nlayer
          do j = 1, ndim(l)
             do k = 0, ndim(l-1)
                gnorm(ir) = gnorm(ir) + nab(l,j,k)**2
             end do
          end do
       end do
       gnorm(ir) = sqrt( gnorm(ir) )
    end do

    gmax = maxval( gnorm(1:sys_nres) )
    if ( gmax > 0.d0 ) then
       do ir = 1, sys_nres
          if ( gnorm(ir) <= 0.d0 ) cycle
          sys_wres(ir) = ( 1.d0 - sys_balance_alpha )*sys_wres(ir) &
               + sys_balance_alpha*gmax/gnorm(ir)
       end do
       write(*,'(a,16es11.3)') "### residual weights rebalanced:", &
            sys_wres(1:sys_nres)
    end if

    deallocate( gnorm, rb, sb, tmb, smb, nab, tdum )
  END SUBROUTINE balance_residual_weights

  !> The weight cube of the trainer, laid out as one vector.
  !!
  !! The cube is (Nlayer, ndim_max, 0:ndim_max) and the weights of layer
  !! l occupy (l, 1:ndim(l), 0:ndim(l-1)); the rest is padding.  L-BFGS
  !! works on the vector of real weights, so the two forms have to be
  !! converted, and the order below is the one kf_update_grad uses, which
  !! keeps the two consistent should they ever meet.
  SUBROUTINE lbfgs_flatten_w( v )
    implicit none
    real(8),intent(OUT) :: v(:)
    integer :: l, j, k, m
    m = 0
    do l = 2, Nlayer
       do j = 1, ndim(l)
          do k = 0, ndim(l-1)
             m = m + 1
             v(m) = weight(l,j,k)
          end do
       end do
    end do
  END SUBROUTINE lbfgs_flatten_w

  !> The inverse of lbfgs_flatten_w, which also pushes the weights into
  !! the library network so that a cost evaluated afterwards is the cost
  !! of what was just written.
  SUBROUTINE lbfgs_unflatten_w( v )
    implicit none
    real(8),intent(IN) :: v(:)
    integer :: l, j, k, m
    m = 0
    do l = 2, Nlayer
       do j = 1, ndim(l)
          do k = 0, ndim(l-1)
             m = m + 1
             weight(l,j,k) = v(m)
          end do
       end do
    end do
    call lnet_sync_weights( weight )
  END SUBROUTINE lbfgs_unflatten_w

  !> The gradient cube as a vector, in the same order.
  SUBROUTINE lbfgs_flatten_grad( nabla, v )
    implicit none
    real(8),intent(IN) :: nabla(Nlayer,ndim_max,0:ndim_max)
    real(8),intent(OUT) :: v(:)
    integer :: l, j, k, m
    m = 0
    do l = 2, Nlayer
       do j = 1, ndim(l)
          do k = 0, ndim(l-1)
             m = m + 1
             v(m) = nabla(l,j,k)
          end do
       end do
    end do
  END SUBROUTINE lbfgs_flatten_grad

  !> The training cost over every point, which is what the line search
  !! tests.  perform_validation recomputes both the training and the
  !! validation costs from the current weights, so it is the honest
  !! evaluation; a cheaper partial sum would make the search test a
  !! different function from the one being minimized.
  REAL(8) FUNCTION lbfgs_full_cost()
    implicit none
    integer :: lb_i
    call perform_validation
    ! The sum over the points, not the mean.  The gradient accumulated in
    ! nabla is a sum, so a mean here would make the Armijo target demand
    ! a decrease NUM_train times larger than the one the gradient
    ! predicts, and no step would ever be accepted: the loss falls, the
    ! test says it has not fallen enough, and the search backtracks to
    ! nothing.  The two must be the same functional.
    ! Cost_train_set(:,0) already carries the Loss_term weight of its
    ! set, applied once in the validation assembly, which is what makes
    ! the history, the patience rule and this search read one and the
    ! same sum.  Applying gd_ratio again here would weight every term
    ! twice -- data + w^2 * residual instead of data + w * residual --
    ! so the terms are summed as they stand.
    lbfgs_full_cost = 0.d0
    do lb_i = 1, Ntot_train_set
       lbfgs_full_cost = lbfgs_full_cost + Cost_train_set(lb_i,0)
    end do
  END FUNCTION lbfgs_full_cost

  !> The loss and its gradient over every training point, computed in one
  !! sweep so that the two are the same functional by construction.
  !!
  !! This is the whole point of the routine.  The minibatch loop of the
  !! trainer draws its points at random, so the gradient it accumulates
  !! belongs to a different sample every epoch, while the cost of the
  !! line search is over all of them.  L-BFGS cannot work across two
  !! functions: the pairs (s,y) assume consecutive gradients are of the
  !! same one, and a line search assumes the direction descends the
  !! function it is testing.  Measuring both here removes the question.
  SUBROUTINE lbfgs_cost_grad( cost, grad )
    implicit none
    real(8),intent(OUT) :: cost
    real(8),intent(OUT) :: grad(:)
    real(8),allocatable :: tvecl(:), svecl(:), nabl(:,:,:)
    real(8),allocatable :: tml(:,:), sml(:,:), rl(:), srcl(:)
    real(8) :: rr, w
    integer :: n, it, ir, l, j, k, m, iset, lb_lo, lb_hi
    real(8) :: cost_v(1)

    allocate( nabl(Nlayer,ndim_max,0:ndim_max) );  nabl = 0.d0
    allocate( tvecl(max(NUM_alpha,1)), svecl(max(NUM_alpha,1)) )
    allocate( tml(max(ndim(Nlayer),1),max(NUM_alpha,1)) )
    allocate( sml(max(ndim(Nlayer),1),max(NUM_alpha,1)) )
    allocate( rl(max(sys_nres,1)), srcl(max(sys_nres,1)) )
    cost = 0.d0

    ! Split across ranks when they are in lockstep (see replicated_step);
    ! the partial cost and gradient are summed after the sweep, which is
    ! exactly the serial sum up to the order of the additions.
    call slice_bounds( NUM_train, lb_lo, lb_hi )
    do n = lb_lo, lb_hi
       it = ind_train(n)
       iset = 0
       do j = 1, Ntot_train_set
          if ( label_start(j) <= it .and. it <= label_end(j) ) then
             iset = j
             exit
          end if
       end do
       if ( iset == 0 ) cycle
       w = gd_ratio( iset )
       if ( w == 0.d0 ) cycle          ! a term switched off is absent
                                        ! from both the cost and the
                                        ! gradient, not from one of them
       if ( form_train(iset) == "PINN" ) then
          if ( sys_nterm > 0 ) then
             call lnet_forward_hod_multi( descriptor_input(it,1:ndim(1)), tml )
             srcl(1:sys_nres) = 0.d0
             if ( sys_use_src ) srcl(1:sys_nres) = sys_src_input(it,1:sys_nres)
             call calc_sys_residual( tml, srcl, rl )
             do ir = 1, sys_nres
                cost = cost + 0.5d0*w*sys_wres(ir)*rl(ir)**2
             end do
             call set_sys_seed( tml, rl, w, sml )
             call lnet_seed_grad_multi( sml, nabl )
          else
             call lnet_forward_hod( descriptor_input(it,1:ndim(1)), tvecl )
             call calc_pinn_residual( tvecl, pinn_src(it), rr )
             cost = cost + 0.5d0*w*rr*rr
             call set_pinn_seed( tvecl, rr, w, svecl )
             call lnet_seed_grad( svecl, nabl )
          end if
       else if ( form_train(iset) == "MATH" ) then
          if ( ndim(Nlayer) > 1 ) then
             call lnet_forward_hod_multi( descriptor_input(it,1:ndim(1)), tml )
             do ir = 1, ndim(Nlayer)
                cost = cost + 0.5d0*w*sys_wcomp(ir) &
                     *( tml(ir,1) - response_input(it,ir) )**2
             end do
             call lnet_value_grad_multi( descriptor_input(it,1:ndim(1)), &
                  response_input(it,1:ndim(Nlayer)), w, nabl )
          else
             call lnet_forward_hod( descriptor_input(it,1:ndim(1)), tvecl )
             cost = cost + 0.5d0*w*( tvecl(1) - response_input(it,1) )**2
             call lnet_value_grad( descriptor_input(it,1:ndim(1)), &
                  response_input(it,1), w, nabl )
          end if
       else if ( form_train(iset) == "MATH_HOD" ) then
          ! derivative targets: the same objective the per-batch path and
          ! the LM path use, so that the line search, the reported cost
          ! and the gradient are one function
          call lnet_forward_hod( descriptor_input(it,1:ndim(1)), tvecl )
          svecl = 0.d0
          do ir = 1, NUM_alpha
             if ( lambda_hod(alpha_deg(ir)) == 0.d0 ) cycle
             cost = cost + 0.5d0*w*lambda_hod(alpha_deg(ir)) &
                  *( tvecl(ir) - hod_target_input(it,ir) )**2
             svecl(ir) = w*lambda_hod(alpha_deg(ir)) &
                  *( tvecl(ir) - hod_target_input(it,ir) )
          end do
          call lnet_seed_grad( svecl, nabl )
       end if
    end do

    ! One reduction for the pair, so that the cost and the gradient stay
    ! the same functional on every rank.
    cost_v(1) = cost
    call sum_over_ranks( cost_v, 1 )
    cost = cost_v(1)
    call sum_over_ranks( nabl, Nlayer*ndim_max*(1+ndim_max) )

    m = 0
    do l = 2, Nlayer
       do j = 1, ndim(l)
          do k = 0, ndim(l-1)
             m = m + 1
             grad(m) = nabl(l,j,k)
          end do
       end do
    end do

    deallocate( nabl, tvecl, svecl, tml, sml, rl, srcl )
  END SUBROUTINE lbfgs_cost_grad

  !> The same loss without the gradient, for the line search.
  SUBROUTINE lbfgs_cost_only( cost )
    implicit none
    real(8),intent(OUT) :: cost
    real(8),allocatable :: gtmp(:)
    allocate( gtmp(NUM_weight) )
    call lbfgs_cost_grad( cost, gtmp )
    deallocate( gtmp )
  END SUBROUTINE lbfgs_cost_only

  !> One L-BFGS iteration over the whole training set.
  !!
  !! Unlike the minibatch rules this is a full-batch method: the pairs
  !! (s,y) only mean something if consecutive gradients are of the same
  !! function, and a resampled minibatch changes the function every step.
  !! That is the trade: each iteration costs a pass over all points, and
  !! buys a direction that carries curvature and a step length that is
  !! searched rather than guessed.
  !!
  !! The line search is a backtracking one on the Armijo condition
  !!
  !!     L(w + t d) <= L(w) + c1 t g.d ,
  !!
  !! which is what removes the step-size problem the first-order rules
  !! all had here: a step that fails to reduce the loss is shortened and
  !! retried, never taken.
  SUBROUTINE lbfgs_iteration( nabla, cost_out, ok )
    implicit none
    real(8),intent(INOUT) :: nabla(Nlayer,ndim_max,0:ndim_max)
    real(8),intent(OUT) :: cost_out
    logical,intent(OUT) :: ok
    real(8),allocatable,save :: sv(:,:), yv(:,:), rhov(:)
    real(8),allocatable,save :: gprev(:), wprev(:)
    integer,save :: npair = 0, newest = 0
    logical,save :: first = .true.
    logical,save :: have_prev = .false.
    real(8),allocatable :: gflat(:), d(:), wflat(:), wtry(:)
    real(8) :: l0, lt, gd, t, sy, dnorm, tsc, lexp
    real(8) :: gdt, tlo, thi
    real(8),allocatable :: gtry(:)
    integer :: nw, l, j, k, m, it

    nw = NUM_weight
    if ( first ) then
       allocate( sv(nw,LBFGS_M), yv(nw,LBFGS_M), rhov(LBFGS_M) )
       allocate( gprev(nw), wprev(nw) )
       sv = 0.d0;  yv = 0.d0;  rhov = 0.d0
       first = .false.
    end if
    allocate( gflat(nw), d(nw), wflat(nw), wtry(nw) )
    if ( LBFGS_wolfe > 0.d0 ) allocate( gtry(nw) )

    call lbfgs_flatten_w( wflat )
    call lbfgs_cost_grad( l0, gflat )

    ! the pair from the previous iteration, accepted only if it carries
    ! positive curvature; on a nonconvex loss it sometimes does not
    ! Whether a previous iterate exists is its own fact, and cannot be
    ! read off npair: that stays zero until a pair has been stored, so
    ! testing it here means the first pair is never formed and the method
    ! silently degenerates to steepest descent.
    if ( have_prev ) then
       sy = dot_product( wflat - wprev, gflat - gprev )
       if ( sy > 1.d-12 ) then
          newest = mod( newest, LBFGS_M ) + 1
          npair = min( npair + 1, LBFGS_M )
          sv(1:nw,newest) = wflat - wprev
          yv(1:nw,newest) = gflat - gprev
          rhov(newest) = 1.d0/sy
       end if
    end if
    have_prev = .true.

    if ( npair == 0 ) then
       d = -gflat                       ! steepest descent to start
    else
       call lbfgs_direction( gflat, d, sv, yv, rhov, npair, newest )
    end if

    gd = dot_product( gflat, d )
    if ( gd >= 0.d0 ) then
       d = -gflat                       ! not a descent direction; reset
       gd = dot_product( gflat, d )
       npair = 0;  newest = 0
    end if

    if ( lbfgs_verbose .or. lbfgs_scan ) &
         write(*,'(a,e16.8,a,e14.6,a,e12.4)') "### search: L0 ", l0, &
         "  g.d ", gd, "  t0 ", t
    wprev = wflat;  gprev = gflat
    ! The first trial step.  A quasi-Newton direction is already scaled,
    ! so t = 1 is the right guess for it and is what makes the method
    ! converge quickly once the memory is filled.  A steepest-descent
    ! direction is not scaled at all: here |g| is of order 1e3, so t = 1
    ! would move the weights by a thousand and the loss to 1e17, and
    ! twenty halvings would not undo it.  Scaling the first step by
    ! 1/|g| is the standard remedy.
    ! Cap the first trial by how far it moves the weights.  A
    ! quasi-Newton direction is scaled and t = 1 suits it, a steepest
    ! descent direction is not scaled at all, and on this problem the
    ! gradient norm is of order a thousand: t = 1 would move the weights
    ! by that much and twenty halvings would not undo it.  Limiting
    ! ||t d|| to LBFGS_dmax covers both cases with one rule.
    t = LBFGS_step0
    dnorm = sqrt( max( dot_product(d,d), 1.d-300 ) )
    if ( t*dnorm > LBFGS_dmax ) t = LBFGS_dmax/dnorm

    ! An exhaustive scan, for diagnosis only: is there any step at all
    ! along this direction that lowers the loss?  If none does, the
    ! direction is not a descent direction however the search is tuned,
    ! and the fault is upstream of the line search.
    if ( lbfgs_scan ) then
       write(*,'(a,e14.6,a,e12.4,a,e12.4)') "### scan: L0 ", l0, &
            "  g.d ", gd, "  |d| ", dnorm
       tsc = 1.d-1
       do it = 1, 12
          wtry = wflat + tsc*d
          call lbfgs_unflatten_w( wtry )
          lt = lbfgs_full_cost()
          write(*,'(a,e11.3,a,e16.8,a,e12.4)') "###   t ", tsc, &
               "  L ", lt, "  L-L0 ", lt - l0
          tsc = tsc*0.1d0
       end do
       ! The slope actually observed, against the one the gradient
       ! predicts.  If these disagree in sign, nabla is not the gradient
       ! of the function the search is testing, and everything downstream
       ! is beside the point.
       wtry = wflat + 1.d-9*d
       call lbfgs_unflatten_w( wtry )
       lt = lbfgs_full_cost()
       write(*,'(a,e14.6,a,e14.6)') "### slope: measured ", &
            ( lt - l0 )/1.d-9, "  predicted ", gd
       call lbfgs_unflatten_w( wflat )
    end if
    ok = .false.
    ! The search.  Armijo alone accepts any sufficiently-decreasing
    ! step, including one far shorter than the model asks for, and a
    ! short step makes s.y small.  Plain L-BFGS tolerates that -- it
    ! reads only one scalar off the newest pair -- but a self-scaled
    ! method reads a curvature factor off EVERY pair, and factors built
    ! from stunted steps are what made SSBFGS lose to L-BFGS here until
    ! the curvature condition was added.  With Lbfgs_wolfe the search
    ! also requires
    !
    !     g(w + t d) . d  >=  c2 * g(w) . d      (c2 in (c1,1))
    !
    ! which is the second Wolfe condition: it rejects steps that stop
    ! while the slope is still steeply negative, and the search grows
    ! t instead of shrinking it in that case.  Each such trial costs a
    ! gradient rather than a cost evaluation.
    tlo = 0.d0;  thi = -1.d0            ! bracket, thi < 0 = not yet set
    do it = 1, LBFGS_maxls
       wtry = wflat + t*d
       call lbfgs_unflatten_w( wtry )
       if ( LBFGS_wolfe > 0.d0 ) then
          call lbfgs_cost_grad( lt, gtry )
          gdt = dot_product( gtry, d )
       else
          call lbfgs_cost_only( lt )
       end if
       if ( lbfgs_verbose ) &
            write(*,'(a,i3,a,e12.4,a,e14.6,a,e14.6)') "###   ls ", it, &
            "  t ", t, "  L ", lt, "  target ", l0 + LBFGS_c1*t*gd
       if ( lt > l0 + LBFGS_c1*t*gd ) then
          ! Armijo violated: the step is too long.
          thi = t
          t = 0.5d0*( tlo + thi )
          cycle
       end if
       if ( LBFGS_wolfe <= 0.d0 ) then
          ok = .true.
          exit
       end if
       if ( gdt >= LBFGS_wolfe*gd ) then
          ok = .true.                    ! both Wolfe conditions hold
          exit
       end if
       ! Armijo holds but the slope is still steep: the step is too
       ! short.  Move the low end up and look farther.
       tlo = t
       if ( thi > 0.d0 ) then
          t = 0.5d0*( tlo + thi )
       else
          t = 2.d0*t
          if ( t*dnorm > 1.d3*LBFGS_dmax ) then
             ok = .true.                 ! no bracket found; keep the step
             exit
          end if
       end if
    end do
    ! Forward-tracking (Lbfgs_expand): a first-trial acceptance says the
    ! quadratic model under-asked, so look farther.  Each doubling costs
    ! one cost evaluation and is kept only while Armijo still holds at
    ! the doubled step; the last accepted (t, L) pair always wins, so
    ! with the keyword at zero the loop body never runs and the pure
    ! backtracking search is reproduced exactly.
    if ( ok .and. it == 1 .and. LBFGS_expand > 0 ) then
       do it = 1, LBFGS_expand
          wtry = wflat + 2.d0*t*d
          call lbfgs_unflatten_w( wtry )
          call lbfgs_cost_only( lexp )
          if ( lbfgs_verbose ) &
               write(*,'(a,i3,a,e12.4,a,e14.6)') "###   fw ", it, &
               "  t ", 2.d0*t, "  L ", lexp
          if ( lexp <= l0 + LBFGS_c1*2.d0*t*gd .and. lexp < lt ) then
             t = 2.d0*t
             lt = lexp
          else
             exit
          end if
       end do
       ! the weights of the accepted step, which the loop above may have
       ! left one doubling past
       wtry = wflat + t*d
       call lbfgs_unflatten_w( wtry )
    end if
    if ( .not. ok ) then
       call lbfgs_unflatten_w( wflat )   ! no acceptable step; stay put
       lt = l0
       npair = 0;  newest = 0            ! the memory is not helping
    end if
    cost_out = lt
    ! Record what the search itself minimizes.  The training history is a
    ! different sum, common to every update rule so that they stay
    ! comparable, and leaving it alone keeps that.  This file is the one
    ! place the L-BFGS objective is visible.
    open(203,file="lbfgs_history.dat",position='append')
    write(203,'(i0,2x,e16.8,2x,e12.4,2x,i3,2x,l2)') istart_step, lt, t, it, ok
    close(203)
    ! Re-evaluate at the weights actually left behind.  If this differs
    ! from the l0 of the next iteration, something between the two has
    ! moved the weights or the loss out from under the search.
    if ( lbfgs_verbose .or. lbfgs_scan ) &
         write(*,'(a,e16.8,a,e16.8)') "### exit: accepted L ", lt, &
         "  recomputed ", lbfgs_full_cost()

    deallocate( gflat, d, wflat, wtry )
  END SUBROUTINE lbfgs_iteration

  !> One Levenberg-Marquardt iteration over the whole training set.
  !!
  !! Every loss form of this trainer is a weighted sum of squares,
  !!   L = (1/2) sum_i w_i r_i(w)^2 ,
  !! with r_i a data misfit, a carried-derivative misfit, or a
  !! collocation residual.  L-BFGS sees only gradients of that sum;
  !! Gauss-Newton sees its structure: with scaled rows
  !! J_i = sqrt(w_i) dr_i/dw and rt_i = sqrt(w_i) r_i, the step solves
  !!
  !!     ( J^T J + mu I ) delta = -J^T rt = -grad L ,
  !!
  !! and mu is adapted from what the step does to the true cost
  !! (down /3 on success, up *5 on failure, the Ngd_trust rule), so the
  !! method moves between Gauss-Newton and small gradient steps by
  !! itself.  Like L-BFGS it is full batch and has no learning rate;
  !! GD_param p1 is the initial mu (1e-3 if unset).  The accept test
  !! recomputes the same composite objective the line search of L-BFGS
  !! uses, so what is minimized, what is tested, and what the history
  !! reports are one object.
  !!
  !! Each row costs one adjoint; the normal matrix is one syrk-shaped
  !! product (bgemm, so BLAS=1 makes it fast) and each mu trial one
  !! dense solve.  Memory is nrows*nw + nw^2 doubles, guarded below.
  SUBROUTINE lm_iteration( cost_out, ok )
    implicit none
    real(8),intent(OUT) :: cost_out
    logical,intent(OUT) :: ok
    real(8),allocatable,save :: Jm(:,:), rt(:)
    real(8),allocatable :: Am(:,:), Amu(:,:), gv(:), db(:)
    real(8),allocatable :: wflat(:), wtry(:)
    real(8),allocatable :: tvecl(:), svecl(:), nabl(:,:,:)
    real(8),allocatable :: tml(:,:), sml(:,:), rl(:), srcl(:)
    real(8),save :: mu = -1.d0
    integer,save :: nrows = -1
    real(8) :: w, s, l0, lt, rr
    logical :: lt_ok
    real(8) :: rsel(max(sys_nres,1))
    integer :: n, it, ir, iset, irow, itry, nw, j, lm_lo, lm_hi
    real(8) :: l0_v(1)

    nw = NUM_weight
    if ( mu < 0.d0 ) then
       mu = 1.d-3
       if ( gd_param(1) > 0.d0 ) mu = gd_param(1)
    end if

    ! ---- count the rows once: one per squared thing in the loss ----
    if ( nrows < 0 ) then
       nrows = 0
       do n = 1, NUM_train
          it = ind_train(n)
          iset = 0
          do j = 1, Ntot_train_set
             if ( label_start(j) <= it .and. it <= label_end(j) ) then
                iset = j;  exit
             end if
          end do
          if ( iset == 0 ) cycle
          if ( gd_ratio(iset) == 0.d0 ) cycle
          if ( form_train(iset) == "PINN" ) then
             if ( sys_nterm > 0 ) then
                nrows = nrows + sys_nres
             else
                nrows = nrows + 1
             end if
          else if ( form_train(iset) == "MATH" ) then
             nrows = nrows + max( ndim(Nlayer), 1 )
          else if ( form_train(iset) == "MATH_HOD" ) then
             do ir = 1, NUM_alpha
                if ( lambda_hod(alpha_deg(ir)) /= 0.d0 ) nrows = nrows + 1
             end do
          end if
       end do
       if ( dble(nrows)*dble(nw)*8.d0 + dble(nw)**2*8.d0 > 3.d9 ) then
          write(*,*) "LM: the Jacobian and the normal matrix need", &
               ( dble(nrows)*dble(nw) + dble(nw)**2 )*8.d0/2.d0**30, &
               " GB (", nrows, " rows x", nw, " weights )"
          write(*,*) "  this direct method is for moderate sizes;", &
               " use LBFGS above them"
          stop
       end if
       allocate( Jm(nrows,nw), rt(nrows) )
       write(*,'(a,i0,a,i0,a)') "### LM: ", nrows, " residual rows, ", &
            nw, " weights"
    end if

    allocate( Am(nw,nw), Amu(nw,nw), gv(nw), db(nw) )
    allocate( wflat(nw), wtry(nw) )
    allocate( nabl(Nlayer,ndim_max,0:ndim_max) )
    allocate( tvecl(max(NUM_alpha,1)), svecl(max(NUM_alpha,1)) )
    allocate( tml(max(ndim(Nlayer),1),max(NUM_alpha,1)) )
    allocate( sml(max(ndim(Nlayer),1),max(NUM_alpha,1)) )
    allocate( rl(max(sys_nres,1)), srcl(max(sys_nres,1)) )

    call lbfgs_flatten_w( wflat )

    ! ---- the scaled rows and residuals ----
    ! Each rank fills its own share of the rows; the normal equations
    ! below are sums over the rows, so summing A, g and the objective
    ! over the ranks gives the same system the whole set would.
    irow = 0
    call slice_bounds( NUM_train, lm_lo, lm_hi )
    do n = lm_lo, lm_hi
       it = ind_train(n)
       iset = 0
       do j = 1, Ntot_train_set
          if ( label_start(j) <= it .and. it <= label_end(j) ) then
             iset = j;  exit
          end if
       end do
       if ( iset == 0 ) cycle
       w = gd_ratio(iset)
       if ( w == 0.d0 ) cycle
       if ( form_train(iset) == "PINN" ) then
          if ( sys_nterm > 0 ) then
             call lnet_forward_hod_multi( descriptor_input(it,1:ndim(1)), tml )
             srcl(1:sys_nres) = 0.d0
             if ( sys_use_src ) srcl(1:sys_nres) = sys_src_input(it,1:sys_nres)
             call calc_sys_residual( tml, srcl, rl )
             do ir = 1, sys_nres
                s = sqrt( w*sys_wres(ir) )
                rsel = 0.d0;  rsel(ir) = s
                ! unweighted seed of s*R_ir: the row is s*dR_ir/dw
                call set_sys_seed( tml, rsel, 1.d0, sml, .false. )
                call lnet_seed_row_multi( sml, nabl )
                irow = irow + 1
                call lbfgs_flatten_grad( nabl, Jm(irow,1:nw) )
                rt(irow) = s*rl(ir)
             end do
          else
             call lnet_forward_hod( descriptor_input(it,1:ndim(1)), tvecl )
             call calc_pinn_residual( tvecl, pinn_src(it), rr )
             s = sqrt( w )
             call set_pinn_seed( tvecl, 1.d0, s, svecl )
             call lnet_seed_row( svecl, nabl )
             irow = irow + 1
             call lbfgs_flatten_grad( nabl, Jm(irow,1:nw) )
             rt(irow) = s*rr
          end if
       else if ( form_train(iset) == "MATH" ) then
          if ( ndim(Nlayer) > 1 ) then
             call lnet_forward_hod_multi( descriptor_input(it,1:ndim(1)), tml )
             do ir = 1, ndim(Nlayer)
                s = sqrt( w*sys_wcomp(ir) )
                sml = 0.d0;  sml(ir,1) = s
                call lnet_seed_row_multi( sml, nabl )
                irow = irow + 1
                call lbfgs_flatten_grad( nabl, Jm(irow,1:nw) )
                rt(irow) = s*( tml(ir,1) - response_input(it,ir) )
             end do
          else
             call lnet_forward_hod( descriptor_input(it,1:ndim(1)), tvecl )
             s = sqrt( w )
             svecl = 0.d0;  svecl(1) = s
             call lnet_seed_row( svecl, nabl )
             irow = irow + 1
             call lbfgs_flatten_grad( nabl, Jm(irow,1:nw) )
             rt(irow) = s*( tvecl(1) - response_input(it,1) )
          end if
       else if ( form_train(iset) == "MATH_HOD" ) then
          call lnet_forward_hod( descriptor_input(it,1:ndim(1)), tvecl )
          do ir = 1, NUM_alpha
             if ( lambda_hod(alpha_deg(ir)) == 0.d0 ) cycle
             s = sqrt( w*lambda_hod(alpha_deg(ir)) )
             svecl = 0.d0;  svecl(ir) = s
             call lnet_seed_row( svecl, nabl )
             irow = irow + 1
             call lbfgs_flatten_grad( nabl, Jm(irow,1:nw) )
             rt(irow) = s*( tvecl(ir) - hod_target_input(it,ir) )
          end do
       end if
    end do

    l0 = 0.5d0*dot_product( rt(1:irow), rt(1:irow) )

    ! ---- normal equations: A = J^T J, g = J^T rt ----
    call bgemm( 'T', 'N', nw, nw, irow, Jm, nrows, Jm, nrows, Am, nw )
    do j = 1, nw
       gv(j) = dot_product( Jm(1:irow,j), rt(1:irow) )
    end do
    ! The three sums over the rows, gathered before the trust loop so
    ! that every rank damps and solves the same system.
    l0_v(1) = l0
    call sum_over_ranks( l0_v, 1 )
    l0 = l0_v(1)
    call sum_over_ranks( Am, nw*nw )
    call sum_over_ranks( gv, nw )

    ! ---- trust loop on mu ----
    ok = .false.
    do itry = 1, 8
       Amu = Am
       do j = 1, nw
          Amu(j,j) = Amu(j,j) + mu
       end do
       db = -gv
       call bposv( nw, Amu, db, lt_ok )
       if ( .not. lt_ok ) then
          ! not positive definite at this mu (roundoff on a nearly
          ! singular J^T J): raise the damping and try again
          mu = min( mu*5.d0, NGD_trust_hi )
          cycle
       end if
       wtry = wflat + db
       call lbfgs_unflatten_w( wtry )
       lt = lbfgs_full_cost()
       if ( lt < l0 ) then
          ok = .true.
          mu = max( mu/3.d0, NGD_trust_lo )
          exit
       end if
       mu = min( mu*5.d0, NGD_trust_hi )
    end do
    if ( .not. ok ) then
       call lbfgs_unflatten_w( wflat )
       lt = l0
    end if
    cost_out = lt

    open(204,file="lm_history.dat",position='append')
    write(204,'(i0,2x,e16.8,2x,e12.4,2x,i3,2x,l2)') istart_step, lt, &
         mu, itry, ok
    close(204)

    deallocate( Am, Amu, gv, db, wflat, wtry )
    deallocate( nabl, tvecl, svecl, tml, sml, rl, srcl )
  END SUBROUTINE lm_iteration

END MODULE sgd_batch_module
