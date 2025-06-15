!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (optimizer_module.f90) is part of DNNF90.
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
! Optimizers for the weight update  dweight = step(gradient, state).
!
! Provenance:
!   - NATURAL_GRAD (damped Fisher metric with eta/mu schedules), ADAGRAD and
!     SIMPLE_GMAXCLIP are ported from the (private) mGGA_subsys version of
!     optimizer_module; the AdaGrad accumulator, which was a non-SAVEd local
!     there, is now proper INOUT state (bug fix).
!   - MOMENTUM / NESTEROV / RMSPROP / RMSPROP_NESTEROV / ADADELTA / ADAM are
!     implemented here following the scaffolding already present in the code
!     base (state arrays gd_param_r/u/v/m, their gd_*.dat logs, and the
!     parameter layout suggested by the legacy input examples).
!
! Conventions:
!   g = nabla_in/NUM_BATCH is the mini-batch mean gradient (all three loss
!   forms MATH / MATH_HOD / PINN accumulate into the same nabla, so every
!   optimizer applies to every loss unchanged).
!   dweight_out is INOUT: it carries the previous step (velocity) for the
!   momentum-family methods and is what gd_dw.dat logs/restores.
!
! GD_param(1:5) layout per method (always give 5 values in input_nn.dat):
!   SIMPLE            eta
!   SIMPLE_SCHEDULE   eta_max eta_min          ( eta = p1/sqrt(step+1)+p2 )
!   SIMPLE_GMAXCLIP   eta                      ( eta = p1/max|g| )
!   MOMENTUM          eta alpha
!   NESTEROV          eta alpha                ( lookahead w+alpha*dw in sgd )
!   ADAGRAD           eta eps
!   RMSPROP           eta rho eps
!   RMSPROP_NESTEROV  eta rho eps alpha        ( lookahead in sgd )
!   ADADELTA          rho eps
!   ADAM              eta beta1 beta2 eps      ( t = istart_step )
!   NATURAL_GRAD      eta mu                   ( + Ngd_* schedule keys )
!
MODULE optimizer_module

  ! Row gathering for the Gram-space route under MPI; serial defaults
  ! make the calls no-ops in the serial build.
  use parallel_module, only: nprocs, replicated_step, sum_over_ranks, &
       gather_rows, total_rows

  use global_variables

  implicit none

  PRIVATE
  PUBLIC :: optimization_driver, lbfgs_direction, read_gdlog, write_gdlog, gauss_solve
  PUBLIC :: ngd_apply_inv, ngd_eta_now

CONTAINS

  SUBROUTINE optimization_driver( nabla_in, jrow_in, r,u,v,m, dweight_out )
    implicit none
    real(8),intent(IN) :: nabla_in(Nlayer,ndim_max,0:ndim_max)
    real(8),intent(IN) :: jrow_in(:,:,:,0:)  ! (NUM_batch,Nlayer,ndim_max,0:ndim_max)
                                             ! per-point rows; used by NATURAL_GRAD only
    real(8),intent(INOUT) :: r(Nlayer,ndim_max,0:ndim_max) ! AdaGrad accumulator
    real(8),intent(INOUT) :: u(Nlayer,ndim_max,0:ndim_max) ! AdaDelta Delta^2 average / momentum velocity
    real(8),intent(INOUT) :: v(Nlayer,ndim_max,0:ndim_max) ! squared-gradient average
    real(8),intent(INOUT) :: m(Nlayer,ndim_max,0:ndim_max) ! Adam first moment
    real(8),intent(INOUT) :: dweight_out(Nlayer,ndim_max,0:ndim_max)
    ! Every rule below is elementwise and is written as a whole-array
    ! statement over the (Nlayer, ndim_max, 0:ndim_max) cube.  That touches
    ! the dead zone outside the live slice (l, 1:ndim(l), 0:ndim(l-1)) as
    ! well, which is harmless because every state cube starts at zero and
    ! the gradient is zero there: each rule maps zeros to zeros, and the
    ! weights of the dead zone are never read.  One rule needs the
    ! regularizer to be positive for that to hold, ADADELTA, whose step is
    ! sqrt(u+eps)/sqrt(v+eps) and would be 0/0 in the dead zone with
    ! eps = 0; the input is rejected in that case.
    real(8) :: eta, bc1, bc2, gmax
    logical,save :: first_time = .true.

    ! Methods that need a global quantity or their own driver first.
    select case ( trim(gd_method) )
    case ( "NATURAL_GRAD" )
       if ( first_time ) then
          open(100,file="learning_rate_eta.log",status='replace')
          open(101,file="diag_part_mu.log",status='replace')
          close(100); close(101)
          first_time = .false.
       end if
       call opt_natural_gradient( nabla_in, jrow_in, dweight_out )
       return
    case ( "SIMPLE_SCHEDULE" )
       eta = gd_param(1)/sqrt(dble(istart_step+1)) + gd_param(2)
    case ( "SIMPLE_GMAXCLIP" )
       ! max|g| is invariant under dividing before or after the max
       ! (division by a positive constant is monotone under rounding)
       gmax = maxval( abs( nabla_in ) )/dble(NUM_BATCH)
       if ( gmax > 0.d0 ) then
          eta = gd_param(1)/gmax
       else
          eta = 0.d0
       end if
    case ( "SIMPLE", "MOMENTUM", "NESTEROV", "ADAGRAD", "RMSPROP", &
           "RMSPROP_NESTEROV", "ADADELTA", "ADAM" )
    case default
       write(*,*) "optimization_driver: unknown GD_method: ",trim(gd_method)
       stop
    end select

    ! The layer index is the fastest dimension of every cube here, so
    ! whole-array statements are contiguous streams; per-layer slices
    ! were measured slower (stride-Nlayer walks touch the same cache
    ! lines without the prefetch).  The mini-batch division is inlined,
    ! which removes the separate g cube and its extra sweep.  The
    ! per-element expression is -c*(nabla/NB) for the plain step and the
    ! documented formula of each method otherwise.
    select case ( trim(gd_method) )

    case ( "SIMPLE" )
       dweight_out = -gd_param(1)*( nabla_in/dble(NUM_BATCH) )

    case ( "SIMPLE_SCHEDULE", "SIMPLE_GMAXCLIP" )
       dweight_out = -eta*( nabla_in/dble(NUM_BATCH) )

    case ( "MOMENTUM", "NESTEROV" )
       ! velocity is carried in dweight_out itself (logged via gd_dw.dat);
       ! for NESTEROV the gradient has been evaluated at the lookahead
       ! point w + alpha*dweight (handled in sgd_minibatch)
       dweight_out = gd_param(2)*dweight_out - gd_param(1)*( nabla_in/dble(NUM_BATCH) )

    case ( "ADAGRAD" )
       r = r + ( nabla_in/dble(NUM_BATCH) )**2
       dweight_out = -gd_param(1)*( nabla_in/dble(NUM_BATCH) )/sqrt( r + gd_param(2) )

    case ( "RMSPROP" )
       v = gd_param(2)*v + (1.d0-gd_param(2))*( nabla_in/dble(NUM_BATCH) )**2
       dweight_out = -gd_param(1)*( nabla_in/dble(NUM_BATCH) )/sqrt( v + gd_param(3) )

    case ( "RMSPROP_NESTEROV" )
       ! Nesterov momentum on the RMSProp-scaled gradient: the velocity
       ! lives in dweight_out, exactly as for MOMENTUM/NESTEROV, and the
       ! gradient has been evaluated at the lookahead point
       ! w + alpha*dweight (handled in sgd_minibatch).  The velocity must
       ! be accumulated, not overwritten, or the momentum term is lost.
       v = gd_param(2)*v + (1.d0-gd_param(2))*( nabla_in/dble(NUM_BATCH) )**2
       dweight_out = gd_param(4)*dweight_out &
            - gd_param(1)*( nabla_in/dble(NUM_BATCH) )/sqrt( v + gd_param(3) )

    case ( "ADADELTA" )
       v = gd_param(1)*v + (1.d0-gd_param(1))*( nabla_in/dble(NUM_BATCH) )**2
       dweight_out = -sqrt( u + gd_param(2) )/sqrt( v + gd_param(2) ) &
                     *( nabla_in/dble(NUM_BATCH) )
       u = gd_param(1)*u + (1.d0-gd_param(1))*dweight_out**2

    case ( "ADAM" )
       m = gd_param(2)*m + (1.d0-gd_param(2))*( nabla_in/dble(NUM_BATCH) )
       v = gd_param(3)*v + (1.d0-gd_param(3))*( nabla_in/dble(NUM_BATCH) )**2
       bc1 = 1.d0 - gd_param(2)**max(iadam_step,1)
       bc2 = 1.d0 - gd_param(3)**max(iadam_step,1)
       dweight_out = -gd_param(1)*( m/bc1 )/( sqrt( v/bc2 ) + gd_param(4) )

    end select

  END SUBROUTINE optimization_driver

  !------------------------------------------------------------------
  ! Damped natural gradient (ported from the mGGA_subsys version):
  !   G = (1/Nb) [ sum_n j_n j_n^T + mu * tr(sum_n j_n j_n^T) * I ]
  !   dweight = -eta(t) * G^{-1} (nabla/Nb)
  ! where j_n are the per-point rows supplied by the caller
  ! (MATH: du/dw; PINN: dR/dw, i.e. a Gauss-Newton metric on the residual;
  !  MATH_HOD: per-point loss gradient, i.e. an empirical Fisher metric).
  ! eta(t), mu(t): SIMPLE p/(1+a t)^b, EXP p*exp(-a t), or constant; with
  ! lower bounds Ngd_eta_bound / Ngd_mu_bound.  The linear system is solved
  ! directly (Gaussian elimination with partial pivoting; no LAPACK needed).
  SUBROUTINE opt_natural_gradient( nabla_in, jrow_in, dweight_out )
    implicit none
    real(8),intent(IN) :: nabla_in(Nlayer,ndim_max,0:ndim_max)
    real(8),intent(IN) :: jrow_in(:,:,:,0:)
    real(8),intent(INOUT) :: dweight_out(Nlayer,ndim_max,0:ndim_max)
    real(8) :: u(Nlayer,ndim_max,0:ndim_max), learning_rate
    call ngd_apply_inv( nabla_in, jrow_in, u, .true. )
    learning_rate = ngd_eta_now()
    dweight_out = -learning_rate*u
  END SUBROUTINE opt_natural_gradient

  !> The learning-rate schedule of the natural gradient, shared by the
  !! plain step and the geodesic-accelerated one.
  REAL(8) FUNCTION ngd_eta_now()
    implicit none
    real(8) :: rtmp
    if ( NGD_schedule_eta=="SIMPLE" ) then
       rtmp = gd_param(1)/dble( 1 + ngd_param_eta(1)*(istart_step-1) )**ngd_param_eta(2)
       ngd_eta_now = max( rtmp, ngd_eta_bound )
    else if ( NGD_schedule_eta=="EXP" ) then
       rtmp = gd_param(1)*exp( -ngd_param_eta(1)*(istart_step-1) )
       ngd_eta_now = max( rtmp, ngd_eta_bound )
    else if ( NGD_schedule_eta=="NONE" ) then
       ngd_eta_now = gd_param(1)
    else
       write(*,*) "opt_natural_gradient: unknown NGD_schedule_eta: ", &
            trim(NGD_schedule_eta), "  (options: NONE, SIMPLE, EXP)"
       stop
    end if
    open(100,file="learning_rate_eta.log",position='append')
    write(100,*) istart_step,ngd_eta_now
    close(100)
  END FUNCTION ngd_eta_now

  !> u = (G + cI)^{-1} (b/N) for an arbitrary cube b, over the rows of
  !! this batch.  The damping schedule and its log are emitted only when
  !! lead is true, so a second solve against the same metric (the
  !! geodesic correction) reuses the same c silently.
  SUBROUTINE ngd_apply_inv( nabla_in, jrow_in, u_out, lead )
    implicit none
    real(8),intent(IN) :: nabla_in(Nlayer,ndim_max,0:ndim_max)
    !> The metric rows.  Their number is NOT the batch size in general:
    !! a system of residuals contributes one Gauss-Newton row per
    !! residual per point, and a multi-component fit one per component,
    !! so the count is read from the array rather than assumed.  The
    !! trainer zero-fills any unused slots, and a zero row contributes
    !! nothing to J^T J or to K beyond the damping already on the
    !! diagonal, so a partially filled buffer is harmless.
    real(8),intent(IN) :: jrow_in(:,:,:,0:)
    real(8),intent(OUT) :: u_out(Nlayer,ndim_max,0:ndim_max)
    logical,intent(IN) :: lead
    integer :: nrow
    ! Heap, not stack: Gm is quadratic in the weight count, which is the
    ! same stack-overflow class removed from every kernel; and the dense
    ! metric bounds the method to small networks, so an unguarded attempt
    ! on a wide net must fail with a message, not an allocator abort.
    real(8),allocatable,save :: Gm(:,:), jr(:,:), b1(:)
    real(8),allocatable,save :: Kd(:,:), jv(:), yv(:)
    !> The Gram route contracts pairs of rows, jr(i,:) . jr(ib,:), which
    !! walks the first index of jr and therefore touches one cache line
    !! per element.  The rows are kept a second time, transposed, so that
    !! those contractions run along contiguous memory: measured on the
    !! five-field system (2741 weights, 600 rows) this is the difference
    !! between 7.5 s and 0.4 s per epoch.  The primal route contracts
    !! along the first index instead and keeps using jr.
    real(8),allocatable,save :: jrT(:,:)
    real(8) :: trace, diag_part, rtmp
    integer :: i, j, l, ib, itmp
    integer :: nrow_all, nb_all
    real(8) :: trace_v(1)

    ! The dual (Gram-space) route never forms the nw x nw metric, so the
    ! dense-size guard applies to the primal route only.
    if ( .not. NGD_dual .and. dble(NUM_weight)**2*8.d0 > 4.d9 ) then
       write(*,*) "opt_natural_gradient: the dense metric needs", &
            dble(NUM_weight)**2*8.d0/2.d0**30, "GB for", NUM_weight, "weights"
       write(*,*) "  NATURAL_GRAD is for small networks; use Ngd_dual (the", &
            " Gram-space solve) or ADAM at this size"
       stop
    end if
    nrow = size( jrow_in, 1 )
    if ( .not. allocated(jr) ) then
       allocate( jr(nrow,NUM_weight), b1(NUM_weight) )
       if ( NGD_dual ) then
          ! Under MPI every rank draws its own share of the batch, so the
          ! metric of the step is built from all of the ranks' rows: the
          ! transposed rows are gathered and the Gram matrix is the
          ! global one.  nrow is the same on every rank, because the draw
          ! takes the same count per Loss_term.
          nrow_all = total_rows( nrow )
          allocate( Kd(nrow_all,nrow_all), jv(nrow_all), yv(nrow_all), &
               jrT(NUM_weight,nrow_all) )
       else
          allocate( Gm(NUM_weight,NUM_weight) )
       end if
    end if
    nrow_all = total_rows( nrow )
    nb_all = NUM_batch
    if ( size(jr,1) /= nrow ) then
       write(*,*) "ngd_apply_inv: the row count changed from", size(jr,1), &
            " to", nrow, " between calls"
       stop
    end if

    ! flatten (canonical order l=Nlayer..2, j, i=0..)
    itmp=1
    do l=Nlayer,2,-1
       do j=1,ndim(l)
          do i=0,ndim(l-1)
             jr(1:nrow,itmp) = jrow_in(1:nrow,l,j,i)
             b1(itmp) = nabla_in(l,j,i)
             itmp = itmp+1
          end do
       end do
    end do

    ! One pass over the rows, so that every contraction below is
    ! contiguous; see the declaration of jrT.  The transposed layout is
    ! also what makes the gather a plain concatenation: rank r's block
    ! is jrT(:, r*nrow+1 : (r+1)*nrow), which is contiguous memory.
    if ( NGD_dual ) then
       jrT(1:NUM_weight,1:nrow) = transpose( jr(1:nrow,1:NUM_weight) )
       call gather_rows( jrT, NUM_weight, nrow )
       ! b is a sum over the batch, so it is summed over the ranks too.
       call sum_over_ranks( b1, NUM_weight )
    end if

    if ( .not. NGD_dual ) then
       Gm = 0.d0
       do ib=1,nrow
          do j=1,NUM_weight
             do i=1,NUM_weight
                Gm(i,j) = Gm(i,j) + jr(ib,i)*jr(ib,j)
             end do
          end do
       end do
       call sum_over_ranks( Gm, NUM_weight*NUM_weight )
       call sum_over_ranks( b1, NUM_weight )
    end if

    ! The two SIMPLE schedules raise 1 + a*(step-1) to a real power, which
    ! is a domain error if the base turns negative.  The step counter is
    ! one based, so the base starts at one and grows for positive a; this
    ! asserts that rather than trusting it.
    if ( istart_step < 1 ) then
       write(*,*) "opt_natural_gradient: the step counter is", istart_step, &
            " but the schedules assume it starts at one"
       stop
    end if
    if ( 1.d0 + ngd_param_mu(1)*dble(istart_step-1) <= 0.d0 .or. &
         1.d0 + ngd_param_eta(1)*dble(istart_step-1) <= 0.d0 ) then
       write(*,*) "opt_natural_gradient: a schedule base has gone", &
            " nonpositive; Ngd_schedule_* wants a >= 0"
       stop
    end if

    ! damping schedule for mu
    if ( NGD_schedule_mu=="SIMPLE" ) then
       rtmp = gd_param(2)/dble( 1 + ngd_param_mu(1)*(istart_step-1) )**ngd_param_mu(2)
       diag_part = max( rtmp, ngd_mu_bound )
    else if ( NGD_schedule_mu=="EXP" ) then
       rtmp = gd_param(2)*exp( -ngd_param_mu(1)*(istart_step-1) )
       diag_part = max( rtmp, ngd_mu_bound )
    else if ( NGD_schedule_mu=="NONE" ) then
       diag_part = gd_param(2)
    else
       ! an unknown name must not silently become a constant schedule
       write(*,*) "opt_natural_gradient: unknown NGD_schedule_mu: ", &
            trim(NGD_schedule_mu), "  (options: NONE, SIMPLE, EXP)"
       stop
    end if
    if ( lead ) then
       open(101,file="diag_part_mu.log",position='append')
       write(101,*) istart_step,diag_part
       close(101)
    end if

    ! Damping.  Two forms, because the natural choice depends on how
    ! large the network is.
    !
    !   TRACE   G + mu tr(G) I     the default and the historical form
    !   ABS     G + mu I           an absolute floor on the eigenvalues
    !
    ! The trace form is relative, which keeps mu meaningful when the
    ! scale of the residual changes but not when the number of weights
    ! does: tr(G) grows with the weight count, so the same mu damps a
    ! network of two thousand weights far more heavily than one of forty.
    ! On a coupled system that difference is the one that matters, and
    ! the absolute form is the one to use there.
    ! The batch average comes first, so that the damping is added to the
    ! metric the solver actually sees.  Adding it before the division
    ! scales it by 1/N_b as well, which leaves the absolute form meaning
    ! something different for every batch size.
    ! Both the metric and the gradient are normalized by the number of
    ! POINTS, not the number of rows.  The objective is
    !   L = (1/2N_b) sum_n sum_r rho_{nr}^2 ,
    ! whose Gauss-Newton matrix is G = J^T J / N_b: a point that carries
    ! several residual rows contributes all of them to one term of the
    ! sum.  Dividing G by the row count instead would shrink the metric
    ! by the number of residuals while leaving the gradient alone, so an
    ! undamped step would grow with the number of residuals and the
    ! absolute damping would mean something different on every system.
    if ( .not. NGD_dual ) Gm = Gm/dble(NUM_batch)
    b1 = b1/dble(NUM_batch)

    ! tr(G) = (1/N_b) sum_i |j_i|^2, from the rows in either route.
    trace = 0.d0
    if ( NGD_dual ) then
       do ib=1,nrow_all
          trace = trace + dot_product( jrT(1:NUM_weight,ib), &
                                       jrT(1:NUM_weight,ib) )
       end do
    else
       do ib=1,nrow
          trace = trace + dot_product( jr(ib,1:NUM_weight), jr(ib,1:NUM_weight) )
       end do
       trace_v(1) = trace
       call sum_over_ranks( trace_v, 1 )
       trace = trace_v(1)
    end if
    trace = trace/dble(nb_all)
    if ( NGD_trust ) diag_part = NGD_trust_mu
    if ( .not.( NGD_damping == "ABS" .or. NGD_trust ) ) &
         diag_part = diag_part*trace     ! the TRACE form is still c*I

    if ( NGD_dual ) then
       ! The identical step through the Gram matrix.  With
       ! G = (1/N) J^T J and any scalar damping c, the push-through
       ! identity gives, for ANY right-hand side b,
       !
       !   (G + cI)^{-1} b = (1/c) [ b - (1/N) J^T K^{-1} (J b) ],
       !   K = cI + (1/N) J J^T   (N_batch x N_batch),
       !
       ! so the nw x nw solve becomes an N_b x N_b one plus two passes
       ! over the rows.  The answer is the primal one to roundoff --
       ! (bench/opt_ngd runs both routes for comparison) -- and the
       ! cost falls from O(N nw^2 + nw^3) to O(N^2 nw + N^3), which is
       ! what makes the natural gradient affordable at coupled-system
       ! widths.  The rows are whatever the trainer built (Gauss-Newton
       ! or empirical Fisher); nothing here depends on which.
       ! K is symmetric, so only the lower triangle is contracted and the
       ! upper one mirrored.
       do ib=1,nrow_all
          do i=ib,nrow_all
             ! same normalization as the primal metric: N_b, not nrow
             Kd(i,ib) = dot_product( jrT(1:NUM_weight,i), &
                                     jrT(1:NUM_weight,ib) )/dble(nb_all)
             Kd(ib,i) = Kd(i,ib)
          end do
          Kd(ib,ib) = Kd(ib,ib) + diag_part
       end do
       do ib=1,nrow_all
          jv(ib) = dot_product( jrT(1:NUM_weight,ib), b1(1:NUM_weight) )
       end do
       yv = jv
       call gauss_solve( nrow_all, Kd, yv )     ! yv <- K^{-1} J b
       ! b1 <- ( b1 - J^T yv / N ) / c, accumulated along the rows so
       ! that the weight index runs over contiguous memory.
       do ib=1,nrow_all
          do i=1,NUM_weight
             b1(i) = b1(i) - jrT(i,ib)*yv(ib)/dble(nb_all)
          end do
       end do
       b1(1:NUM_weight) = b1(1:NUM_weight)/diag_part
    else
       do i=1,NUM_weight
          Gm(i,i) = Gm(i,i) + diag_part
       end do
       call gauss_solve( NUM_weight, Gm, b1 )   ! b1 <- G^{-1} b1
    end if

    ! The cube is padded to ndim_max; the slots beyond ndim(l) are
    ! never written by the loop and must be zero rather than whatever
    ! the caller's stack held (under -finit-real=snan they trap in the
    ! caller's scaling of the step).
    u_out = 0.d0
    itmp=1
    do l=Nlayer,2,-1
       do j=1,ndim(l)
          do i=0,ndim(l-1)
             u_out(l,j,i) = b1(itmp)
             itmp = itmp+1
          end do
       end do
    end do

  END SUBROUTINE ngd_apply_inv

  ! solve A x = b in place (b <- x); Gaussian elimination, partial pivoting
  SUBROUTINE gauss_solve( n, A, b )
    implicit none
    integer,intent(IN) :: n
    real(8),intent(INOUT) :: A(n,n), b(n)
    integer :: i,j,k,ip
    real(8) :: pmax, rtmp, fac, anorm, ptol

    ! A pivot is judged against the size of the matrix, not against zero:
    ! an exactly zero pivot is rare, while a pivot far below the scale of
    ! A means the damped metric is numerically singular and the step that
    ! follows would be noise (or NaN) rather than a descent direction.
    anorm = 0.d0
    do j=1,n
       do i=1,n
          anorm = max( anorm, abs(A(i,j)) )
       end do
    end do
    ptol = 1.d-13*anorm

    do k=1,n-1
       ip = k; pmax = abs(A(k,k))
       do i=k+1,n
          if ( abs(A(i,k)) > pmax ) then
             pmax = abs(A(i,k)); ip = i
          end if
       end do
       if ( pmax <= ptol ) then
          write(*,*) "gauss_solve: the metric is numerically singular at", &
               " column", k, " of", n
          write(*,*) "  pivot", pmax, " against a matrix scale of", anorm
          write(*,*) "  raise the damping (GD_param mu, NGD_mu_bound, or"
          write(*,*) "  NGD_trust), or use NGD_dual on fewer rows."
          stop
       end if
       if ( ip /= k ) then
          do j=k,n
             rtmp=A(k,j); A(k,j)=A(ip,j); A(ip,j)=rtmp
          end do
          rtmp=b(k); b(k)=b(ip); b(ip)=rtmp
       end if
       do i=k+1,n
          fac = A(i,k)/A(k,k)
          A(i,k:n) = A(i,k:n) - fac*A(k,k:n)
          b(i) = b(i) - fac*b(k)
       end do
    end do
    do i=n,1,-1
       b(i) = ( b(i) - sum( A(i,i+1:n)*b(i+1:n) ) )/A(i,i)
    end do

  END SUBROUTINE gauss_solve

  !------------------------------------------------------------------
  SUBROUTINE read_gdlog(dw,r,u,v,m,i_step)
    implicit none
    integer,parameter :: udw=31,ur=32,uu=33,uv=34,um=35
    integer :: i,j
    integer,intent(INOUT) :: i_step
    real(8),intent(INOUT) :: dw(Nlayer,ndim_max,0:ndim_max)
    real(8),intent(INOUT) :: r(Nlayer,ndim_max,0:ndim_max)
    real(8),intent(INOUT) :: u(Nlayer,ndim_max,0:ndim_max)
    real(8),intent(INOUT) :: v(Nlayer,ndim_max,0:ndim_max)
    real(8),intent(INOUT) :: m(Nlayer,ndim_max,0:ndim_max)

    open(udw,file='gd_dw.dat',status='old')
    read(udw,*) i_step
    do i=2,Nlayer
       read(udw,'()')
       do j=1,ndim(i)
          read(udw,*) dw(i,j,0:ndim(i-1))
       end do
    end do
    close(udw)

    if ( gd_method=="ADAGRAD" ) then
       open(ur,file='gd_r.dat',status='old')
       read(ur,*) i_step
       do i=2,Nlayer
          read(ur,'()')
          do j=1,ndim(i)
             read(ur,*) r(i,j,0:ndim(i-1))
          end do
       end do
       close(ur)
    end if

    if ( gd_method=="ADADELTA" ) then
       open(uu,file='gd_u.dat',status='old')
       read(uu,*) i_step
       do i=2,NLayer
          read(uu,'()')
          do j=1,ndim(i)
             read(uu,*) u(i,j,0:ndim(i-1))
          end do
       end do
       close(uu)
    end if

    if ( (gd_method=="RMSPROP").or.(gd_method=="RMSPROP_NESTEROV").or. &
         (gd_method=="ADADELTA").or.(gd_method=="ADAM") ) then
       open(uv,file='gd_v.dat',status='old')
       read(uv,*) i_step
       do i=2,Nlayer
          read(uv,'()')
          do j=1,ndim(i)
             read(uv,*) v(i,j,0:ndim(i-1))
          end do
       end do
       close(uv)
    end if

    if ( gd_method=="ADAM" ) then
       open(um,file='gd_m.dat',status='old')
       read(um,*) i_step
       do i=2,Nlayer
          read(um,'()')
          do j=1,ndim(i)
             read(um,*) m(i,j,0:ndim(i-1))
          end do
       end do
       close(um)
    end if

  END SUBROUTINE read_gdlog


  SUBROUTINE write_gdlog(dw,r,u,v,m,i_step)
    implicit none
    integer,parameter :: udw=36,ur=37,uu=38,uv=39,um=40
    integer :: i,j
    integer,intent(IN) :: i_step
    real(8),intent(IN) :: dw(Nlayer,ndim_max,0:ndim_max)
    real(8),intent(IN) :: r(Nlayer,ndim_max,0:ndim_max)
    real(8),intent(IN) :: u(Nlayer,ndim_max,0:ndim_max)
    real(8),intent(IN) :: v(Nlayer,ndim_max,0:ndim_max)
    real(8),intent(IN) :: m(Nlayer,ndim_max,0:ndim_max)

    open(udw,file='gd_dw.dat',status='replace')
    write(udw,*) i_step
    do i=2,Nlayer
       write(udw,*) "#dw,L",i
       do j=1,ndim(i)
          write(udw,'(100f23.16)') dw(i,j,0:ndim(i-1))
       end do
    end do
    close(udw)

    if ( gd_method=="ADAGRAD" ) then
       open(ur,file='gd_r.dat',status='replace')
       write(ur,*) i_step
       do i=2,Nlayer
          write(ur,*) "#r,L=",i
          do j=1,ndim(i)
             write(ur,'(100e20.10)') r(i,j,0:ndim(i-1))
          end do
       end do
       close(ur)
    end if

    if ( gd_method=="ADADELTA" ) then
       open(uu,file='gd_u.dat',status='replace')
       write(uu,*) i_step
       do i=2,Nlayer
          write(uu,*) "#u,L=",i
          do j=1,ndim(i)
             write(uu,'(100f23.16)') u(i,j,0:ndim(i-1))
          end do
       end do
       close(uu)
    end if

    if ( (gd_method=="RMSPROP").or.(gd_method=="RMSPROP_NESTEROV").or. &
         (gd_method=="ADADELTA").or.(gd_method=="ADAM") ) then
       open(uv,file='gd_v.dat',status='replace')
       write(uv,*) i_step
       do i=2,Nlayer
          write(uv,*) "#v,L=",i
          do j=1,ndim(i)
             write(uv,'(100f23.16)') v(i,j,0:ndim(i-1))
          end do
       end do
       close(uv)
    end if

    if ( gd_method=="ADAM" ) then
       open(um,file='gd_m.dat',status='replace')
       write(um,*) i_step
       do i=2,Nlayer
          write(um,*) "#m,L=",i
          do j=1,ndim(i)
             write(um,'(100f23.16)') m(i,j,0:ndim(i-1))
          end do
       end do
       close(um)
    end if

  END SUBROUTINE write_gdlog

  !> One L-BFGS direction from the stored pairs, by the two-loop
  !! recursion of Nocedal.
  !!
  !!   q = g
  !!   for k = newest .. oldest:   a_k = rho_k s_k.q ;  q -= a_k y_k
  !!   q *= gamma                                     (initial Hessian)
  !!   for k = oldest .. newest:   b = rho_k y_k.q ;    q += (a_k - b) s_k
  !!   d = -q
  !!
  !! A pair with s.y <= 0 would break the positive definiteness the
  !! recursion assumes, so it is discarded: that is the curvature
  !! condition, and on a nonconvex loss it does get violated.
  SUBROUTINE lbfgs_direction( g, d, sv, yv, rhov, npair, newest )
    implicit none
    real(8),intent(IN) :: g(:)
    real(8),intent(OUT) :: d(:)
    real(8),intent(IN) :: sv(:,:), yv(:,:), rhov(:)
    integer,intent(IN) :: npair, newest
    real(8),allocatable :: q(:), alpha(:), tauv(:), hy(:)
    integer,allocatable :: ord(:)
    real(8) :: gamma, b, sy, yy, yhy
    integer :: i, k, n

    n = size(g)
    allocate( q(n), alpha(max(npair,1)), tauv(max(npair,1)) )
    allocate( ord(max(npair,1)) )

    ! ord(i) is the storage slot of the i-th OLDEST pair, which is the
    ! order the recursion builds H in.
    do i = 1, npair
       k = newest - npair + i
       do while ( k < 1 )
          k = k + npair
       end do
       ord(i) = k
    end do

    tauv = 1.d0
    if ( LBFGS_selfscale .and. npair > 0 ) then
       ! The Oren-Luenberger factor of level j is
       !
       !   tau_j = s_j.y_j / ( y_j^T H_j y_j ),
       !
       ! and H_j is the operator built from the pairs OLDER than j,
       ! whose own factors are already fixed: the levels can therefore
       ! be filled oldest first, each one applying the recursion it
       ! sits on top of to its own y.  Approximating H_j by the
       ! identity is the cheap alternative and it is not safe: once
       ! several pairs have contributed their scalings, H_j is nowhere
       ! near the identity and the factor it implies is wrong enough to
       ! lose to plain L-BFGS.  The nested application below costs
       ! O(m^2 n) per direction, which next to one full-batch gradient
       ! is nothing.
       allocate( hy(n) )
       do i = 1, npair
          sy = dot_product( sv(1:n,ord(i)), yv(1:n,ord(i)) )
          call ss_apply_h( yv(1:n,ord(i)), hy, sv, yv, rhov, ord, i-1, tauv, n )
          yhy = dot_product( yv(1:n,ord(i)), hy )
          if ( yhy > 0.d0 .and. sy > 0.d0 ) tauv(i) = sy/yhy
       end do
       deallocate( hy )
    end if

    ! ---- the two-loop recursion ----
    q = g
    do i = npair, 1, -1
       alpha(i) = rhov(ord(i))*dot_product( sv(1:n,ord(i)), q )
       q = q - alpha(i)*yv(1:n,ord(i))
    end do

    ! The initial Hessian.  Standard L-BFGS takes gamma = s.y / y.y of
    ! the newest pair and applies it once, at the innermost level: one
    ! scalar stands in for the curvature of every stored pair.
    ! Self-scaled BFGS gives every pair its own factor at its own
    ! level (tauv above), so H_0 is the identity there.
    gamma = 1.d0
    if ( npair > 0 .and. .not. LBFGS_selfscale ) then
       sy = dot_product( sv(1:n,newest), yv(1:n,newest) )
       yy = dot_product( yv(1:n,newest), yv(1:n,newest) )
       if ( yy > 0.d0 ) gamma = sy/yy
    end if
    q = gamma*q

    ! With the self-scaled update
    !   H_{j+1} = tau_j V_j^T H_j V_j + rho_j s_j s_j^T,
    !   V_j = I - rho_j y_j s_j^T,
    ! applying it to a vector gives, with z the value the recursion has
    ! already accumulated and b = rho_j y_j.z,
    !   H_{j+1} q = tau_j ( z - b s_j ) + alpha_j s_j,
    ! so the only difference from the plain two-loop is the factor
    ! tau_j.  All tau = 1 recovers it exactly, and with a single stored
    ! pair the two agree identically (tau_1 = gamma) -- verified to
    ! eight digits over fifty iterations on the EHD cold start.
    do i = 1, npair
       b = rhov(ord(i))*dot_product( yv(1:n,ord(i)), q )
       q = tauv(i)*q + ( alpha(i) - tauv(i)*b )*sv(1:n,ord(i))
    end do
    d = -q
    deallocate( q, alpha, tauv, ord )
  END SUBROUTINE lbfgs_direction

  !> Apply the limited-memory operator built from the nlev OLDEST
  !! stored pairs, with the self-scaling factors already determined for
  !! those levels, to an arbitrary vector.  nlev = 0 is the identity.
  SUBROUTINE ss_apply_h( v, out, sv, yv, rhov, ord, nlev, tauv, n )
    implicit none
    integer,intent(IN) :: n, nlev
    real(8),intent(IN) :: v(n)
    real(8),intent(OUT) :: out(n)
    real(8),intent(IN) :: sv(:,:), yv(:,:), rhov(:), tauv(:)
    integer,intent(IN) :: ord(:)
    real(8) :: al(max(nlev,1)), b
    integer :: i
    out = v
    if ( nlev <= 0 ) return
    do i = nlev, 1, -1
       al(i) = rhov(ord(i))*dot_product( sv(1:n,ord(i)), out )
       out = out - al(i)*yv(1:n,ord(i))
    end do
    do i = 1, nlev
       b = rhov(ord(i))*dot_product( yv(1:n,ord(i)), out )
       out = tauv(i)*out + ( al(i) - tauv(i)*b )*sv(1:n,ord(i))
    end do
  END SUBROUTINE ss_apply_h


END MODULE optimizer_module
