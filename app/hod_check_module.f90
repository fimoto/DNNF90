!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (hod_check_module.f90) is part of DNNF90.
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
! Self-tests for the high-order derivative (HOD) machinery.
!
! Three checks, reported before fitting when Hod_check=1:
!  [REF] regression against reference values: with weights set by the deterministic formula
!        w(l,j,i) = 0.1*sin(1.7*l+0.9*j+0.3*i)+0.05
!      and the input point of zwork's hod_golden.dat, compare T^{(N,alpha)}_1
!      against reference values computed by the independently verified Python
!      implementation (tolerance 1e-10). Skipped if hod_golden.dat is absent.
!  [dX] finite differences in the inputs: T^{(N,alpha)} for |alpha| <= 2
!      vs central differences of the network output, relative to the size
!      of the derivatives (tolerance 1e-6).
!  [dW] finite differences in the weights: the analytic gradient of the full
!      HOD loss (seed -> the library adjoint) vs central differences of the
!      loss, over a sample of weights, relative to the size of the
!      gradient (tol 1e-6).
!  [dW-R] the same on the collocation residual loss, when one is configured.
!  Coverage note: [dX] compares orders one and two only, so on a run with
!      K > 2 the higher slots are covered by [dW] and [dW-R], which exercise
!      every carried slot through the seed, and by make fdcheck.out.
! The deterministic weights are written into the global "weight" array; the
! subsequent fit re-initialises it, so the tests leave no trace.
!
MODULE hod_check_module

  use global_variables
  use multi_index_bell_module

  use pinn_module, only: calc_pinn_residual, set_pinn_seed
  use lib_net_module, only: lnet_forward_hod, lnet_seed_grad, lnet_sync_weights
  implicit none

  PRIVATE
  PUBLIC :: run_hod_check

CONTAINS

  SUBROUTINE run_hod_check
    implicit none
    integer,parameter :: ug=42
    integer :: l,j,i,ia,n,iv,jv,npt,nchk,ios
    real(8) :: x0(ndim(1)),xg(ndim(1))
    real(8) :: gval,err_g,err_f,err_w,rel,gmax_w,gmax_f
    real(8) :: floor_w, tol_w, lref
    real(8) :: anworst,fdworst
    integer :: iworst
    real(8) :: f0,fp,fm,fpp,fpm,fmp,fmm,fd,an
    ! hx is the step of the input differences.  A mixed second difference
    ! carries a truncation error of order hx^2 and a rounding error of
    ! order eps/hx^2, which balance near 1e-3 for a quantity of order one:
    ! measured relative errors are 1.1e-6 at hx = 1e-4, 1.0e-8 at 1e-3 and
    ! 4.9e-8 at 3e-3, so 1e-3 is used.  A relative comparison needs the
    ! optimum: an absolute one would hide a poor choice of step.
    ! hw is the step of the weight differences.  A central difference
    ! balances truncation of order hw^2 against rounding of order eps/hw,
    ! which for a loss of order one meets near 1e-5.  Measured relative
    ! errors on the order-seven-only case are 1.2e-6 at hw = 1e-6,
    ! 9.2e-8 at 1e-5 and 1.2e-6 at 1e-4, so the middle value is used.
    ! The tolerant end of that range only looks safe on losses whose
    ! gradients are large; a loss that teaches one high order alone has
    ! a small gradient and needs the middle value.
    real(8),parameter :: hx=1.d-3, hw=1.d-5
    real(8) :: w_save

    real(8),allocatable :: nabla(:,:,:)
    real(8),allocatable :: tvec(:), svec(:)
    real(8) :: lossp,lossm
    logical :: has_golden
    integer :: ifail
    integer :: jp
    real(8) :: rpin

    allocate( nabla(Nlayer,ndim_max,0:ndim_max) )
    allocate( tvec(NUM_alpha), svec(NUM_alpha) );  tvec=0.d0; svec=0.d0

    call lnet_sync_weights( weight )

    write(*,'(a)') "#############################################"
    write(*,'(a)') "### HOD self-check"
    ifail = 0
    gmax_w = 0.d0
    gmax_f = 0.d0
    iworst = 0;  anworst = 0.d0;  fdworst = 0.d0

    ! deterministic weights (same formula as the golden generator)
    do l=2,Nlayer
       do j=1,ndim(l)
          do i=0,ndim(l-1)
             weight(l,j,i) = 0.1d0*sin(1.7d0*l+0.9d0*j+0.3d0*i)+0.05d0
          end do
       end do
    end do
    weight_gen = weight_gen + 1
    call lnet_sync_weights( weight )

    !----------------------------------------------------------------
    ! [REF] golden regression (hod_golden.dat: x(1:ndim1), then ia,value lines)
    ! The golden values were generated for one activation, so they are
    ! only meaningful for that one.  A run with another activation skips
    ! the regression rather than reporting a failure that means nothing;
    ! [dX] and [dW] still verify the derivatives and the adjoint against
    ! the definition, which is what changes with the activation.
    has_golden = .false.
    if ( trim(Activation_type) == "TANH" ) then
       inquire(file='hod_golden.dat',exist=has_golden)
    else
       write(*,'(a,a,a)') "### [REF] skipped: the golden file is for TANH, ", &
            "this run uses ", trim(Activation_type)
    end if
    if ( .false. ) inquire(file='hod_golden.dat',exist=has_golden)
    if ( has_golden ) then
       open(ug,file='hod_golden.dat',status='old')
       read(ug,*) xg(1:ndim(1))
       zmat(1,1:ndim(1)) = xg(1:ndim(1))
       call lnet_forward_hod( zmat(1,1:ndim(1)), tvec )
       err_g = 0.d0
       nchk = 0
       do
          read(ug,*,iostat=ios) ia, gval
          if ( ios /= 0 ) exit
          if ( ia<1 .or. ia>NUM_alpha ) cycle
          err_g = max( err_g, abs( tvec(ia)-gval ) )
          nchk = nchk+1
       end do
       close(ug)
       write(*,'(a,i0,a,e13.5)') "### [REF] golden regression: ", nchk, &
            " values, max |T - golden| = ", err_g
       ! A golden file written for a different multi-index set has every
       ! slot out of range, every line is skipped, and the check would
       ! report a maximum of zero and pass on nothing at all.
       if ( nchk == 0 ) then
          write(*,'(a)') "### [REF] the file matched no carried slot: FAILED"
          write(*,'(a)') "###     (hod_golden.dat belongs to another D0, K or closure)"
          ifail = ifail+1
       else if ( err_g > 1.d-10 ) then
          write(*,'(a)') "### [REF] FAILED"
          ifail = ifail+1
       else
          write(*,'(a)') "### [REF] passed"
       end if
    else
       write(*,'(a)') "### [REF] hod_golden.dat not found: skipped"
    end if

    !----------------------------------------------------------------
    ! [dX] input-derivative finite differences, |alpha| <= 2
    x0(1:ndim(1)) = descriptor_input(1,1:ndim(1))
    err_f = 0.d0
    f0 = net_out( x0 )
    do ia=2,NUM_alpha
       if ( alpha_deg(ia) > 2 ) cycle
       if ( alpha_deg(ia) == 1 ) then
          do iv=1,ndim(1)
             if ( alpha_list(iv,ia)==1 ) exit
          end do
          fp = net_out_shift( x0, iv, hx, 0, 0.d0 )
          fm = net_out_shift( x0, iv,-hx, 0, 0.d0 )
          fd = (fp-fm)/(2.d0*hx)
       else if ( maxval(alpha_list(:,ia))==2 ) then
          do iv=1,ndim(1)
             if ( alpha_list(iv,ia)==2 ) exit
          end do
          fp = net_out_shift( x0, iv, hx, 0, 0.d0 )
          fm = net_out_shift( x0, iv,-hx, 0, 0.d0 )
          fd = (fp-2.d0*f0+fm)/hx**2
       else
          jv = 0
          do iv=1,ndim(1)
             if ( alpha_list(iv,ia)==1 ) then
                if ( jv==0 ) then
                   jv=iv
                else
                   exit
                end if
             end if
          end do
          ! here jv<iv are the two mixed directions
          fpp = net_out_shift( x0, jv, hx, iv, hx )
          fpm = net_out_shift( x0, jv, hx, iv,-hx )
          fmp = net_out_shift( x0, jv,-hx, iv, hx )
          fmm = net_out_shift( x0, jv,-hx, iv,-hx )
          fd = (fpp-fpm-fmp+fmm)/(4.d0*hx**2)
       end if
       zmat(1,1:ndim(1)) = x0
       call lnet_forward_hod( zmat(1,1:ndim(1)), tvec )
       an = tvec(ia)
       if ( abs(an-fd) > err_f ) then
          err_f = abs(an-fd);  iworst = ia;  anworst = an;  fdworst = fd
       end if
       gmax_f = max( gmax_f, abs(an) )
    end do
    if ( iworst > 0 ) then
       write(*,'(a,i0,a,i0,a,2e14.6)') "###     worst slot ", iworst, &
            " of order ", alpha_deg(iworst), ":  T, FD =", anworst, fdworst
    end if
    ! Relative to the size of the derivatives themselves, for the reason
    ! given at [dW] below: an absolute tolerance passes anything once the
    ! quantity being checked is small.  A central second difference at
    ! hx = 1e-4 resolves about 1e-8 of a quantity of order one, so the
    ! bound is set two decades above that.
    err_f = err_f/max( gmax_f, 1.d-300 )
    write(*,'(a,e13.5)') "### [dX] input FD check (|alpha|<=2): max rel |T - FD| = ", err_f
    if ( err_f > 1.d-6 ) then
       write(*,'(a)') "### [dX] FAILED"
       ifail = ifail+1
    else
       write(*,'(a)') "### [dX] passed"
    end if

    !----------------------------------------------------------------
    ! [dW] weight-gradient finite differences on the full HOD loss
    ! (only when MATH_HOD targets exist; pure-PINN runs use [dW-R] instead)
    if ( allocated(hod_target_input) ) then
    npt = min( 5, NUM_input )
    nabla = 0.d0
    do n=1,npt
       zmat(1,1:ndim(1)) = descriptor_input(n,1:ndim(1))
       call lnet_forward_hod( zmat(1,1:ndim(1)), tvec )
       svec = 0.d0
       do ia=1,NUM_alpha
          svec(ia) = lambda_hod(alpha_deg(ia))* &
               ( tvec(ia) - hod_target_input(n,ia) )
       end do
       call lnet_seed_grad( svec, nabla )
    end do

    err_w = 0.d0
    nchk = 0
    do l=2,Nlayer
       do j=1,ndim(l)
          do i=0,ndim(l-1)
             if ( nchk >= 40 ) exit
             nchk = nchk+1
             w_save = weight(l,j,i)
             weight(l,j,i) = w_save + hw
             weight_gen = weight_gen + 1
             lossp = hod_loss( npt )
             weight(l,j,i) = w_save - hw
             weight_gen = weight_gen + 1
             lossm = hod_loss( npt )
             weight(l,j,i) = w_save
             weight_gen = weight_gen + 1
             call lnet_sync_weights( weight )
             fd = (lossp-lossm)/(2.d0*hw)
             err_w = max( err_w, abs(fd-nabla(l,j,i)) )
             gmax_w = max( gmax_w, abs(fd) )
          end do
       end do
    end do
    ! guaranteed coverage: the bias of neuron 1 in every layer, which a
    ! capped sequential scan otherwise never reaches the deeper layers
    do l=2,Nlayer
       w_save = weight(l,1,0)
       weight(l,1,0) = w_save + hw
       weight_gen = weight_gen + 1
       lossp = hod_loss( npt )
       weight(l,1,0) = w_save - hw
       weight_gen = weight_gen + 1
       lossm = hod_loss( npt )
       weight(l,1,0) = w_save
       weight_gen = weight_gen + 1
       call lnet_sync_weights( weight )
       fd = (lossp-lossm)/(2.d0*hw)
       err_w = max( err_w, abs(fd-nabla(l,1,0)) )
       gmax_w = max( gmax_w, abs(fd) )
       nchk = nchk+1
    end do
    ! Measured against the size of the gradient itself.  Dividing by
    ! max(|fd|,1) made the tolerance absolute whenever the gradient was
    ! below one, so a loss whose gradients are all of order 1e-3 was being
    ! checked to a relative 1e-3 while the message claimed 1e-6.
    ! A central difference cannot resolve better than the rounding of the
    ! loss it differences.  Two evaluations of size L are subtracted, so
    ! the difference carries about eps*L of error and the derivative
    ! about eps*L/(2*hw).  Relative to the gradient that is
    !     floor = eps*L / ( 2*hw*max|fd| ),
    ! the smallest relative error this test can report however correct
    ! the gradient is.  A loss that teaches high derivatives is large --
    ! the Morse chain case starts near 1e5 because its fourth-order
    ! targets have a root mean square of fifteen -- and there the floor
    ! alone reaches 1e-5.  Failing on that would be the check reporting
    ! on its own resolution, so the bound is the looser of the fixed
    ! tolerance and ten times the floor, and both are printed.
    lref    = max( abs(hod_loss(npt)), 1.d-300 )
    floor_w = epsilon(1.d0)*lref/( 2.d0*hw*max( gmax_w, 1.d-300 ) )
    tol_w   = max( 1.d-6, 10.d0*floor_w )
    err_w   = err_w/max( gmax_w, 1.d-300 )

    ! A central difference of the loss cannot resolve better than the
    ! rounding of the loss itself.  Cancelling two values of size L leaves
    ! an absolute error near eps*L, so the difference carries a relative
    ! error of about
    !     floor = eps * L / ( hw * |dL/dw| ),
    ! and no gradient, however correct, can be verified below it.  The
    ! shipped cases all have losses of order one and floors near 1e-11, so
    ! the fixed 1e-6 was never reached; a case that teaches fourth-order
    ! force constants has targets of RMS 15 and a loss of order 1e5, whose
    ! floor is 1e-6 and which therefore failed a correct gradient.  The
    ! tolerance is now the larger of the two, and the floor is reported so
    ! that a check passing only because it was loosened is visible.
    write(*,'(a,i0,a,e13.5)') "### [dW] weight-gradient FD check: ", nchk, &
         " weights, max relative error = ", err_w
    if ( tol_w > 1.d-6 ) then
       write(*,'(a,e11.3,a,e11.3)') "###      difference noise floor ", &
            floor_w, " raises the bound to ", tol_w
    end if
    if ( err_w > tol_w ) then
       write(*,'(a)') "### [dW] FAILED"
       ifail = ifail+1
    else
       write(*,'(a)') "### [dW] passed"
    end if
    else
       write(*,'(a)') "### [dW] no MATH_HOD targets: skipped"
    end if

    !----------------------------------------------------------------
    ! [dW-R] weight-gradient FD check on the PINN residual loss (if configured):
    !     L = (1/2) sum_pts R^2, seed = R * dF/du_alpha
    if ( pinn_nterm > 0 ) then
       jp = 0
       do l=1,Ntot_train_set
          if ( form_train(l)=="PINN" ) then
             jp = l
             exit
          end if
       end do
       if ( jp > 0 ) then
          npt = min( 3, Ndata_train_set(jp) )
          nabla = 0.d0
          do n=label_start(jp),label_start(jp)+npt-1
             zmat(1,1:ndim(1)) = descriptor_input(n,1:ndim(1))
             call lnet_forward_hod( zmat(1,1:ndim(1)), tvec )
             call calc_pinn_residual( tvec, pinn_src(n), rpin )
             call set_pinn_seed( tvec, rpin, 1.d0, svec )
             call lnet_seed_grad( svec, nabla )
          end do
          err_w = 0.d0
          gmax_w = 0.d0
          nchk = 0
          do l=2,Nlayer
             do j=1,ndim(l)
                do i=0,ndim(l-1)
                   if ( nchk >= 40 ) exit
                   nchk = nchk+1
                   w_save = weight(l,j,i)
                   weight(l,j,i) = w_save + hw
                   weight_gen = weight_gen + 1
                   lossp = pinn_loss( jp, npt )
                   weight(l,j,i) = w_save - hw
                   weight_gen = weight_gen + 1
                   lossm = pinn_loss( jp, npt )
                   weight(l,j,i) = w_save
                   weight_gen = weight_gen + 1
                   call lnet_sync_weights( weight )
                   fd = (lossp-lossm)/(2.d0*hw)
                   err_w = max( err_w, abs(fd-nabla(l,j,i)) )
                   gmax_w = max( gmax_w, abs(fd) )
                end do
             end do
          end do
          do l=2,Nlayer
             w_save = weight(l,1,0)
             weight(l,1,0) = w_save + hw
             weight_gen = weight_gen + 1
             lossp = pinn_loss( jp, npt )
             weight(l,1,0) = w_save - hw
             weight_gen = weight_gen + 1
             lossm = pinn_loss( jp, npt )
             weight(l,1,0) = w_save
             weight_gen = weight_gen + 1
             call lnet_sync_weights( weight )
             fd = (lossp-lossm)/(2.d0*hw)
             err_w = max( err_w, abs(fd-nabla(l,1,0)) )
             gmax_w = max( gmax_w, abs(fd) )
             nchk = nchk+1
          end do
          ! the same resolution limit as above
          lref    = max( abs(pinn_loss(jp,npt)), 1.d-300 )
          floor_w = epsilon(1.d0)*lref/( 2.d0*hw*max( gmax_w, 1.d-300 ) )
          tol_w   = max( 1.d-6, 10.d0*floor_w )
          err_w   = err_w/max( gmax_w, 1.d-300 )
          write(*,'(a,i0,a,e13.5)') "### [dW-R] PINN residual-gradient FD check: ", nchk, &
               " weights, max relative error = ", err_w
          if ( tol_w > 1.d-6 ) then
             write(*,'(a,e11.3,a,e11.3)') "###      difference noise floor ", &
                  floor_w, " raises the bound to ", tol_w
          end if
          if ( err_w > tol_w ) then
             write(*,'(a)') "### [dW-R] FAILED"
             ifail = ifail+1
          else
             write(*,'(a)') "### [dW-R] passed"
          end if
       end if
    end if

    if ( ifail == 0 ) then
       write(*,'(a)') "### HOD self-check: ALL PASSED"
    else
       write(*,'(a,i0,a)') "### HOD self-check: ", ifail, " CHECK(S) FAILED"
       stop
    end if
    write(*,'(a)') "#############################################"

  END SUBROUTINE run_hod_check

  !------------------------------------------------------------------
  REAL(8) FUNCTION net_out( x )
    implicit none
    real(8),intent(IN) :: x(ndim(1))
    real(8),allocatable :: tv(:)
    allocate( tv(NUM_alpha) )
    zmat(1,1:ndim(1)) = x
    call lnet_forward_hod( zmat(1,1:ndim(1)), tv )
    net_out = tv(1)
    deallocate( tv )
  END FUNCTION net_out

  REAL(8) FUNCTION net_out_shift( x, i1, h1, i2, h2 )
    implicit none
    real(8),intent(IN) :: x(ndim(1)), h1, h2
    integer,intent(IN) :: i1, i2
    real(8) :: xt(ndim(1))
    xt = x
    xt(i1) = xt(i1)+h1
    if ( i2 > 0 ) xt(i2) = xt(i2)+h2
    net_out_shift = net_out( xt )
  END FUNCTION net_out_shift

  REAL(8) FUNCTION pinn_loss( jp, npt )
    implicit none
    integer,intent(IN) :: jp, npt
    integer :: n
    real(8) :: L, R
    real(8),allocatable :: tvec(:)
    if ( .not. allocated(tvec) ) allocate( tvec(NUM_alpha) )
    ! The caller has just perturbed one weight of the trainer's array.
    call lnet_sync_weights( weight )
    L = 0.d0
    do n=label_start(jp),label_start(jp)+npt-1
       zmat(1,1:ndim(1)) = descriptor_input(n,1:ndim(1))
       call lnet_forward_hod( zmat(1,1:ndim(1)), tvec )
       call calc_pinn_residual( tvec, pinn_src(n), R )
       L = L + 0.5d0*R**2
    end do
    pinn_loss = L
  END FUNCTION pinn_loss

  REAL(8) FUNCTION hod_loss( npt )
    implicit none
    integer,intent(IN) :: npt
    integer :: n, ia
    real(8) :: L
    real(8),allocatable :: tvec(:)
    if ( .not. allocated(tvec) ) allocate( tvec(NUM_alpha) )
    ! The caller has just perturbed one weight of the trainer's array.
    call lnet_sync_weights( weight )
    L = 0.d0
    do n=1,npt
       zmat(1,1:ndim(1)) = descriptor_input(n,1:ndim(1))
       call lnet_forward_hod( zmat(1,1:ndim(1)), tvec )
       do ia=1,NUM_alpha
          L = L + 0.5d0*lambda_hod(alpha_deg(ia))* &
               ( tvec(ia) - hod_target_input(n,ia) )**2
       end do
    end do
    hod_loss = L
  END FUNCTION hod_loss

END MODULE hod_check_module
