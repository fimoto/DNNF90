! FD verification of net_backward_point for a dense K=3 seed.
program dbg_backward_fd
  use multi_index_bell_module, only: tabset_t, tabset_init, tabset_free
  use net_module, only: net_t, net_init, net_free, work_t, work_init, &
       work_free, net_eval_hod
  use train_module, only: twork_t, grad_t, bwork_t, twork_init, grad_init, &
       grad_zero, bwork_init, net_forward_point, net_backward_point, &
       net_grad_batch, grad_free, twork_free, bwork_free
  implicit none
  integer,parameter :: D0=2, K=3, NL=3, NB=7
  integer :: ndim(NL), dummy(D0,1), l,i,j, na
  type(tabset_t) :: ts
  type(net_t)    :: nt
  type(twork_t)  :: tw
  type(grad_t)   :: g, gref
  type(bwork_t)  :: bw
  type(work_t)   :: wk
  integer :: ib, ifail
  real(8) :: Xb(D0,NB), Yb(NB), coefb, gnorm, lp, lm
  real(8),allocatable :: seedv(:), tev(:)
  real(8),allocatable :: tout(:), tp(:), tm(:), seed(:)
  real(8) :: h, num, ana, emax, x(D0), wsave
  ndim = (/ D0, 5, 1 /)
  call tabset_init( ts, D0, K, 0, dummy )        ! nseed=0 -> dense
  na = ts%na
  call net_init( nt, NL, ndim, ts )
  ! deterministic weights
  do l=2,NL
     do j=1,ndim(l)
        do i=0,ndim(l-1)
           nt%w(l,j,i) = 0.3d0*sin( 17.d0*l + 7.d0*j + 3.d0*i + 0.5d0 )
        end do
     end do
  end do
  call twork_init( tw, nt )
  call grad_init( g, nt )
  allocate( tout(na), tp(na), tm(na), seed(na), seedv(na), tev(na) )
  ifail = 0
  do i=1,na
     seed(i) = cos( 2.7d0*i )        ! arbitrary fixed seed vector
  end do
  x = (/ 0.37d0, -0.61d0 /)

  call net_forward_point( nt, tw, x, tout )
  g%nabla = 0.d0
  call net_backward_point( nt, tw, seed, g )

  h = 1.d-6
  emax = 0.d0
  do l=2,NL
     do j=1,ndim(l)
        do i=0,ndim(l-1)
           wsave = nt%w(l,j,i)
           nt%w(l,j,i) = wsave + h
           call net_forward_point( nt, tw, x, tp )
           nt%w(l,j,i) = wsave - h
           call net_forward_point( nt, tw, x, tm )
           nt%w(l,j,i) = wsave
           num = sum( seed(1:na)*(tp(1:na)-tm(1:na)) )/(2.d0*h)
           ana = g%nabla(l,j,i)
           emax = max( emax, abs(num-ana)/max(abs(num),1.d-12) )
        end do
     end do
  end do
  write(*,'(a,e12.4)') "point adjoint vs FD          : ", emax
  if ( emax > 1.d-6 ) ifail = ifail + 1

  !--------------------------------------------------------------------
  ! Batched value path.  Two independent checks:
  !   (a) against the per-point engine, which is the only other way the
  !       same gradient can be formed in this tree
  !   (b) against central differences of the batch loss
  !--------------------------------------------------------------------
  call bwork_init( bw, nt, NB )
  do ib=1,NB
     do i=1,D0
        Xb(i,ib) = 0.4d0*sin( 1.3d0*ib + 0.7d0*i )
     end do
     Yb(ib) = 0.2d0*cos( 2.1d0*ib )
  end do
  coefb = 0.7d0

  call grad_zero( g )
  call net_grad_batch( nt, bw, Xb, Yb, coefb, g )

  ! (a) the same batch, point by point
  call grad_init( gref, nt )
  call grad_zero( gref )
  do ib=1,NB
     call net_forward_point( nt, tw, Xb(:,ib), tout )
     seedv = 0.d0
     seedv(1) = coefb*( tout(1) - Yb(ib) )
     call net_backward_point( nt, tw, seedv, gref )
  end do
  emax = 0.d0
  gnorm = 0.d0
  do l=2,NL
     do j=1,ndim(l)
        do i=0,ndim(l-1)
           emax  = max( emax, abs( g%nabla(l,j,i) - gref%nabla(l,j,i) ) )
           gnorm = max( gnorm, abs( gref%nabla(l,j,i) ) )
        end do
     end do
  end do
  write(*,'(a,e12.4)') "batch vs per-point engine    : ", emax/max(gnorm,1.d-12)
  if ( emax/max(gnorm,1.d-12) > 1.d-12 ) ifail = ifail + 1

  ! (b) central differences of the batch loss
  h = 1.d-5
  emax = 0.d0
  gnorm = 0.d0
  do l=2,NL
     do j=1,ndim(l)
        do i=0,ndim(l-1)
           wsave = nt%w(l,j,i)
           nt%w(l,j,i) = wsave + h
           lp = batch_loss( nt, tw, Xb, Yb, coefb, NB, tout )
           nt%w(l,j,i) = wsave - h
           lm = batch_loss( nt, tw, Xb, Yb, coefb, NB, tout )
           nt%w(l,j,i) = wsave
           num = (lp-lm)/(2.d0*h)
           emax  = max( emax, abs( num - g%nabla(l,j,i) ) )
           gnorm = max( gnorm, abs( g%nabla(l,j,i) ) )
        end do
     end do
  end do
  write(*,'(a,e12.4)') "batch adjoint vs FD          : ", emax/max(gnorm,1.d-12)
  if ( emax/max(gnorm,1.d-12) > 1.d-6 ) ifail = ifail + 1

  !--------------------------------------------------------------------
  ! Evaluation entry against the training forward.  net_eval_hod is the
  ! path a host uses for inference and carries its own matrix product,
  ! so it is checked against the forward the gradients are built on.
  !--------------------------------------------------------------------
  call work_init( wk, nt )
  call net_forward_point( nt, tw, x, tout )
  call net_eval_hod( nt, wk, x, tev )
  emax = maxval( abs( tev(1:na) - tout(1:na) ) )/max( maxval(abs(tout(1:na))), 1.d-12 )
  write(*,'(a,e12.4)') "net_eval_hod vs forward      : ", emax
  if ( emax > 1.d-12 ) ifail = ifail + 1

  ! Release what the check built.  It is short lived, but this file is
  ! also the smallest complete example of the instance lifetime.
  call work_free( wk );  call grad_free( gref );  call bwork_free( bw )
  call grad_free( g );   call twork_free( tw )
  call net_free( nt );   call tabset_free( ts )
  deallocate( tout, tp, tm, seed, seedv, tev )

  if ( ifail == 0 ) then
     write(*,'(a)') "gradient checks: ALL PASSED"
  else
     write(*,'(a,i0,a)') "gradient checks: ", ifail, " FAILED"
     ! This is the check the build documentation points at first, so a
     ! failure has to reach the shell: reporting it only in the printout
     ! let every pipeline treat a broken gradient as a success.
     stop 1
  end if

CONTAINS

  !> (coef/2) sum_n ( N(x_n) - y_n )^2 over the batch
  REAL(8) FUNCTION batch_loss( ntl, twl, X, Y, coef, nb, tbuf )
    implicit none
    type(net_t),intent(IN) :: ntl
    type(twork_t),intent(INOUT) :: twl
    real(8),intent(IN) :: X(:,:), Y(:), coef
    integer,intent(IN) :: nb
    real(8),intent(INOUT) :: tbuf(:)
    integer :: n
    real(8) :: L
    L = 0.d0
    do n=1,nb
       call net_forward_point( ntl, twl, X(:,n), tbuf )
       L = L + 0.5d0*coef*( tbuf(1) - Y(n) )**2
    end do
    batch_loss = L
  END FUNCTION batch_loss

end program dbg_backward_fd
