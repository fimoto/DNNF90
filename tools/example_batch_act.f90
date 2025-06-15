! Does the batched value path agree with the per-point path for every
! activation?
!
! The batched path applies sigma to a whole minibatch and multiplies the
! deltas by sigma'.  Both must be the activation the table set carries,
! for the value and for the derivative alike; taking either from tanh
! leaves the weight gradient wrong wherever the activation is not tanh.
! This compares the two paths on the same weights and points.
program batchact
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, tabset_t, &
       tabset_from_current
  use net_module, only: net_t, net_init, work_t, work_init
  use train_module, only: twork_t, twork_init, grad_t, grad_init, grad_zero, &
       bwork_t, bwork_init, net_grad_batch, net_grad_point, net_forward_point
  implicit none
  integer,parameter :: D0=2, NB=7, NL=4
  type(net_t) :: nt
  type(tabset_t) :: ts
  type(twork_t) :: tw
  type(grad_t) :: g1, g2
  type(bwork_t) :: bw
  real(8) :: X(D0,NB), Y(NB), emax, den, tval(64), seed(64)
  integer :: sd(D0,1), dims(NL), ia, ib, l, j, k
  integer :: ifail = 0
  character(len=8) :: nm(0:4)
  nm = (/ "TANH    ","SIN     ","ERF     ","BESSEL  ","BESSEL1 " /)
  sd = 0
  dims = (/ D0, 6, 6, 1 /)
  call random_seed()
  call random_number( X );  X = 1.2d0*(X-0.5d0)
  call random_number( Y );  Y = Y - 0.5d0
  call init_hod_tables( D0, 1, 0, sd )
  do ia = 0, 4
     call tabset_from_current( ts )
     ts%iact = ia
     call net_init( nt, NL, dims, ts )
     call twork_init( tw, nt );  call grad_init( g1, nt );  call grad_init( g2, nt )
     call bwork_init( bw, nt, NB )
     call random_number( nt%w );  nt%w = 0.6d0*(nt%w-0.5d0)
     call grad_zero( g1 );  call net_grad_batch( nt, bw, X, Y, 1.d0, g1 )
     call grad_zero( g2 )
     do ib = 1, NB
        call net_forward_point( nt, tw, X(:,ib), tval(1:NUM_alpha) )
        seed = 0.d0
        seed(1) = tval(1) - Y(ib)          ! d(1/2 (u-y)^2)/du
        call net_grad_point( nt, tw, X(:,ib), seed(1:NUM_alpha), g2 )
     end do
     emax = 0.d0;  den = 0.d0
     do l=2,NL
        do j=1,dims(l)
           do k=0,dims(l-1)
              emax = max( emax, abs(g1%nabla(l,j,k)-g2%nabla(l,j,k)) )
              den  = max( den , abs(g2%nabla(l,j,k)) )
           end do
        end do
     end do
     write(*,'(a,a,a,es12.4)') ' ', nm(ia), ' batch vs per-point, max rel = ', emax/max(den,1.d-30)
     if ( emax/max(den,1.d-30) > 1.d-8 ) ifail = ifail + 1
  end do
  if ( ifail == 0 ) then
     write(*,'(a)') " passed"
  else
     write(*,'(a,i0,a)') " FAILED on ", ifail, " activation(s)"
     stop 1
  end if
end program batchact
