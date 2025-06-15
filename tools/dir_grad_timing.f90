! This file (tools/dir_grad_timing.f90) is part of DNNF90.
! (MIT License; see LICENSE at the repository root.)
!
! One weight gradient of the directional loss of the framework
! comparison: the squared K-th pure derivative along x_1, summed over
! a fixed batch, on a 4-8-8-1 tanh network.  This is the loss every
! column of the directional group differentiates, so the rows of the
! comparison table put the same computation side by side.  The seed-closure
! mechanism carries exactly the K+1 pure-x_1 multi-indices; the kernels
! called are the ones the trainer calls.
!
!   dir_grad_timing.out <K> <repeat>
program dir_grad_timing
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha
  use net_module, only: net_t, net_init, net_free
  use train_module, only: twork_t, twork_init, twork_free, grad_t, &
       grad_init, grad_free, grad_zero, net_forward_point, net_backward_point
  implicit none
  integer,parameter :: NPT = 20
  type(net_t) :: nt
  type(twork_t) :: tw
  type(grad_t) :: g
  real(8),allocatable :: tval(:), seed(:), x(:,:)
  real(8) :: t0, t1, loss
  integer :: dims(4), kmax, nrep, i, n
  integer :: sd(4,1)
  character(32) :: a
  call get_command_argument(1,a); read(a,*) kmax
  call get_command_argument(2,a); read(a,*) nrep
  sd = 0;  sd(1,1) = kmax
  call init_hod_tables( 4, kmax, 1, sd )
  dims = (/ 4, 8, 8, 1 /)
  call net_init( nt, 4, dims )
  call twork_init( tw, nt )
  call grad_init( g, nt )
  allocate( tval(NUM_alpha), seed(NUM_alpha), x(4,NPT) )
  call random_seed();  call random_number( x );  x = 2.d0*x - 1.d0
  call random_number( nt%w );  nt%w = 0.3d0*(nt%w - 0.5d0)
  call one_batch()
  call cpu_time(t0)
  do i = 1, nrep
     call one_batch()
  end do
  call cpu_time(t1)
  write(*,'(a,es22.12)') '    computed loss =', loss
  write(*,'(a,i2,a,i4,a,f10.4,a)') ' directional weight gradient  K=', &
       kmax, '  carried=', NUM_alpha, '   ', 1.d3*(t1-t0)/nrep, ' ms'
  call grad_free( g );  call twork_free( tw );  call net_free( nt )
contains
  subroutine one_batch()
    loss = 0.d0
    call grad_zero( g )
    do n = 1, NPT
       call net_forward_point( nt, tw, x(:,n), tval )
       seed = 0.d0
       seed(NUM_alpha) = 2.d0*tval(NUM_alpha)   ! d(loss)/d(top derivative)
       loss = loss + tval(NUM_alpha)**2
       call net_backward_point( nt, tw, seed, g )
    end do
  end subroutine one_batch
end program dir_grad_timing
