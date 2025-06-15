! One weight gradient of the same loss the framework comparison times.
!
! bench/post/compare_frameworks.py times one gradient of a loss built
! from derivatives of the network with respect to its inputs, over a
! fixed batch.  This is the same quantity for the library, so that the
! two can be put in one table.  It is NOT an epoch: an epoch covers all
! training points in batches, and comparing an epoch against a single
! framework gradient overstates the framework by the number of batches.
!
!   fwd_grad_timing.out <D0> <K> <width> <npoints> <repeat>
!
! The loss is the sum over the batch of the squares of every carried
! derivative, whose seed is 2*T, matching the hod setting of the script.
program fwd_grad_timing
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha
  use net_module, only: net_t, net_init, net_free, work_t, work_init, &
       work_free
  use train_module, only: twork_t, grad_t, twork_init, grad_init, &
       grad_zero, net_forward_point, net_backward_point, twork_free, &
       grad_free
#ifdef PHASE_TIMING
  use train_module, only: phase_t_gemm, phase_t_bell, phase_reset
#endif
  implicit none
  type(net_t) :: nt
  type(work_t) :: wk
  type(twork_t) :: tw
  type(grad_t) :: g
  real(8),allocatable :: x(:,:), tval(:), seed(:)
  real(8) :: t0, t1, loss
  integer :: d0, kmax, width, npt, nrep, dims(4)
  integer :: seeds(1,1)
  integer,allocatable :: sd(:,:)
  integer :: i, n, r
  character(32) :: a

  call get_command_argument(1,a); read(a,*) d0
  call get_command_argument(2,a); read(a,*) kmax
  call get_command_argument(3,a); read(a,*) width
  call get_command_argument(4,a); read(a,*) npt
  call get_command_argument(5,a); read(a,*) nrep

  allocate( sd(d0,1) );  sd = 0
  call init_hod_tables( d0, kmax, 0, sd )
  dims = (/ d0, width, width, 1 /)
  call net_init( nt, 4, dims )
  call work_init( wk, nt )
  call twork_init( tw, nt )
  call grad_init( g, nt )
  allocate( x(d0,npt), tval(NUM_alpha), seed(NUM_alpha) )

  call random_seed()
  call random_number( x );  x = 2.d0*x - 1.d0
  call random_number( nt%w );  nt%w = 0.3d0*(nt%w - 0.5d0)

  ! warm up; the accumulated loss is printed as evidence that the run
  ! is a real computation on a live network
  loss = 0.d0
  call grad_zero( g )
  do n = 1, npt
     call net_forward_point( nt, tw, x(:,n), tval )
     loss = loss + sum( tval**2 )
     seed = 2.d0*tval
     call net_backward_point( nt, tw, seed, g )
  end do
  write(*,'(a,es22.12)') '    computed loss =', loss

  call cpu_time(t0)
#ifdef PHASE_TIMING
  call phase_reset()
#endif
  do r = 1, nrep
     call grad_zero( g )
     do n = 1, npt
        call net_forward_point( nt, tw, x(:,n), tval )
        seed = 2.d0*tval
        call net_backward_point( nt, tw, seed, g )
     end do
  end do
  call cpu_time(t1)

  write(*,'(a,i0,a,i0,a,i0,a,i0)') " D0=", d0, " K=", kmax, &
       " derivatives=", NUM_alpha, " points=", npt
  write(*,'(a,f12.4)') " library            ", 1.d3*(t1-t0)/dble(nrep)
#ifdef PHASE_TIMING
  ! How the cost divides between the linear contraction (one GEMM per
  ! layer, maps onto cuBLAS directly) and the Bell-table part (an
  ! irregular gather, which does not).  Which dominates decides whether
  ! a CUDA port is worth the work.
  write(*,'(a,f12.4,a,f6.1,a)') " gemm phase         ", &
       1.d3*phase_t_gemm/dble(nrep), "   (", &
       1.d2*phase_t_gemm/max(phase_t_gemm+phase_t_bell,1.d-30), " %)"
  write(*,'(a,f12.4,a,f6.1,a)') " bell-table phase   ", &
       1.d3*phase_t_bell/dble(nrep), "   (", &
       1.d2*phase_t_bell/max(phase_t_gemm+phase_t_bell,1.d-30), " %)"
#endif

  call grad_free( g );  call twork_free( tw );  call work_free( wk )
  call net_free( nt )
  deallocate( x, tval, seed, sd )
end program fwd_grad_timing
