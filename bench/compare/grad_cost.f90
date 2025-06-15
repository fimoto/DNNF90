! This file (bench/compare/grad_cost.f90) is part of DNNF90.
! (MIT License; see LICENSE at the repository root.)
!
! The library side of the cost comparison in bench/compare/grad_cost.py.
! It times exactly the same quantity: one gradient of a loss with respect
! to every weight, over one batch of NPT points, with the loss built from
! derivatives of the network with respect to its inputs.
!
!   ./grad_cost.out kdv     D0 = 2, K = 3, 2-16-16-1, the KdV residual
!   ./grad_cost.out hod7    D0 = 4, K = 7, 4-8-8-1, every mixed partial
!
! The kernels called here are the ones the trainer calls; there is no
! separate implementation for the benchmark.
program grad_cost
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, alpha_deg
  use net_module, only: net_t, net_init, net_free
  use train_module, only: twork_t, twork_init, twork_free, grad_t, &
       grad_init, grad_free, grad_zero, net_forward_point, &
       net_backward_point
  implicit none
  integer,parameter :: NPT = 20, NREP = 200
  type(net_t) :: nt
  type(twork_t) :: tw
  type(grad_t) :: g
  real(8),allocatable :: tval(:), seed(:), x(:,:)
  real(8) :: t0, t1, r
  integer :: seeds4(4,3), dims(4), d0, kmax, i, n, ia
  character(16) :: which

  call get_command_argument(1, which)

  if ( trim(which) == "kdv" ) then
     d0 = 2;  kmax = 3;  dims = (/ 2, 16, 16, 1 /)
     ! the residual needs u, u_x, u_t and u_xxx: their closure
     seeds4 = 0
     seeds4(1,1) = 1                     ! u_x
     seeds4(2,2) = 1                     ! u_t
     seeds4(1,3) = 3                     ! u_xxx
     call init_hod_tables( d0, kmax, 3, seeds4(1:d0,1:3) )
  else
     d0 = 4;  kmax = 7;  dims = (/ 4, 8, 8, 1 /)
     seeds4 = 0
     call init_hod_tables( d0, kmax, 0, seeds4(1:d0,1:1) )
  end if

  call net_init( nt, 4, dims )
  call twork_init( tw, nt )
  call grad_init( g, nt )
  allocate( tval(NUM_alpha), seed(NUM_alpha), x(d0,NPT) )

  call random_seed()
  call random_number( x )
  x = 2.d0*x - 1.d0
  call random_number( nt%w );  nt%w = 0.3d0*(nt%w - 0.5d0)

  ! one gradient evaluation over the batch, repeated
  call one_batch()
  call cpu_time(t0)
  do i = 1, NREP
     call one_batch()
  end do
  call cpu_time(t1)

  write(*,'(a,a,a,i4,a,f9.4,a,i0,a)') "dnnf90   ", trim(which), &
       "    ", NUM_alpha, " derivatives  ", &
       1.d3*(t1-t0)/dble(NREP), " ms per gradient over ", NPT, " points"

  deallocate( tval, seed, x )
  call grad_free( g );  call twork_free( tw );  call net_free( nt )

CONTAINS

  SUBROUTINE one_batch()
    implicit none
    integer :: n
    call grad_zero( g )
    do n = 1, NPT
       call net_forward_point( nt, tw, x(1:d0,n), tval )
       if ( trim(which) == "kdv" ) then
          ! residual r = u_t + 3 u u_x + u_xxx, seed dL/dT = r * dr/dT.
          ! The closure of {u_x, u_t, u_xxx} is carried in degree order,
          ! so the five slots are 1:(0,0) 2:(1,0) 3:(0,1) 4:(2,0) 5:(3,0):
          ! u_t is slot 3 and u_xxx is slot 5.
          r = tval(3) + 3.d0*tval(1)*tval(2) + tval(5)
          seed = 0.d0
          seed(1) = r*3.d0*tval(2)
          seed(2) = r*3.d0*tval(1)
          seed(3) = r
          seed(5) = r
       else
          ! fit every carried derivative to zero: dL/dT = T
          seed = tval
       end if
       call net_backward_point( nt, tw, seed, g )
    end do
  END SUBROUTINE one_batch

end program grad_cost
