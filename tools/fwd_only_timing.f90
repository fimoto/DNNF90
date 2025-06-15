!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (fwd_only_timing.f90) is part of DNNF90.
!
!  DNNF90 is free software released under the MIT License.
!  You should have received a copy of the MIT License (file LICENSE
!  in the root directory of this distribution) along with DNNF90.
!  If not, see <https://opensource.org/licenses/MIT>.
!
! Forward pass only: the time to produce every carried derivative at one
! point, with no weight gradient.  This is the quantity ADOL-C's
! tensor_eval is compared against in the manuscript (Section 4.1), which
! also returns derivative values alone and no weight dependence.  The
! gradient version of the same setting is tools/fwd_grad_timing.f90.
!
!   fwd_only_timing.out <D0> <K> <width> <npoints> <repeat>
!
! Prints the accumulated value sum (evidence that the run is a real
! computation), the total time, and the time per point.
!
program fwd_only_timing
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha
  use net_module, only: net_t, work_t, net_init, work_init, work_free, net_free
  use train_module, only: twork_t, twork_init, twork_free, net_forward_point
  implicit none
  type(net_t) :: nt
  type(work_t) :: wk
  type(twork_t) :: tw
  real(8),allocatable :: x(:,:), tval(:)
  real(8) :: t0, t1, acc
  integer :: d0, kmax, width, npt, nrep, dims(4)
  integer,allocatable :: sd(:,:)
  integer :: n, r
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
  allocate( x(d0,npt), tval(NUM_alpha) )

  call random_seed()
  call random_number( x );  x = 2.d0*x - 1.d0
  call random_number( nt%w );  nt%w = 0.3d0*(nt%w - 0.5d0)

  acc = 0.d0
  do n = 1, npt
     call net_forward_point( nt, tw, x(:,n), tval )
     acc = acc + sum( tval**2 )
  end do
  write(*,'(a,i0,a,i0,a,i0)') '### D0=', d0, '  K=', kmax, &
       '  derivatives=', NUM_alpha
  write(*,'(a,es22.12)') '    computed value sum =', acc

  call cpu_time(t0)
  do r = 1, nrep
     do n = 1, npt
        call net_forward_point( nt, tw, x(:,n), tval )
        acc = acc + tval(1)
     end do
  end do
  call cpu_time(t1)
  write(*,'(a,f12.4,a)') ' forward, all points  ', &
       1.d3*(t1-t0)/dble(nrep), ' ms'
  write(*,'(a,f12.4,a)') ' forward, per point   ', &
       1.d3*(t1-t0)/dble(nrep)/dble(npt), ' ms'

  call twork_free(tw); call work_free(wk); call net_free(nt)
end program fwd_only_timing
