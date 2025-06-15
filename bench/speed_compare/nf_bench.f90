program nf_bench
  use nf, only: dense, input, network, sgd, tanhf
  implicit none
  type(network) :: net
  real, allocatable :: x(:,:), y(:,:)
  integer :: i, ic0, ic1, cr, nep
  character(16) :: arg
  call get_command_argument(1, arg); read(arg,*) nep
  allocate( x(1,190), y(1,190) )
  do i=1,190
     x(1,i) = -1.0 + 2.0*real(i-1)/189.0
     y(1,i) = x(1,i)**2 + 1.0
  end do
  net = network([ input(1), dense(768, tanhf()), dense(768, tanhf()), &
                  dense(768, tanhf()), dense(768, tanhf()), dense(768, tanhf()), &
                  dense(1) ])
  call system_clock(ic0,cr)
  call net % train( x, y, batch_size=190, epochs=nep, optimizer=sgd(learning_rate=0.1) )
  call system_clock(ic1)
  write(*,"(a,i4,a,f8.3,a)") "nf epochs=",nep,"  time=",dble(ic1-ic0)/cr," s"
end program
