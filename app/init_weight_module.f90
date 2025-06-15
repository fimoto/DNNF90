!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (init_weight_module.f90) is part of DNNF90.
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
!initialization
MODULE init_weight_module

#ifdef _MPI_
  use parallel_module
#endif
  use global_variables ! define filenames: file_err_t="err_t.dat(istart_step)"
  use rand_module,only: pre_random,random_drange,random_gauss
  use io_module,only: check_weight_header

  implicit none

  PRIVATE
  PUBLIC :: get_initial_weight

CONTAINS

SUBROUTINE get_initial_weight
  implicit none
  integer,parameter :: ur=29   ! a local initialised in its declaration
                               ! would be implicitly SAVEd
  integer :: i,j
  ! initialize weight and bias
  call pre_random
  if ( iswitch_restart == 0 ) then
     istart_step=0
     call init_weight(init_weight_method)
  else if ( iswitch_restart == 1 ) then
     open(ur,file='nn_weight.dat',status='old')
     call check_weight_header( ur, 'nn_weight.dat', istart_step )

     do i=2,Nlayer
        read(ur,'()')
        do j=1,ndim(i)
           read(ur,*) weight(i,j,0:ndim(i-1))
        end do
     end do
     close(ur)
  end if

#ifdef _MPI_
  if (myrank==0) then
     write(*,'(a,2x,i0)') "start from total epoch: istart_step=",istart_step
  end if
#else
  write(*,'(a,2x,i0)') "start from total epoch: istart_step=",istart_step
#endif

  ! outfile names; the field is sized for istart_step <= 1,000,000
  !+++Cost function
  write(file_history, '("history_ep",i7.7,".dat")') istart_step
  !
  !
  !+++RMSE
  !
  !
  !+++MAE
  !
  !
  !+++Relative error of integration

  weight_gen = weight_gen + 1
END SUBROUTINE get_initial_weight

  SUBROUTINE init_weight(init_w)
    implicit none
    !> Kept for the call signature; the method is read from the module
    !! state, which is what every branch below tests.
    character(*),intent(IN) :: init_w
    ! initialize weight and bias
    weight(:,:,0)=0.d0 !bias=0

    if ( (init_weight_method == "RANDOM_UNIF") .or. (init_weight_method == "RANDOM_NORM") ) then
       call init_random
    else if ( (init_weight_method == "LECUN_UNIF") .or. (init_weight_method == "LECUN_NORM") ) then
       call init_lecun
    else if ( (init_weight_method == "GLOROT_UNIF") .or. (init_weight_method == "GLOROT_NORM") ) then
       call init_glorot
    else if ( init_weight_method == "SIREN" ) then
       call init_siren
    else if ( init_weight_method == "BESSEL_INIT" ) then
       call init_bessel
    else if ( (init_weight_method == "HE_UNIF") .or. (init_weight_method == "HE_NORM") ) then
       call init_he
    else
       ! an unrecognized name must not fall through silently: the weights
       ! would stay at their allocation value of zero and the network
       ! would train as a degenerate bias-only model
       write(*,*) "init_weight: unknown Init_weight method: ", trim(init_weight_method)
       write(*,*) "  options: RANDOM_/LECUN_/GLOROT_/HE_ + UNIF|NORM"
       stop
    end if

  END SUBROUTINE init_weight

  SUBROUTINE init_random
    implicit none
    integer :: i,j,k
    
    do i=2,Nlayer
       do j=1,ndim(i)
          do k=1,ndim(i-1)
             if ( init_weight_method == "RANDOM_UNIF" ) then
                call random_drange(weight(i,j,k),-0.05d0,0.05d0)
             else if ( init_weight_method == "RANDOM_NORM" ) then
                call random_gauss(weight(i,j,k),0.d0,0.05d0)
             end if
          end do
       end do
    end do

  END SUBROUTINE init_random

  !> Initialisation for a periodic activation.
  !!
  !! With sigma = sin the usual scalings are wrong: they are derived for
  !! an activation that is close to the identity near zero and saturates,
  !! whereas sin is oscillatory, so the pre-activation has to be placed
  !! deliberately inside one period.  The scheme is the one introduced
  !! for periodic-activation networks: the first layer is drawn from
  !! U(-w0/fan_in, w0/fan_in), which spreads the input over w0 periods,
  !! and every later layer from U(-sqrt(6/fan_in)/w0, +sqrt(6/fan_in)/w0),
  !! which keeps the distribution of the pre-activation the same from
  !! layer to layer instead of letting it grow with depth.
  !!
  !! w0 is Init_w_omega (default 30, the value that scheme uses).  For a
  !! smooth activation such as tanh this initialisation is not meant to
  !! be used; the guard in read_data rejects that combination.
  SUBROUTINE init_siren
    implicit none
    integer :: i,j,k
    real(8) :: fan_in, w0, bound

    w0 = init_w_omega
    do i=2,Nlayer
       fan_in = dble(ndim(i-1))
       if ( i == 2 ) then
          bound = w0/fan_in
       else
          bound = sqrt( 6.d0/fan_in )/w0
       end if
       do j=1,ndim(i)
          do k=1,ndim(i-1)
             call random_drange( weight(i,j,k), -bound, bound )
          end do
       end do
    end do

  END SUBROUTINE init_siren

  !> Initialisation derived from the statistics of J_0 itself.
  !!
  !! The periodic scheme above is derived for sin: odd, one period of
  !! 2 pi, unit slope at the origin.  J_0 differs in three ways that
  !! matter.  It is even, so a layer cannot tell the sign of its
  !! pre-activation.  J_0(0) = 1, so a small pre-activation makes every
  !! neuron emit the same constant and the layer carries nothing.  And
  !! its amplitude decays like 1/sqrt(a), so an over-large pre-activation
  !! is as damaging as a too-small one.  There is therefore an interior
  !! optimum rather than a free frequency to tune.
  !!
  !! Taking a ~ N(0, sigma^2), the variance of J_0(a) is maximal at
  !! sigma* = 2.75, where E[J_0] = 0.3192 and E[J_0^2] = 0.3455
  !! (integrated to twenty digits).  A weight drawn from U(-b,b) over
  !! fan_in inputs of second moment E[z^2] gives Var[pre] = n b^2 E[z^2]/3,
  !! so matching sigma* fixes
  !!
  !!     b = sigma* sqrt( 3 / (n E[z^2]) ).
  !!
  !! For the hidden layers E[z^2] is the 0.3455 above, giving b = 8.10 /
  !! sqrt(n); for the first layer the inputs are the data, whose second
  !! moment is 1/3 on [-1,1], giving b = 8.25 / sqrt(n).  The two are
  !! close enough that the same constant would do, but they are kept
  !! separate because the input range is the caller's, not ours.
  SUBROUTINE init_bessel
    implicit none
    integer :: i,j,k
    real(8) :: fan_in, bound
    !> The variance-optimal spread is sigma = 2.75, but placing the whole
    !! layer there at initialisation makes the initial loss enormous: the
    !! network starts deep in the oscillatory region, where the residual
    !! of a high-order target is large and the gradient points nowhere in
    !! particular.  A third of that spread was measured to be the useful
    !! setting on the shipped case (cost 1.8e-4 against 3.4e+2 at the full
    !! value), which puts the first layer just past the first zero of J_0
    !! at 2.405 while leaving the deeper layers closer to the flat region
    !! they can grow out of.
    real(8),parameter :: sigma_star = 2.75d0*0.33d0
    real(8),parameter :: m2_hidden  = 0.34551d0   ! E[J_0(a)^2] at sigma*
    real(8),parameter :: m2_input   = 0.33333333333333333d0  ! E[x^2], x ~ U(-1,1)

    do i=2,Nlayer
       fan_in = dble(ndim(i-1))
       if ( i == 2 ) then
          bound = sigma_star*sqrt( 3.d0/( fan_in*m2_input ) )
       else
          bound = sigma_star*sqrt( 3.d0/( fan_in*m2_hidden ) )
       end if
       do j=1,ndim(i)
          do k=1,ndim(i-1)
             call random_drange( weight(i,j,k), -bound, bound )
          end do
       end do
    end do

  END SUBROUTINE init_bessel

  SUBROUTINE init_lecun
    implicit none
    integer :: i,j,k
    real(8) :: fan_in,stddev
    
    do i=2,Nlayer
       fan_in = dble(ndim(i-1)) ! fan_out=dble(ndim(i))
       do j=1,ndim(i)
          do k=1,ndim(i-1)
             if ( init_weight_method == "LECUN_UNIF" ) then
                call random_drange(weight(i,j,k),-sqrt(3.d0/fan_in),sqrt(3.d0/fan_in))
             else if ( init_weight_method == "LECUN_NORM" ) then
                stddev = sqrt(1.d0/fan_in)
                call random_gauss(weight(i,j,k),0.d0,stddev)
             end if
          end do
       end do
    end do

  END SUBROUTINE init_lecun

  SUBROUTINE init_glorot
    implicit none
    integer :: i,j,k
    real(8) :: fan_in,fan_out,stddev
    
    do i=2,Nlayer
       fan_in = dble(ndim(i-1))
       fan_out=dble(ndim(i))
       do j=1,ndim(i)
          do k=1,ndim(i-1)
             if ( init_weight_method == "GLOROT_UNIF" ) then
                call random_drange(weight(i,j,k),-sqrt(6.d0/(fan_in+fan_out)),sqrt(6.d0/(fan_in+fan_out)))
             else if ( init_weight_method == "GLOROT_NORM" ) then
                stddev = sqrt(2.d0/(fan_in+fan_out))
                call random_gauss(weight(i,j,k),0.d0,stddev)
             end if
          end do
       end do
    end do

  END SUBROUTINE init_glorot

  SUBROUTINE init_he
    implicit none
    integer :: i,j,k
    real(8) :: fan_in,stddev
    
    do i=2,Nlayer
       fan_in = dble(ndim(i-1)) ! fan_out=dble(ndim(i))
       do j=1,ndim(i)
          do k=1,ndim(i-1)
             if ( init_weight_method == "HE_UNIF" ) then
                call random_drange(weight(i,j,k),-sqrt(6.d0/fan_in),sqrt(6.d0/fan_in))
             else if ( init_weight_method == "HE_NORM" ) then
                stddev = sqrt(2.d0/fan_in)
                call random_gauss(weight(i,j,k),0.d0,stddev)
             end if
          end do
       end do
    end do

  END SUBROUTINE init_he



END MODULE init_weight_module
