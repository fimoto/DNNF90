!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (convert_weight.f90) is part of DNNF90.
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
! -----------------------------------------------------------------------
!add one new layer
program convert_weight
  implicit none
  integer,parameter :: unit_in=2,unit_conv=10,uw=11
  integer :: i,j,k,ierr,itmp
  character(15) :: cbuf
  integer :: Nlayer,ndim_max
  integer,allocatable :: ndim(:)
  real(8),allocatable :: weight(:,:,:)
!
  integer :: Nlayer_new,ndim_max_new,i_step,i_func,i_act
  integer,allocatable :: ndim_new(:)
  real(8),allocatable :: weight_new(:,:,:)
!
  integer :: Nadd !# of new additonal diagonal layers

  open(unit_conv, file='convert_nn.dat',status='old')
  read(unit_conv,*) cbuf,Nlayer_new
  allocate( ndim_new(Nlayer_new) )
  do i=1,Nlayer_new
     read(unit_conv,*) ndim_new(i)
  end do
  close(unit_conv)
  ndim_max_new = maxval( ndim_new )
  allocate( weight_new(Nlayer_new,ndim_max_new,0:ndim_max_new) ); weight_new=0.d0

  open(unit_in, file='nn_weight.dat',status='old')
  read(unit_in,*) i_step
  read(unit_in,*) cbuf,i_func
  read(unit_in,*) cbuf,i_act
  read(unit_in,*) Nlayer
  close(unit_in)
    
  allocate( ndim(Nlayer) )

  open(unit_in, file='nn_weight.dat',status='old')
  read(unit_in,'()') !i_step
  read(unit_in,'()') !i_func
  read(unit_in,'()') !Activation_out
  read(unit_in,'()') !Nlayer
  do i=1,Nlayer
     read(unit_in,*) ndim(i)
  end do
  close(unit_in)

  ndim_max=maxval(ndim)
  allocate( weight(Nlayer,ndim_max,0:ndim_max) ); weight=0.d0
  write(*,*) "Nlayer=",NLayer
  write(*,*) "ndim_max=",ndim_max
  write(*,*) "weight allocated"

  open(unit_in, file='nn_weight.dat',status='old')
  read(unit_in,'()') !i_step
  read(unit_in,'()') !i_func
  read(unit_in,'()') !Activation_out
  read(unit_in,'()') !Nlayer
  do i=1,Nlayer
     read(unit_in,'()') !ndim
  end do
  do i=2,Nlayer
     read(unit_in,'()')
     do j=1,ndim(i)
        itmp=ndim(i-1)
        read(unit_in,*) weight(i,j,0:itmp)
     end do
  end do
  close(unit_in)
  
  write(*,*) "weight has been read"


! weight -> weight_new
  do i=2,Nlayer-1
     do j=1,ndim(i)
        itmp=ndim(i-1)
        weight_new(i,j,0:itmp) = weight(i,j,0:itmp)
     end do
  end do

! inset diagonal additional layers
  Nadd = Nlayer_new - Nlayer
  do i=Nlayer,Nlayer+Nadd-1
     do j=1,ndim(Nlayer-1)
        weight_new(i,j,j) = 1.d0
     end do
  end do

  weight_new(Nlayer_new,1:ndim_max,0:ndim_max) = weight(Nlayer,1:ndim_max,0:ndim_max)

  open(uw,file="weight_new.dat",status="replace")
  write(uw,'(i0)') i_step
  write(uw,'(a4,1x,i0)') "func",i_func
  write(uw,'(a14,1x,i0)') "Activation_out",i_act
  write(uw,'(i0)') Nlayer_new
  do i=1,Nlayer_new
     write(uw,'(i0)') ndim_new(i)
  end do

  do i=2,Nlayer_new
     write(uw,*) "#l=",i
     do j=1,ndim_new(i)
        write(uw,'(100f20.10)') weight_new(i,j,0:ndim_new(i-1))
     end do
  end do

  close(uw)



end program convert_weight
