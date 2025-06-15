!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (rand_module.f90) is part of DNNF90.
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
MODULE rand_module

  use global_variables, only: rand_seed_in

#ifdef _MPI_
  use parallel_module
#endif
  
  implicit none

  PRIVATE
  PUBLIC :: pre_random,random_irange,random_drange,random_gauss,&
       get_map_random

CONTAINS

  SUBROUTINE get_map_random( Nt,map_random,iepoch )
    implicit none
    integer,intent(IN) :: Nt
    integer,intent(IN) :: iepoch   ! evolves a fixed seed between reshuffles
    integer,intent(OUT) :: map_random(Nt)
    integer::seedsize,c
    integer,allocatable::seed(:)
    integer :: i,ierr,ib,i_flag,ibuf
    integer,allocatable :: ista(:)
    logical,allocatable :: seen(:)

    !In fortran90, seed is array.
    ! To get seedsize, use below.
    call random_seed(size=seedsize)

    !Allocate seed array.
    allocate(seed(1:seedsize))
  
    !Get system time. 
    call system_clock(count=c)

    !Substitute "seed" using system time, or use the fixed input seed
    !(Rand_seed key; >0 -> reproducible runs).
    if ( rand_seed_in > 0 ) then
       ! A constant put-seed would make every reshuffle draw the same
       ! permutation, silently turning Shuffle 1 into a no-op under a
       ! fixed Rand_seed; folding the epoch in keeps runs reproducible
       ! while letting successive reshuffles differ.
       seed = rand_seed_in + iepoch
    else
       seed = c
    end if

    !share seed in each rank
#ifdef _MPI_
    allocate(ista(MPI_STATUS_SIZE))
    if (myrank==0) then
       do i=1,nprocs-1
          call mpi_send(seed(1),seedsize,mpi_integer,i,0,mpi_comm_world,ierr)
       end do
    else
       call mpi_recv(seed(1),seedsize,mpi_integer,0,0,mpi_comm_world,ista,ierr)
    end if
#endif

    !Set "seed" to produce random number obey to system time.
    call random_seed(put=seed)

    ! Same draw as before (identical random calls and identical accept
    ! and reject decisions); the duplicate test is a mark lookup instead
    ! of a scan over everything drawn so far, which turned a permutation
    ! of N training points into an O(N^2 log N) operation.
    allocate( seen(Nt) );  seen = .false.
    do ib=1,Nt
       if (ib==1) then
          call random_irange(ibuf,1,Nt)
       else
          i_flag=0
          do while ( i_flag == 0 )
             call random_irange(ibuf,1,Nt)
             i_flag=1
             if ( seen(ibuf) ) i_flag=0
          end do
       end if
       map_random(ib)=ibuf
       seen(ibuf) = .true.
    end do
    deallocate( seen )
    deallocate( seed )
#ifdef _MPI_
    deallocate( ista )
#endif


  END SUBROUTINE get_map_random

  SUBROUTINE pre_random
    implicit none
    integer::seedsize,c
    integer,allocatable::seed(:)
integer :: i,ierr
integer,allocatable :: ista(:)

    !In fortran90, seed is array.
    ! To get seedsize, use below.
    call random_seed(size=seedsize)

    !Allocate seed array.
    allocate(seed(1:seedsize))
  
    !Get system time. 
    call system_clock(count=c)

    !Substitute "seed" using system time, or use the fixed input seed
    !(Rand_seed key; >0 -> reproducible runs).
    if ( rand_seed_in > 0 ) then
       seed = rand_seed_in
    else
       seed = c
    end if

    !share seed in each rank
#ifdef _MPI_
    allocate(ista(MPI_STATUS_SIZE))
    if (myrank==0) then
       do i=1,nprocs-1
          call mpi_send(seed(1),seedsize,mpi_integer,i,0,mpi_comm_world,ierr)
       end do
    else
       call mpi_recv(seed(1),seedsize,mpi_integer,0,0,mpi_comm_world,ista,ierr)
    end if
#endif

    !Set "seed" to produce random number obey to system time.
    call random_seed(put=seed)
  
    RETURN
  END SUBROUTINE pre_random

  SUBROUTINE random_irange(iout,imin,imax)
    integer,intent(in)::imin,imax
    integer,intent(out)::iout
    real(8)::d
    
    call random_number(d)
    d=d*dble(imax-imin+1)+dble(imin)-0.5d0
    iout=floor(d)
    if(d-dble(iout).ge.0.5d0)then
       iout=iout+1
    endif
  END SUBROUTINE random_irange

  SUBROUTINE random_drange(dout,dmin,dmax)
    real(8),intent(in)::dmin,dmax
    real(8),intent(out)::dout
    real(8)::d
    call random_number(d)
    dout=d*(dmax-dmin)+dmin
  END SUBROUTINE random_drange

  !> Normal deviate.  sig is the standard deviation, not the variance:
  !! gout = mu + sig*x with x standard normal.  Every caller passes a
  !! standard deviation (sqrt(1/fan_in) and so on).  Passing a variance
  !! here gives the wrong initialization scale without any diagnostic.
  SUBROUTINE random_gauss(gout,mu,sig)
    real(8),intent(in)::mu,sig
    real(8),intent(out)::gout
    real(8) :: pi,r1,r2,x
    !Box-Muller method
    pi=acos(-1.d0)
    ! random_number returns [0,1), and log(0) would make the deviate
    ! infinite (or NaN once multiplied by a zero cosine), so an exact
    ! zero is redrawn
    r1 = 0.d0
    do while ( r1 <= 0.d0 )
       call random_number(r1)
    end do
    call random_number(r2)
    x = sqrt(-2.d0*log(r1))*cos(2.d0*pi*r2)

    gout = mu + sig*x

  END SUBROUTINE random_gauss


END MODULE rand_module
