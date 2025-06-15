!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (net_module.f90) is part of DNNF90.
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
! Instance based evaluation of a trained network and of all carried mixed
! derivatives.
!
! Why this module exists.  The trainer keeps the network and its
! intermediate arrays in module variables, which means one process holds
! exactly one network and no two threads may evaluate at the same time.
! A machine learning force field needs the opposite: one network per
! element species, all resident at once, evaluated concurrently over the
! atoms of a structure.  Here the network lives in a net_t and every
! mutable intermediate lives in a work_t, so
!
!   * an arbitrary number of networks can coexist in one process, and
!   * one work_t per thread makes evaluation thread safe.
!
! The multi-index and Bell tables in multi_index_bell_module stay shared.
! They are written once by init_hod_tables and are read only afterwards,
! which is exactly the data that should be shared rather than duplicated.
!
! This module deliberately does not use global_variables.
! -----------------------------------------------------------------------
MODULE net_module

#ifdef USE_BLAS
  use blas_wrap_module, only: bgemm
#endif
  use multi_index_bell_module, only: tabset_t, tabset_from_current, tabset_free, tanh_derivs_ts, act_derivs_ts

  implicit none

  PRIVATE
  PUBLIC :: net_t, work_t
  PUBLIC :: net_init, net_load, net_save, net_free
  PUBLIC :: work_init, work_free
  PUBLIC :: net_eval_hod, net_eval_hod_multi

  !> One complete, independent network (for example one element species).
  TYPE :: net_t
     integer :: nlayer = 0
     integer :: ndmax  = 0
     integer,allocatable :: ndim(:)  ! (nlayer) width of each layer
     real(8),allocatable :: w(:,:,:)  ! (nlayer, ndmax, 0:ndmax)
     type(tabset_t) :: tab               ! this network's own table set
  END TYPE net_t

  !> Mutable scratch for one evaluation.  Use one instance per thread.
  !! All per-call scratch lives here as well: the kernels then use no
  !! large automatic arrays, so they neither overflow the stack under
  !! -fstack-arrays or -Ofast builds (the weight-slice scratch is
  !! quadratic in the width) nor call the allocator per point.
  !! The components are ALLOCATABLE, not POINTER, on purpose: the
  !! kernels read them through the derived type inside their hot loops,
  !! and the no-alias, contiguous guarantee of allocatable is worth
  !! about 30 per cent of the whole gradient at -O3 (measured; pointer
  !! components force the compiler to assume any two of them may
  !! overlap).  Nothing here is ever pointer-associated.
  TYPE :: work_t
     real(8),allocatable :: T(:,:,:)  ! (0:ndmax, na, 2) rolling planes
     !> Copy of the final plane for every output component, filled by
     !! net_eval_hod so that net_eval_hod_multi can read it after the
     !! rolling planes have been reused.
     real(8),allocatable :: Tout(:,:)
     real(8),allocatable :: S(:,:)  ! (ndmax, na) one layer
     real(8),allocatable :: wl(:,:)  ! (ndmax, 0:ndmax) weight slice
     ! neuron-axis vectorized scratch: j is the fast dimension
     real(8),allocatable :: bqv(:,:,:)  ! (ndmax, na, kmax)
     real(8),allocatable :: dt(:,:)  ! (ndmax, 0:kmax+1)
     real(8),allocatable :: tv(:)
     real(8),allocatable :: ttv(:)
     real(8),allocatable :: bsv(:)
  END TYPE work_t

CONTAINS

  !> Build an empty network of the given shape.
  SUBROUTINE net_init( nt, nlayer, ndim, ts )
    implicit none
    type(net_t),intent(OUT) :: nt
    integer,intent(IN) :: nlayer, ndim(nlayer)
    type(tabset_t),intent(IN),optional :: ts
    if ( present(ts) ) then
       ! tabset_t carries allocatable components, so this is a deep
       ! copy: every network owns its own table set, there is no
       ! ownership flag, and the caller may free ts at any time.  The
       ! tables are index lists and small coefficients: 0.4 MB at the
       ! settings of the paper (d0=4, K=7), 0.6 MB at sixty descriptors
       ! with K=2, 2.5 MB at a hundred (measured), so the copy per
       ! committee member is cheap at any realistic size.
       nt%tab = ts
    else
       call tabset_from_current( nt%tab )
    end if
    ! Contract checks.  The input width must equal the number of variables
    ! the table set differentiates with respect to: the forward pass seeds
    ! T(j, ind_e1(j)) for j = 1..ndim(1), and ind_e1 has length d0, so a
    ! wider input layer reads past it and then writes T at whatever index
    ! that garbage happens to be.  Under bounds checking this is caught at
    ! once, but an optimized build corrupts memory and still returns a
    ! plausible number, so the mismatch is rejected here instead.
    if ( nlayer < 2 ) then
       write(*,*) "net_init: nlayer must be at least 2, got", nlayer
       stop
    end if
    if ( ndim(1) /= nt%tab%d0 ) then
       write(*,*) "net_init: input width", ndim(1), &
                  " does not match the table set d0 =", nt%tab%d0
       write(*,*) "  build the tables with the same number of input", &
                  " variables as the network takes"
       stop
    end if
    ! Several outputs are allowed.  The recursion runs over all neurons of
    ! every layer including the last, so nothing in it distinguishes the
    ! output width; net_eval_hod returns the first component and
    ! net_eval_hod_multi returns all of them.  A vector field is then one
    ! network with a shared hidden representation rather than one network
    ! per component.
    if ( ndim(nlayer) < 1 ) then
       write(*,*) "net_init: output width", ndim(nlayer), " must be >= 1"
       stop
    end if
    nt%nlayer = nlayer
    nt%ndmax  = maxval( ndim(1:nlayer) )
    allocate( nt%ndim(nlayer) )
    nt%ndim(1:nlayer) = ndim(1:nlayer)
    allocate( nt%w(nlayer,nt%ndmax,0:nt%ndmax) )
    nt%w = 0.d0
  END SUBROUTINE net_init

  !> Read a trained network from a file in nn_weight.dat format.
  SUBROUTINE net_load( nt, filename, ts )
    implicit none
    type(net_t),intent(OUT) :: nt
    character(len=*),intent(IN) :: filename
    type(tabset_t),intent(IN),optional :: ts
    integer :: ur, i, j, nlayer
    integer,allocatable :: nd(:)
    character(len=32) :: ckey
    integer :: ival, ifunc
    integer :: ios
    logical :: ex
    ur = 87
    ! A missing or unreadable weight file is an ordinary mistake for a
    ! host to make, so it is reported rather than left to the runtime.
    inquire( file=trim(filename), exist=ex )
    if ( .not. ex ) then
       write(*,*) "net_load: no such file: ", trim(filename)
       stop
    end if
    open( ur, file=trim(filename), status='old', iostat=ios )
    if ( ios /= 0 ) then
       write(*,*) "net_load: cannot open ", trim(filename), " (iostat=", ios, ")"
       stop
    end if
    read(ur,'()')                      ! epoch
    ! The library kernels implement the tanh activation with an identity
    ! output layer (the HOD contract).  A weight file trained with any
    ! other setting must not load silently into the wrong analytics.
    read(ur,*) ckey, ifunc             ! func: the activation it was trained with
    if ( ifunc < 0 .or. ifunc > 4 ) then
       write(*,*) "net_load: ", trim(filename), " names an unknown", &
                  " activation (func=", ifunc, ")"
       stop
    end if
    read(ur,*) ckey, ival              ! Activation_out
    if ( ival /= 0 ) then
       write(*,*) "net_load: ", trim(filename), " has a nonlinear output", &
                  " layer (Activation_out=", ival, ")"
       write(*,*) "  the HOD contract requires an identity output"
       stop
    end if
    read(ur,*) nlayer
    allocate( nd(nlayer) )
    do i=1,nlayer
       read(ur,*) nd(i)
    end do
    if ( present(ts) ) then
       call net_init( nt, nlayer, nd, ts )
    else
       call net_init( nt, nlayer, nd )
    end if
    ! The analytics that will evaluate this network must be the ones it
    ! was trained with; evaluating a J_0 network as a tanh one is silent
    ! and wrong, so it is refused here.
    if ( nt%tab%iact /= ifunc ) then
       write(*,*) "net_load: ", trim(filename), " was trained with", &
                  " activation code", ifunc, " but the table set carries", &
                  nt%tab%iact
       write(*,*) "  codes: 0 TANH, 1 SIN, 2 ERF, 3 BESSEL, 4 BESSEL1"
       stop
    end if
    do i=2,nlayer
       read(ur,'()')                   ! "#l=" header
       do j=1,nd(i)
          read(ur,*) nt%w(i,j,0:nd(i-1))
       end do
    end do
    close(ur)
    deallocate( nd )
  END SUBROUTINE net_load

  !> Write a network in nn_weight.dat format (readable by net_load and
  !! by the standalone trainer with Restart 1).
  SUBROUTINE net_save( nt, filename )
    implicit none
    type(net_t),intent(IN) :: nt
    character(len=*),intent(IN) :: filename
    integer :: uw, i, j
    uw = 88
    open( uw, file=trim(filename), status='replace' )
    write(uw,'(i0)') 0
    ! Record the activation this network actually carries, so that a
    ! reader cannot evaluate the file with different analytics; net_load
    ! refuses a file whose code does not match its table set.
    write(uw,'(a,1x,i0)') "func", nt%tab%iact
    write(uw,'(a)') "Activation_out 0"
    write(uw,'(i0)') nt%nlayer
    do i=1,nt%nlayer
       write(uw,'(i0)') nt%ndim(i)
    end do
    do i=2,nt%nlayer
       write(uw,*) "#l=",i
       do j=1,nt%ndim(i)
          write(uw,'(100es25.16e3)') nt%w(i,j,0:nt%ndim(i-1))
       end do
    end do
    close(uw)
  END SUBROUTINE net_save

  SUBROUTINE net_free( nt )
    implicit none
    type(net_t),intent(INOUT) :: nt
    if ( allocated(  nt%ndim) ) deallocate( nt%ndim )
    if ( allocated(  nt%w) )    deallocate( nt%w )
    ! The table set is the larger part of a network at force-field sizes,
    ! so a network that built its own releases it here.  One that was
    ! handed a table set only aliases it: releasing that would free the
    ! same arrays once per member of a committee.
    call tabset_free( nt%tab )
    nt%nlayer = 0
    nt%ndmax  = 0
  END SUBROUTINE net_free

  !> Allocate scratch matching a network.  One per thread.
  SUBROUTINE work_init( wk, nt )
    implicit none
    type(work_t),intent(OUT) :: wk
    type(net_t),intent(IN) :: nt
    allocate( wk%T(0:nt%ndmax,nt%tab%na,2) )
    allocate( wk%Tout(nt%ndim(nt%nlayer),nt%tab%na) );  wk%Tout = 0.d0
    allocate( wk%S(nt%ndmax,nt%tab%na) )
    allocate( wk%wl(nt%ndmax,0:nt%ndmax) )
    allocate( wk%bqv(nt%ndmax,nt%tab%na,max(nt%tab%kmax,1)) )
    allocate( wk%dt(nt%ndmax,0:nt%tab%kmax+1) )
    allocate( wk%tv(nt%ndmax), wk%ttv(nt%ndmax), wk%bsv(nt%ndmax) )
    wk%T = 0.d0
    wk%S = 0.d0
    wk%T(0,1,1:2) = 1.d0     ! constant bias channel, never rewritten
  END SUBROUTINE work_init

  SUBROUTINE work_free( wk )
    implicit none
    type(work_t),intent(INOUT) :: wk
    if ( allocated(  wk%T) )  deallocate( wk%T )
    if ( allocated(  wk%Tout) )  deallocate( wk%Tout )
    if ( allocated(  wk%S) )  deallocate( wk%S )
    if ( allocated(  wk%wl) ) deallocate( wk%wl, wk%bqv, wk%dt, wk%tv, wk%ttv, wk%bsv )
  END SUBROUTINE work_free

  !> Evaluate the network and all carried mixed derivatives at x.
  !!
  !! On return t(1:nt%tab%na) holds the derivatives in the canonical
  !! order, with t(1) the value itself.  For a force field the entries
  !! with |alpha|=1 are the derivatives of the energy with respect to the
  !! descriptor components, which is what the chain rule to the atomic
  !! forces needs.
  !!
  !! Reads only nt, the shared read-only tables, and its own wk, so it is
  !! safe to call concurrently with one wk per thread.
  SUBROUTINE net_eval_hod( nt, wk, x, t )
    implicit none
    type(net_t),intent(IN) :: nt
    type(work_t),intent(INOUT) :: wk
    real(8),intent(IN)  :: x(:)
    real(8),intent(OUT) :: t(:)
    integer :: l,j,i,ia,q,it,p,nd,ndm,jj
    integer :: pc,pm                       ! rolling planes
    real(8) :: dsc(0:nt%tab%kmax+1)

    ! ----- input layer -----
    pm = 1
    do j=1,nt%ndim(1)
       wk%T(j,1,pm) = x(j)
       wk%T(j,2:nt%tab%na,pm) = 0.d0
       if ( nt%tab%ind_e1(j) > 0 ) wk%T(j,nt%tab%ind_e1(j),pm) = 1.d0
    end do

    do l=2,nt%nlayer
       pc = 3 - pm
       nd  = nt%ndim(l)
       ndm = nt%ndim(l-1)
       wk%wl(1:nd,0:ndm) = nt%w(l,1:nd,0:ndm)

#ifdef USE_BLAS
       ! The slot axis is a batch axis, so this is one GEMM:
       !   S(1:nd, 1:na) = wl(1:nd, 0:ndm) T(0:ndm, 1:na).
       ! An optimized BLAS is 5 to 7 times faster than the loop shape at
       ! force-field widths.  Summation order differs from the reference
       ! loops, so results agree to roundoff, not bitwise; hence opt-in.
       call bgemm( 'N', 'N', nd, nt%tab%na, ndm+1, &
                   wk%wl, nt%ndmax, wk%T(0:,1:,pm), nt%ndmax+1, &
                   wk%S, nt%ndmax )
#else
       do ia=1,nt%tab%na
          wk%S(1:nd,ia) = wk%wl(1:nd,0)*wk%T(0,ia,pm)
          do i=1,ndm
             wk%S(1:nd,ia) = wk%S(1:nd,ia) + wk%wl(1:nd,i)*wk%T(i,ia,pm)
          end do
       end do
#endif

       if ( l < nt%nlayer ) then
          ! Neuron-axis vectorized Bell recurrence (see train_module for
          ! the full note): loop interchange only, so results are bitwise
          ! identical to the per-neuron reference while j runs at SIMD
          ! width over the contiguous first dimension.
          ! scalar tanh derivatives per neuron (see the note in
          ! train_module: the vector tanh is not bitwise reproducible)
          do jj=1,nd
             call act_derivs_ts( nt%tab, wk%S(jj,1), nt%tab%kmax+1, dsc )
             wk%dt(jj,0:nt%tab%kmax+1) = dsc(0:nt%tab%kmax+1)
          end do
          wk%T(1:nd,1,pc) = wk%dt(1:nd,0)
          do ia=2,nt%tab%na
             p = nt%tab%alpha_deg(ia)
             wk%bqv(1:nd,ia,1) = wk%S(1:nd,ia)
             wk%ttv(1:nd) = wk%dt(1:nd,1)*wk%S(1:nd,ia)
             do q=2,p
                wk%bsv(1:nd) = 0.d0
                do it=nt%tab%fq_start(ia,q),nt%tab%fq_start(ia,q)+nt%tab%fq_num(ia,q)-1
                   wk%bsv(1:nd) = wk%bsv(1:nd) + nt%tab%fq_c(it) &
                        *wk%S(1:nd,nt%tab%fq_ib(it))*wk%bqv(1:nd,nt%tab%fq_id(it),q-1)
                end do
                wk%bqv(1:nd,ia,q) = wk%bsv(1:nd)
                wk%ttv(1:nd) = wk%ttv(1:nd) + wk%dt(1:nd,q)*wk%bsv(1:nd)
             end do
             wk%T(1:nd,ia,pc) = wk%ttv(1:nd)
          end do
       else
          wk%T(1:nd,1:nt%tab%na,pc) = wk%S(1:nd,1:nt%tab%na)
       end if
       pm = pc
    end do

    t(1:nt%tab%na) = wk%T(1,1:nt%tab%na,pm)
    if ( allocated( wk%Tout) ) &
         wk%Tout(1:nt%ndim(nt%nlayer),1:nt%tab%na) = &
         wk%T(1:nt%ndim(nt%nlayer),1:nt%tab%na,pm)

  END SUBROUTINE net_eval_hod


  !> Every carried derivative of every output component.
  !!
  !! The propagation is over all neurons of every layer, the output layer
  !! included, so a network with several outputs needs no different
  !! recursion: only the extraction differs.  net_eval_hod returns the
  !! first component, which is what a scalar field needs; this returns
  !! the whole array, tm(i,ia) being multi-index ia of output i.
  !!
  !! A vector field u = (u_1 .. u_m) is then one network rather than m of
  !! them, so the hidden representation is shared and a residual may
  !! couple the components.
  SUBROUTINE net_eval_hod_multi( nt, wk, x, tm )
    implicit none
    type(net_t),intent(IN) :: nt
    type(work_t),intent(INOUT) :: wk
    real(8),intent(IN) :: x(:)
    real(8),intent(OUT) :: tm(:,:)
    integer :: nout, ia
    real(8) :: tdum(nt%tab%na)
    nout = nt%ndim(nt%nlayer)
    if ( size(tm,1) < nout .or. size(tm,2) < nt%tab%na ) then
       write(*,*) "net_eval_hod_multi: tm must be at least", nout, &
            " by", nt%tab%na
       stop
    end if
    call net_eval_hod( nt, wk, x, tdum )
    ! net_eval_hod leaves the final plane in wk%T; read every component
    do ia = 1, nt%tab%na
       tm(1:nout,ia) = wk%Tout(1:nout,ia)
    end do
  END SUBROUTINE net_eval_hod_multi

END MODULE net_module
