!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (c_api_module.f90) is part of DNNF90.
!  (MIT License; see LICENSE at the repository root.)
!
! -----------------------------------------------------------------------
! C compatible interface (ISO_C_BINDING) to the instance based engine.
!
! This is the embedding layer for non-Fortran hosts: n2p2 and LAMMPS are
! C++, ASE and i-PI are Python, and all of them can call these entry
! points through the C ABI.  Networks, work spaces and gradient
! accumulators are referred to by opaque integer handles, so no Fortran
! derived type ever crosses the language boundary.
!
! Conventions
!   * every function returns an int: handles are > 0, errors are < 0,
!     DNNF90_OK (0) means success for routines without a handle result
!   * arrays are plain double* / int*; the caller owns them
!   * one work handle per thread makes evaluation and gradient
!     accumulation thread safe (the handle registries themselves must be
!     mutated, that is created and freed, outside parallel regions)
! -----------------------------------------------------------------------
MODULE c_api_module

  use iso_c_binding
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, HOD_D0, &
                                     alpha_list, hod_tables_ready, &
                                     tabset_t, tabset_from_current
  use net_module
  use train_module

  implicit none

  PRIVATE

  integer,parameter :: MAXH = 64        ! handles per registry

  type(net_t),  save :: rnet(MAXH)
  logical,      save :: unet(MAXH) = .false.
  type(work_t), save :: rwrk(MAXH)
  logical,      save :: uwrk(MAXH) = .false.
  !> Generation counter per network slot.  A slot is reused after
  !! dnnf90_net_free, so a slot number alone does not identify a
  !! network: a caller holding a work space built for the previous
  !! occupant would index it against the new one, whose layer widths may
  !! differ.  Every dependent registry records the generation it was
  !! built against and is refused when it no longer matches.
  integer,      save :: gennet(MAXH) = 0
  integer,      save :: wnet(MAXH) = 0  ! net a work space belongs to
  integer,      save :: wgen(MAXH) = 0  ! generation it was built against
  integer,      save :: tgen(MAXH) = 0
  integer,      save :: ggen(MAXH) = 0
  integer,      save :: tnet(MAXH) = 0  ! net a training work space belongs to
  integer,      save :: gnet(MAXH) = 0  ! net an accumulator belongs to
  type(twork_t),save :: rtwk(MAXH)
  logical,      save :: utwk(MAXH) = .false.
  type(grad_t), save :: rgrd(MAXH)
  logical,      save :: ugrd(MAXH) = .false.
  type(tabset_t),save :: rtab(MAXH)
  logical,       save :: utab(MAXH) = .false.

CONTAINS

  !> Argument checks at the C boundary.  The Fortran side stops with a
  !! message on a bad order, but stopping is not what a library caller
  !! expects, and a negative dimension, seed count or seed component
  !! would be used as an array bound before any check ran.  These
  !! return .false. for anything the tables or the network cannot take,
  !! and the entry points then return -1 as their contract says.
  LOGICAL FUNCTION table_args_ok( d0, k, nseed, seeds )
    integer(c_int),intent(IN) :: d0, k, nseed
    integer(c_int),intent(IN),optional :: seeds(*)
    integer :: i
    table_args_ok = .false.
    if ( d0 < 1 .or. k < 1 .or. k > 15 .or. nseed < 0 ) return
    if ( present(seeds) ) then
       do i = 1, d0*nseed
          if ( seeds(i) < 0 .or. seeds(i) > k ) return
       end do
    end if
    table_args_ok = .true.
  END FUNCTION table_args_ok

  LOGICAL FUNCTION net_args_ok( nlayer, dims )
    integer(c_int),intent(IN) :: nlayer
    integer(c_int),intent(IN) :: dims(*)
    integer :: l
    net_args_ok = .false.
    if ( nlayer < 2 ) return
    do l = 1, nlayer
       if ( dims(l) < 1 ) return
    end do
    net_args_ok = .true.
  END FUNCTION net_args_ok

  !> Build the shared tables for the dense set of all derivatives up to
  !! order k in d0 variables.  Call once before anything else.
  INTEGER(c_int) FUNCTION dnnf90_tables_init_dense( d0, k ) bind(C)
    integer(c_int),value :: d0, k
    integer :: dummy(1,1)
    if ( .not. table_args_ok( d0, k, 0_c_int ) ) then
       dnnf90_tables_init_dense = -1; return
    end if
    call init_hod_tables( int(d0), int(k), 0, dummy )
    dnnf90_tables_init_dense = 0
  END FUNCTION dnnf90_tables_init_dense

  !> Build the shared tables restricted to the downward closure of nseed
  !! seed multi-indices.  seeds is d0*nseed ints, seed s occupying
  !! seeds(1+(s-1)*d0 : s*d0).
  INTEGER(c_int) FUNCTION dnnf90_tables_init_closure( d0, k, nseed, seeds ) bind(C)
    integer(c_int),value :: d0, k, nseed
    integer(c_int),intent(IN) :: seeds(*)
    integer :: sa(max(d0,1),max(nseed,1)), s
    if ( .not. table_args_ok( d0, k, nseed, seeds ) ) then
       dnnf90_tables_init_closure = -1; return
    end if
    do s=1,nseed
       sa(1:d0,s) = int( seeds(1+(s-1)*d0 : s*d0) )
    end do
    call init_hod_tables( int(d0), int(k), int(nseed), sa )
    dnnf90_tables_init_closure = 0
  END FUNCTION dnnf90_tables_init_closure

  !> Number of carried multi-indices (slot 1 is the value itself).
  INTEGER(c_int) FUNCTION dnnf90_nderiv() bind(C)
    dnnf90_nderiv = int( NUM_alpha, c_int )
  END FUNCTION dnnf90_nderiv

  !> Multi-index of slot ia, written to a(1:d0).
  INTEGER(c_int) FUNCTION dnnf90_alpha( ia, a ) bind(C)
    integer(c_int),value :: ia
    integer(c_int),intent(OUT) :: a(*)
    if ( ia < 1 .or. ia > NUM_alpha ) then
       dnnf90_alpha = -1; return
    end if
    a(1:HOD_D0) = int( alpha_list(1:HOD_D0,ia), c_int )
    dnnf90_alpha = 0
  END FUNCTION dnnf90_alpha

  !> Load a trained network from a file in nn_weight.dat format.
  !! filename is a NUL terminated C string.  Returns a handle.
  INTEGER(c_int) FUNCTION dnnf90_net_load( filename ) bind(C)
    character(kind=c_char),intent(IN) :: filename(*)
    character(len=4096) :: fname
    integer :: i, id
    logical :: ex
    fname = ' '
    do i=1,len(fname)
       if ( filename(i) == c_null_char ) exit
       fname(i:i) = filename(i)
    end do
    inquire( file=trim(fname), exist=ex )
    if ( .not. ex ) then
       dnnf90_net_load = -2; return
    end if
    id = free_slot( unet )
    if ( id < 0 ) then
       dnnf90_net_load = -1; return
    end if
    call net_load( rnet(id), trim(fname) )
    unet(id) = .true.
    dnnf90_net_load = id
  END FUNCTION dnnf90_net_load

  !> Create an empty network of the given shape.  Returns a handle.
  INTEGER(c_int) FUNCTION dnnf90_net_create( nlayer, dims ) bind(C)
    integer(c_int),value :: nlayer
    integer(c_int),intent(IN) :: dims(*)
    integer :: id
    if ( .not. net_args_ok( nlayer, dims ) ) then
       dnnf90_net_create = -1; return
    end if
    id = free_slot( unet )
    if ( id < 0 ) then
       dnnf90_net_create = -1; return
    end if
    call net_init( rnet(id), int(nlayer), int(dims(1:nlayer)) )
    unet(id) = .true.
    dnnf90_net_create = id
  END FUNCTION dnnf90_net_create

  !> Set the parameters of layer l (2..nlayer) from wrow, which holds
  !! ndim(l) rows of length ndim(l-1)+1, each row being the bias followed
  !! by the incoming weights, exactly like one "#l=" block of
  !! nn_weight.dat.
  INTEGER(c_int) FUNCTION dnnf90_net_set_layer( nid, l, wrow ) bind(C)
    integer(c_int),value :: nid, l
    real(c_double),intent(IN) :: wrow(*)
    integer :: j, ndm, k0
    if ( bad_net(nid) .or. l < 2 .or. l > rnet(nid)%nlayer ) then
       dnnf90_net_set_layer = -1; return
    end if
    ! Row j occupies ndm+1 doubles: the bias followed by the ndm incoming
    ! weights, packed with no padding, exactly like one "#l=" block of
    ! nn_weight.dat.  The stride is therefore ndm+1, not ndm+2.
    ndm = rnet(nid)%ndim(l-1)
    do j=1,rnet(nid)%ndim(l)
       k0 = (j-1)*(ndm+1)
       rnet(nid)%w(l,j,0:ndm) = wrow(k0+1:k0+ndm+1)
    end do
    dnnf90_net_set_layer = 0
  END FUNCTION dnnf90_net_set_layer

  INTEGER(c_int) FUNCTION dnnf90_net_free( nid ) bind(C)
    integer(c_int),value :: nid
    if ( bad_net(nid) ) then
       dnnf90_net_free = -1; return
    end if
    call net_free( rnet(nid) )
    unet(nid) = .false.
    ! advance the generation so that anything built against this
    ! occupant is refused rather than reinterpreted
    gennet(nid) = gennet(nid) + 1
    dnnf90_net_free = 0
  END FUNCTION dnnf90_net_free

  !> Evaluation work space for a network.  One per thread.
  INTEGER(c_int) FUNCTION dnnf90_work_create( nid ) bind(C)
    integer(c_int),value :: nid
    integer :: id
    if ( bad_net(nid) ) then
       dnnf90_work_create = -1; return
    end if
    id = free_slot( uwrk )
    if ( id < 0 ) then
       dnnf90_work_create = -1; return
    end if
    call work_init( rwrk(id), rnet(nid) )
    uwrk(id) = .true.
    wnet(id) = nid
    wgen(id) = gennet(nid)
    dnnf90_work_create = id
  END FUNCTION dnnf90_work_create

  INTEGER(c_int) FUNCTION dnnf90_work_free( wid ) bind(C)
    integer(c_int),value :: wid
    if ( wid < 1 .or. wid > MAXH ) then
       dnnf90_work_free = -1; return
    end if
    if ( .not. uwrk(wid) ) then
       dnnf90_work_free = -1; return
    end if
    call work_free( rwrk(wid) )
    uwrk(wid) = .false.
    dnnf90_work_free = 0
  END FUNCTION dnnf90_work_free

  !> Evaluate: t(1:nderiv) receives the value and all carried mixed
  !! derivatives at x(1:d0).
  INTEGER(c_int) FUNCTION dnnf90_eval( nid, wid, x, t ) bind(C)
    integer(c_int),value :: nid, wid
    real(c_double),intent(IN)  :: x(*)
    real(c_double),intent(OUT) :: t(*)
    ! Allocatable, not automatic: the bounds of an automatic array are
    ! evaluated on entry, before the first executable statement, so
    ! sizing it from rnet(nid) would touch the registry with an
    ! unvalidated handle and crash on exactly the calls this function is
    ! supposed to reject.
    real(8),allocatable :: tf(:)
    if ( bad_net(nid) .or. wid < 1 .or. wid > MAXH ) then
       dnnf90_eval = -1; return
    end if
    if ( .not. uwrk(wid) ) then
       dnnf90_eval = -1; return
    end if
    ! the slot AND the occupant it was built for
    if ( wnet(wid) /= nid .or. wgen(wid) /= gennet(nid) ) then
       dnnf90_eval = -1; return
    end if
    allocate( tf(rnet(nid)%tab%na) )
    call net_eval_hod( rnet(nid), rwrk(wid), x(1:rnet(nid)%ndim(1)), tf )
    t(1:rnet(nid)%tab%na) = tf(1:rnet(nid)%tab%na)
    deallocate( tf )
    dnnf90_eval = 0
  END FUNCTION dnnf90_eval

  ! ------------------- training entry points -------------------

  INTEGER(c_int) FUNCTION dnnf90_twork_create( nid ) bind(C)
    integer(c_int),value :: nid
    integer :: id
    if ( bad_net(nid) ) then
       dnnf90_twork_create = -1; return
    end if
    id = free_slot( utwk )
    if ( id < 0 ) then
       dnnf90_twork_create = -1; return
    end if
    call twork_init( rtwk(id), rnet(nid) )
    utwk(id) = .true.
    tnet(id) = nid
    tgen(id) = gennet(nid)
    dnnf90_twork_create = id
  END FUNCTION dnnf90_twork_create

  INTEGER(c_int) FUNCTION dnnf90_twork_free( tid ) bind(C)
    integer(c_int),value :: tid
    if ( tid < 1 .or. tid > MAXH ) then
       dnnf90_twork_free = -1; return
    end if
    if ( .not. utwk(tid) ) then
       dnnf90_twork_free = -1; return
    end if
    call twork_free( rtwk(tid) )
    utwk(tid) = .false.
    dnnf90_twork_free = 0
  END FUNCTION dnnf90_twork_free

  INTEGER(c_int) FUNCTION dnnf90_grad_create( nid ) bind(C)
    integer(c_int),value :: nid
    integer :: id
    if ( bad_net(nid) ) then
       dnnf90_grad_create = -1; return
    end if
    id = free_slot( ugrd )
    if ( id < 0 ) then
       dnnf90_grad_create = -1; return
    end if
    call grad_init( rgrd(id), rnet(nid) )
    ugrd(id) = .true.
    gnet(id) = nid
    ggen(id) = gennet(nid)
    dnnf90_grad_create = id
  END FUNCTION dnnf90_grad_create

  INTEGER(c_int) FUNCTION dnnf90_grad_free( gid ) bind(C)
    integer(c_int),value :: gid
    if ( gid < 1 .or. gid > MAXH ) then
       dnnf90_grad_free = -1; return
    end if
    if ( .not. ugrd(gid) ) then
       dnnf90_grad_free = -1; return
    end if
    call grad_free( rgrd(gid) )
    ugrd(gid) = .false.
    dnnf90_grad_free = 0
  END FUNCTION dnnf90_grad_free

  INTEGER(c_int) FUNCTION dnnf90_grad_zero( gid ) bind(C)
    integer(c_int),value :: gid
    if ( gid < 1 .or. gid > MAXH ) then
       dnnf90_grad_zero = -1; return
    end if
    if ( .not. ugrd(gid) ) then
       dnnf90_grad_zero = -1; return
    end if
    call grad_zero( rgrd(gid) )
    dnnf90_grad_zero = 0
  END FUNCTION dnnf90_grad_zero

  !> Add one point to a gradient accumulator: forward, adjoint and
  !! accumulation.  seed(1:nderiv) is dL/dT at the output, that is the
  !! loss specific seeding supplied by the host (for a force field, the
  !! host builds it from its own energy and force residuals).
  INTEGER(c_int) FUNCTION dnnf90_grad_point( nid, tid, x, seed, gid ) bind(C)
    integer(c_int),value :: nid, tid, gid
    real(c_double),intent(IN) :: x(*), seed(*)
    real(8),allocatable :: sf(:)          ! see dnnf90_eval for why
    if ( bad_net(nid) ) then
       dnnf90_grad_point = -1; return
    end if
    if ( tid<1 .or. tid>MAXH .or. gid<1 .or. gid>MAXH ) then
       dnnf90_grad_point = -1; return
    end if
    if ( (.not.utwk(tid)) .or. (.not.ugrd(gid)) ) then
       dnnf90_grad_point = -1; return
    end if
    if ( tnet(tid) /= nid .or. gnet(gid) /= nid .or. &
         tgen(tid) /= gennet(nid) .or. ggen(gid) /= gennet(nid) ) then
       dnnf90_grad_point = -1; return     ! scratch of a different network
    end if
    allocate( sf(rnet(nid)%tab%na) )
    sf(1:rnet(nid)%tab%na) = seed(1:rnet(nid)%tab%na)
    call net_grad_point( rnet(nid), rtwk(tid), x(1:rnet(nid)%ndim(1)), sf, rgrd(gid) )
    deallocate( sf )
    dnnf90_grad_point = 0
  END FUNCTION dnnf90_grad_point

  !> Adam update of a network from an accumulator (same formulas and bias
  !! correction as the standalone trainer).
  INTEGER(c_int) FUNCTION dnnf90_adam_step( nid, gid, eta, beta1, beta2, eps, &
                                            nbatch, istep ) bind(C)
    integer(c_int),value :: nid, gid, nbatch, istep
    real(c_double),value :: eta, beta1, beta2, eps
    if ( bad_net(nid) .or. gid < 1 .or. gid > MAXH ) then
       dnnf90_adam_step = -1; return
    end if
    if ( .not. ugrd(gid) ) then
       dnnf90_adam_step = -1; return
    end if
    if ( gnet(gid) /= nid .or. ggen(gid) /= gennet(nid) ) then
       dnnf90_adam_step = -1; return
    end if
    call opt_adam_step( rnet(nid), rgrd(gid), eta, beta1, beta2, eps, &
                        int(nbatch), int(istep) )
    dnnf90_adam_step = 0
  END FUNCTION dnnf90_adam_step

  ! ---------------- multiple table sets (per species D0 and K) ----------------

  !> Build an independent table set for the dense derivative family and
  !! return its handle.  Sets with different d0 and k coexist; bind each
  !! network to one with dnnf90_net_create_ts / dnnf90_net_load_ts.
  INTEGER(c_int) FUNCTION dnnf90_tabset_dense( d0, k ) bind(C)
    integer(c_int),value :: d0, k
    integer :: dummy(1,1), id
    if ( .not. table_args_ok( d0, k, 0_c_int ) ) then
       dnnf90_tabset_dense = -1; return
    end if
    id = free_slot( utab )
    if ( id < 0 ) then
       dnnf90_tabset_dense = -1; return
    end if
    call init_hod_tables( int(d0), int(k), 0, dummy )
    call tabset_from_current( rtab(id) )
    utab(id) = .true.
    dnnf90_tabset_dense = id
  END FUNCTION dnnf90_tabset_dense

  INTEGER(c_int) FUNCTION dnnf90_tabset_closure( d0, k, nseed, seeds ) bind(C)
    integer(c_int),value :: d0, k, nseed
    integer(c_int),intent(IN) :: seeds(*)
    integer :: sa(max(d0,1),max(nseed,1)), sidx, id
    if ( .not. table_args_ok( d0, k, nseed, seeds ) ) then
       dnnf90_tabset_closure = -1; return
    end if
    id = free_slot( utab )
    if ( id < 0 ) then
       dnnf90_tabset_closure = -1; return
    end if
    do sidx=1,nseed
       sa(1:d0,sidx) = int( seeds(1+(sidx-1)*d0 : sidx*d0) )
    end do
    call init_hod_tables( int(d0), int(k), int(nseed), sa )
    call tabset_from_current( rtab(id) )
    utab(id) = .true.
    dnnf90_tabset_closure = id
  END FUNCTION dnnf90_tabset_closure

  INTEGER(c_int) FUNCTION dnnf90_tabset_nderiv( tid ) bind(C)
    integer(c_int),value :: tid
    if ( tid < 1 .or. tid > MAXH ) then
       dnnf90_tabset_nderiv = -1; return
    end if
    if ( .not. utab(tid) ) then
       dnnf90_tabset_nderiv = -1; return
    end if
    dnnf90_tabset_nderiv = int( rtab(tid)%na, c_int )
  END FUNCTION dnnf90_tabset_nderiv

  INTEGER(c_int) FUNCTION dnnf90_net_create_ts( tid, nlayer, dims ) bind(C)
    integer(c_int),value :: tid, nlayer
    integer(c_int),intent(IN) :: dims(*)
    integer :: id
    if ( .not. net_args_ok( nlayer, dims ) ) then
       dnnf90_net_create_ts = -1; return
    end if
    if ( tid < 1 .or. tid > MAXH ) then
       dnnf90_net_create_ts = -1; return
    end if
    if ( .not. utab(tid) ) then
       dnnf90_net_create_ts = -1; return
    end if
    id = free_slot( unet )
    if ( id < 0 ) then
       dnnf90_net_create_ts = -1; return
    end if
    call net_init( rnet(id), int(nlayer), int(dims(1:nlayer)), rtab(tid) )
    unet(id) = .true.
    dnnf90_net_create_ts = id
  END FUNCTION dnnf90_net_create_ts

  !> Set the activation a table set carries: 0 TANH, 1 SIN, 2 ERF,
  !! 3 BESSEL (J_0), 4 BESSEL1 (J_1).  A table set is built with tanh by
  !! default, so this must be called before loading a weight file that
  !! was trained with anything else.  Returns 0 on success.
  INTEGER(c_int) FUNCTION dnnf90_tabset_activation( tid, iact ) bind(C)
    integer(c_int),value :: tid, iact
    if ( tid < 1 .or. tid > MAXH ) then
       dnnf90_tabset_activation = -1; return
    end if
    if ( .not. utab(tid) ) then
       dnnf90_tabset_activation = -1; return
    end if
    if ( iact < 0 .or. iact > 4 ) then
       dnnf90_tabset_activation = -2; return
    end if
    rtab(tid)%iact = int(iact)
    dnnf90_tabset_activation = 0
  END FUNCTION dnnf90_tabset_activation

  !> Load a network onto a named table set.  The plain dnnf90_net_load
  !! uses whatever table set is current, which for a weight file trained
  !! with a non-tanh activation is refused by net_load; this entry lets
  !! the caller supply the matching set, so that J_0 or sine networks
  !! can be read back through the C interface.
  INTEGER(c_int) FUNCTION dnnf90_net_load_ts( tid, filename ) bind(C)
    integer(c_int),value :: tid
    character(kind=c_char),intent(IN) :: filename(*)
    character(len=4096) :: fname
    integer :: i, id
    logical :: ex
    if ( tid < 1 .or. tid > MAXH ) then
       dnnf90_net_load_ts = -1; return
    end if
    if ( .not. utab(tid) ) then
       dnnf90_net_load_ts = -1; return
    end if
    fname = ' '
    do i=1,len(fname)
       if ( filename(i) == c_null_char ) exit
       fname(i:i) = filename(i)
    end do
    inquire( file=trim(fname), exist=ex )
    if ( .not. ex ) then
       dnnf90_net_load_ts = -2; return
    end if
    id = free_slot( unet )
    if ( id < 0 ) then
       dnnf90_net_load_ts = -1; return
    end if
    call net_load( rnet(id), trim(fname), rtab(tid) )
    unet(id) = .true.
    dnnf90_net_load_ts = id
  END FUNCTION dnnf90_net_load_ts

  ! ------------------- helpers -------------------

  INTEGER FUNCTION free_slot( used )
    logical,intent(IN) :: used(MAXH)
    integer :: i
    free_slot = -1
    do i=1,MAXH
       if ( .not. used(i) ) then
          free_slot = i; return
       end if
    end do
  END FUNCTION free_slot

  LOGICAL FUNCTION bad_net( nid )
    integer(c_int),intent(IN) :: nid
    bad_net = ( nid < 1 .or. nid > MAXH )
    if ( .not. bad_net ) bad_net = .not. unet(nid)
  END FUNCTION bad_net

END MODULE c_api_module
