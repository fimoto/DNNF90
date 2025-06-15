! -----------------------------------------------------------------------
! This file (committee_run_module.f90) is part of DNNF90.
! (MIT License; see LICENSE at the repository root.)
!
! Committee evaluation driven from input_nn.dat.
!
! The members are networks trained independently, which the trainer does
! not produce in one run: the usual recipe is the same input with a
! different Rand_seed, saving each nn_weight.dat under its own name.
! This module consumes those files.  It builds the library committee on
! the multi-index set the current input defines, evaluates every input
! point, and writes the mean and the sample standard deviation of every
! carried slot, so the spread of a derivative is available and not only
! that of the value.
!
! The spread is what an on-the-fly scheme thresholds when it decides
! whether the expensive reference calculation has to be called, so the
! output is written per point rather than reduced to one number.
! -----------------------------------------------------------------------
MODULE committee_run_module

  use global_variables
  use multi_index_bell_module, only: tabset_t, tabset_init, tabset_free, &
       tabset_from_current, hod_tables_ready, NUM_alpha, alpha_deg
  use net_module, only: net_t, net_load, net_free, work_t, work_init, work_free
  use committee_module, only: committee_t, comm_eval, comm_free
  implicit none
  PRIVATE
  PUBLIC :: run_committee

CONTAINS

  SUBROUTINE run_committee
    implicit none
    type(committee_t) :: cm
    type(tabset_t)    :: ts
    type(work_t)      :: wk
    real(8),allocatable :: tmean(:), tstd(:)
    integer,parameter :: uo=61
    integer :: k, n, j, is, ie, na
    character(40) :: fname
    logical :: ex
    integer :: seed0(ndim(1), 1)

    if ( n_committee < 2 ) then
       write(*,*) "Task COMMITTEE requires a Committee block listing at"
       write(*,*) "  least two member weight files"
       stop
    end if

    write(*,'(a)') "### Committee evaluation"
    write(*,'(a,i0,a)') "###   ", n_committee, " members"

    ! Every member must carry the multi-index set of the current input,
    ! otherwise the slots would not line up.  net_load takes the table
    ! set, so the members share the one the trainer built.
    cm%nmem = n_committee
    allocate( cm%mem(n_committee) )
    if ( .not. hod_tables_ready ) then
       ! A plain value input builds no tables.  With derivative output
       ! requested the dense first-order set carries every dN/dx_i, so
       ! their spread is reported too; otherwise the value alone.
       if ( iswitch_out_deriv /= 0 ) then
          call tabset_init( ts, ndim(1), 1, 0, seed0 )
       else
          seed0 = 0
          call tabset_init( ts, ndim(1), 1, 1, seed0 )
       end if
    else
       call tabset_from_current( ts )
    end if
    ! The table set must carry the activation the members were trained
    ! with, or net_load refuses every non-tanh member.
    select case ( trim(Activation_type) )
    case ( "SIN" );     ts%iact = 1
    case ( "ERF" );     ts%iact = 2
    case ( "BESSEL" );  ts%iact = 3
    case ( "BESSEL1" ); ts%iact = 4
    case default;       ts%iact = 0
    end select
    do k=1,n_committee
       inquire( file=trim(committee_file(k)), exist=ex )
       if ( .not. ex ) then
          write(*,*) "Committee: cannot open ", trim(committee_file(k))
          stop
       end if
       call net_load( cm%mem(k), trim(committee_file(k)), ts )
    end do

    na = cm%mem(1)%tab%na
    do k=2,n_committee
       if ( cm%mem(k)%tab%na /= na ) then
          write(*,*) "Committee: member", k, " carries", cm%mem(k)%tab%na, &
               " slots but member 1 carries", na
          stop
       end if
    end do
    write(*,'(a,i0,a)') "###   ", na, " carried slot(s) per point"

    call work_init( wk, cm%mem(1) )
    allocate( tmean(na), tstd(na) )

    do j=1,Ntot_train_set
       is = label_start(j)
       ie = label_end(j)
       write(fname,'("output_committee_set",i4.4,".dat")') j
       open(uo,file=trim(fname),status='replace')
       write(uo,'(a)') "# committee evaluation: mean and sample standard"
       write(uo,'(a)') "# deviation over the members, per carried slot"
       write(uo,'(a,i0,a,i0)') "# members = ", n_committee, "   slots = ", na
       if ( na > 1 ) then
          write(uo,'(a)') "# x(1:D0)  mean(1:slots)  std(1:slots)"
          write(uo,'(a)') "# slot order is that of hod_alpha_order.dat"
       else
          write(uo,'(a)') "# x(1:D0)  mean  std"
       end if
       do n=is,ie
          call comm_eval( cm, wk, descriptor_input(n,1:ndim(1)), tmean, tstd )
          write(uo,'(100e20.10)') descriptor_input(n,1:ndim(1)), &
               tmean(1:na), tstd(1:na)
       end do
       close(uo)
       write(*,'(a,a,a,i0,a)') "###   wrote ", trim(fname), " (", ie-is+1, " points)"
    end do

    ! A single number is convenient for a quick look: the largest spread
    ! of the value over all points, which is what a threshold would see.
    call summarize( cm, wk, tmean, tstd, na )

    deallocate( tmean, tstd )
    call work_free( wk )
    call comm_free( cm )
    ! the members only aliased this table set, so it is released once
    call tabset_free( ts )
  END SUBROUTINE run_committee

  SUBROUTINE summarize( cm, wk, tmean, tstd, na )
    implicit none
    type(committee_t),intent(IN) :: cm
    type(work_t),intent(INOUT) :: wk
    real(8),intent(INOUT) :: tmean(:), tstd(:)
    integer,intent(IN) :: na
    integer :: n, ia
    real(8) :: smax(na), smean(na)
    smax = 0.d0;  smean = 0.d0
    do n=1,NUM_input
       call comm_eval( cm, wk, descriptor_input(n,1:ndim(1)), tmean, tstd )
       do ia=1,na
          smax(ia)  = max( smax(ia), tstd(ia) )
          smean(ia) = smean(ia) + tstd(ia)
       end do
    end do
    smean = smean/dble(NUM_input)
    write(*,'(a)') "###   spread over the input set"
    write(*,'(a)') "###     slot  |alpha|      mean std       max std"
    do ia=1,na
       if ( na > 1 ) then
          write(*,'(a,i8,i8,2es15.5)') "###", ia, alpha_deg(ia), smean(ia), smax(ia)
       else
          write(*,'(a,i8,a8,2es15.5)') "###", ia, "value", smean(ia), smax(ia)
       end if
    end do
  END SUBROUTINE summarize

END MODULE committee_run_module
