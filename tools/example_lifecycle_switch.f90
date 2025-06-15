! Can one process run init -> free -> init with a DIFFERENT configuration?
!
! global_free releases the allocations; reset_global_state restores the
! variables that have none.  Without the second step the System term
! count, the NGD flags, the committee count and the rest survive into
! the next case and describe a problem that is no longer there.  This
! alternates a five-residual System case (D0=3, K=2) with a plain DATA
! case (D0=1, K=1) and checks both directions.
!
! Run from a directory that holds the two case directories named below.
program lifecycle_switch
  use global_variables, only: sys_nterm, sys_nres, Nlayer
  use api_module, only: dnnf90_init, dnnf90_free, dnnf90_nderiv
  implicit none
  integer :: n1, n2, n3, nfail
  nfail = 0
  call run( 'lc_sys',   n1 )
  call check( 'after the system case',   nfail )
  call run( 'lc_plain', n2 )
  call check( 'after the plain case',    nfail )
  call run( 'lc_sys',   n3 )
  call check( 'after the system case again', nfail )
  write(*,'(a,i0,a,i0,a,i0)') ' carried slots: system ', n1, &
       ', plain ', n2, ', system again ', n3
  if ( n1 /= n3 ) then
     write(*,'(a)') ' FAIL: the second visit to the same case differs from the first'
     nfail = nfail + 1
  end if
  if ( n1 == n2 ) then
     write(*,'(a)') ' FAIL: the two cases should not carry the same set'
     nfail = nfail + 1
  end if
  if ( nfail == 0 ) then
     write(*,'(a)') ' passed'
  else
     write(*,'(a,i0,a)') ' FAILED on ', nfail, ' check(s)'
     stop 1
  end if
contains
  subroutine run( d, n )
    character(len=*),intent(IN) :: d
    integer,intent(OUT) :: n
    call chdir( d )
    call dnnf90_init
    n = dnnf90_nderiv()
    call dnnf90_free
    call chdir( '..' )
  end subroutine run
  subroutine check( what, nf )
    character(len=*),intent(IN) :: what
    integer,intent(INOUT) :: nf
    if ( sys_nterm /= 0 .or. sys_nres /= 0 ) then
       write(*,'(a,a,a,i0,a,i0)') ' FAIL: ', what, &
            ', sys_nterm = ', sys_nterm, ', sys_nres = ', sys_nres
       nf = nf + 1
    end if
  end subroutine check
end program lifecycle_switch
