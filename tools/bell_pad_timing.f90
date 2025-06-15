! Is the Bell composition worth regularizing?
!
! The phase measurement (fwd_grad_timing with PHASE_TIMING) says that
! under an optimized BLAS the composition is 89-95 % of the gradient
! cost, so any further speedup of the library must come from this loop.
! Its tables are per-(slot, order) lists of (coefficient, two indices);
! padding them to a regular (slot, order, max-terms) block turns the
! variable-trip gather into a fixed-trip one, which is the shape a
! vectorizer or a dense kernel wants.  The price is arithmetic on zero
! coefficients.  This tool measures both forms on the same random data
! and reports the padding overhead (padded terms / real terms) next to
! the measured time ratio, which is the number the decision needs.
!
!   bell_pad_timing.out <D0> <K> <width> <nrep>
!
! Both kernels compute T(1:nd, ia) for every slot from the same S and
! sigma^(q) inputs; the results are compared to 1e-13 so the padded
! kernel is verified, not assumed, to compute the same composition.
program bell_pad_timing
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha
  use net_module, only: net_t, net_init, net_free
  implicit none
  type(net_t) :: nt
  integer :: d0, kmax, width, nrep, dims(4)
  integer,allocatable :: sd(:,:)
  character(32) :: a
  integer :: na, nd, tmax, nreal, npad
  integer :: ia, q, p, it, j, r, t
  real(8),allocatable :: S(:,:), dt(:,:), bqv(:,:,:), bsv(:), ttv(:)
  real(8),allocatable :: Tlist(:,:), Tpad(:,:)
  real(8),allocatable :: pc(:,:,:)
  integer,allocatable :: pib(:,:,:), pid(:,:,:)
  real(8) :: t0, t1, tlist_s, tpad_s, err

  call get_command_argument(1,a); read(a,*) d0
  call get_command_argument(2,a); read(a,*) kmax
  call get_command_argument(3,a); read(a,*) width
  call get_command_argument(4,a); read(a,*) nrep

  allocate( sd(d0,1) );  sd = 0
  call init_hod_tables( d0, kmax, 0, sd )
  dims = (/ d0, width, width, 1 /)
  call net_init( nt, 4, dims )
  na = nt%tab%na;  nd = width

  ! ---- padded copies of the fq tables ----
  tmax = 0;  nreal = 0
  do ia=2,na
     do q=2,nt%tab%alpha_deg(ia)
        tmax = max( tmax, nt%tab%fq_num(ia,q) )
        nreal = nreal + nt%tab%fq_num(ia,q)
     end do
  end do
  npad = 0
  do ia=2,na
     do q=2,nt%tab%alpha_deg(ia)
        npad = npad + tmax
     end do
  end do
  allocate( pc(tmax,na,2:max(kmax,2)), pib(tmax,na,2:max(kmax,2)), &
            pid(tmax,na,2:max(kmax,2)) )
  pc = 0.d0;  pib = 1;  pid = 1     ! padding reads slot 1 with weight 0
  do ia=2,na
     do q=2,nt%tab%alpha_deg(ia)
        do t=1,nt%tab%fq_num(ia,q)
           it = nt%tab%fq_start(ia,q) + t - 1
           pc(t,ia,q)  = nt%tab%fq_c(it)
           pib(t,ia,q) = nt%tab%fq_ib(it)
           pid(t,ia,q) = nt%tab%fq_id(it)
        end do
     end do
  end do

  allocate( S(nd,na), dt(nd,0:kmax+1), bqv(nd,na,kmax), bsv(nd), ttv(nd) )
  allocate( Tlist(nd,na), Tpad(nd,na) )
  call random_seed()
  call random_number( S );  S = S - 0.5d0
  call random_number( dt );  dt = dt - 0.5d0

  ! ---- list form: the loop of forward_core, verbatim shape ----
  call cpu_time(t0)
  do r=1,nrep
     Tlist(1:nd,1) = dt(1:nd,0)
     do ia=2,na
        p = nt%tab%alpha_deg(ia)
        bqv(1:nd,ia,1) = S(1:nd,ia)
        ttv(1:nd) = dt(1:nd,1)*S(1:nd,ia)
        do q=2,p
           bsv(1:nd) = 0.d0
           do it=nt%tab%fq_start(ia,q),nt%tab%fq_start(ia,q)+nt%tab%fq_num(ia,q)-1
              bsv(1:nd) = bsv(1:nd) + nt%tab%fq_c(it) &
                   *S(1:nd,nt%tab%fq_ib(it))*bqv(1:nd,nt%tab%fq_id(it),q-1)
           end do
           bqv(1:nd,ia,q) = bsv(1:nd)
           ttv(1:nd) = ttv(1:nd) + dt(1:nd,q)*bsv(1:nd)
        end do
        Tlist(1:nd,ia) = ttv(1:nd)
     end do
  end do
  call cpu_time(t1);  tlist_s = (t1-t0)/dble(nrep)

  ! ---- padded form: fixed trip count, zero-weight padding ----
  call cpu_time(t0)
  do r=1,nrep
     Tpad(1:nd,1) = dt(1:nd,0)
     do ia=2,na
        p = nt%tab%alpha_deg(ia)
        bqv(1:nd,ia,1) = S(1:nd,ia)
        ttv(1:nd) = dt(1:nd,1)*S(1:nd,ia)
        do q=2,p
           bsv(1:nd) = 0.d0
           do t=1,tmax
              bsv(1:nd) = bsv(1:nd) + pc(t,ia,q) &
                   *S(1:nd,pib(t,ia,q))*bqv(1:nd,pid(t,ia,q),q-1)
           end do
           bqv(1:nd,ia,q) = bsv(1:nd)
           ttv(1:nd) = ttv(1:nd) + dt(1:nd,q)*bsv(1:nd)
        end do
        Tpad(1:nd,ia) = ttv(1:nd)
     end do
  end do
  call cpu_time(t1);  tpad_s = (t1-t0)/dble(nrep)

  err = maxval( abs( Tlist - Tpad ) )
  write(*,'(a,i0,a,i0,a,i0,a,i0)') " D0=", d0, " K=", kmax, &
       " slots=", na, " width=", nd
  write(*,'(a,i0,a,i0,a,f6.2,a,i0)') " real terms ", nreal, &
       "  padded ", npad, "  overhead x", dble(npad)/dble(max(nreal,1)), &
       "  tmax ", tmax
  write(*,'(a,f12.5,a)') " list form   ", 1.d3*tlist_s, " ms"
  write(*,'(a,f12.5,a,f6.2,a)') " padded form ", 1.d3*tpad_s, " ms  (x", &
       tpad_s/tlist_s, " of list)"
  write(*,'(a,e10.2)') " max |difference| ", err
  if ( err > 1.d-13 ) then
     write(*,*) "MISMATCH: the padded kernel does not reproduce the list one"
     stop 1
  end if
  call net_free( nt )
end program bell_pad_timing
