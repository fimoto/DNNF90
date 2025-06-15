! Is the adjoint of a product term right?
!
! The residual of a system may hold a product of two or three derivative
! factors, and its seed follows the product rule: factor m is seeded with
! the product of every other factor.  That is easy to write and easy to
! get subtly wrong, in two ways in particular.
!
!   - when two factors are the same entry, as in u_x times u_x, both
!     contributions land on it and must add, giving the factor of two;
!   - with three or four factors, the seed of each is a product of the
!     others,
!     not of all three.
!
! Both are checked here against central differences of the loss, on a
! network small enough for the check to be cheap.  Every term form the
! input language offers appears in the residual below.
program verify_product_adj
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, alpha_index
  use net_module, only: net_t, net_init, net_free, work_t, work_init, &
       work_free, net_eval_hod_multi
  use train_module, only: twork_t, grad_t, twork_init, grad_init, &
       twork_free, grad_free, net_forward_point, net_backward_point_multi
  use global_variables, only: sys_nres, sys_nterm, sys_res, sys_coeff, &
       sys_nfac, sys_fcomp, sys_find, sys_has_src
  use pinn_module, only: calc_sys_residual, set_sys_seed
  implicit none
  integer,parameter :: D0 = 2, KMAX = 2, NC = 3, W = 5
  integer,parameter :: NR = 4        ! residuals (more than components here)
  type(net_t) :: nt
  type(work_t) :: wk
  type(twork_t) :: tw
  type(grad_t) :: g
  real(8),allocatable :: tm(:,:), sm(:,:), tdum(:)
  real(8) :: x(D0), R(NR), src(NR)
  real(8) :: hw, lp, lm, fd, an, emax, gmax, wsave
  integer :: sd(D0,1), dims(4), l, j, k, nchk, ix, iy, ixx

  sd = 0
  call init_hod_tables( D0, KMAX, 0, sd )
  dims = (/ D0, W, W, NC /)
  call net_init( nt, 4, dims )
  call work_init( wk, nt )
  call twork_init( tw, nt )
  call grad_init( g, nt )
  allocate( tm(NC,NUM_alpha), sm(NC,NUM_alpha), tdum(NUM_alpha) )

  ix = alpha_index( (/ 1,0 /) )
  iy = alpha_index( (/ 0,1 /) )
  ixx = alpha_index( (/ 2,0 /) )

  sys_nres = NR;  sys_nterm = 0;  sys_has_src = .false.

  ! R1: one factor, two factors of different entries, and a square
  call term1( 1, 1, ixx, 1.d0 )                      ! u1_xx
  call term2( 1, 2, 1  , 1, ix , 0.7d0 )             ! u2 * u1_x
  call term2( 1, 1, ix , 1, ix , 0.4d0 )             ! (u1_x)^2   <- the square
  ! R2: a product of two derivatives of different components
  call term2( 2, 2, ix , 3, iy , 1.3d0 )             ! u2_x * u3_y
  call term1( 2, 3, ix , -0.5d0 )
  ! R3: three factors, including a repeated one
  call term3( 3, 1, 1, 2, ix , 3, iy , 0.9d0 )       ! u1 * u2_x * u3_y
  call term3( 3, 2, 1, 2, 1 , 2, ix , 0.6d0 )        ! u2 * u2 * u2_x
  call term1( 3, 3, 1, 0.3d0 )
  ! R4: four factors -- all different, two coinciding, and all four the
  ! same entry.  The last is the case the product rule has to weight by
  ! four, and it is the arity the seventh-order Lax equation needs.
  call term4( 4, 1, 1  , 2, ix , 3, iy , 1, ixx, 1.1d0 )
  call term4( 4, 2, ix , 2, ix , 3, 1  , 1, iy , 0.8d0 )
  call term4( 4, 3, iy , 3, iy , 3, iy , 3, iy , 0.5d0 )

  call random_seed()
  call random_number( nt%w );  nt%w = 0.7d0*( nt%w - 0.5d0 )
  x = (/ 0.37d0, -0.21d0 /)
  src = 0.d0

  g%nabla = 0.d0
  call net_forward_point( nt, tw, x, tdum )
  call net_eval_hod_multi( nt, wk, x, tm )
  call calc_sys_residual( tm, src, R )
  call set_sys_seed( tm, R, 1.d0, sm )
  call net_backward_point_multi( nt, tw, sm, g )

  hw = 1.d-5;  emax = 0.d0;  gmax = 0.d0;  nchk = 0
  do l = 2, nt%nlayer
     do j = 1, nt%ndim(l)
        do k = 0, nt%ndim(l-1)
           wsave = nt%w(l,j,k)
           nt%w(l,j,k) = wsave + hw
           call net_eval_hod_multi( nt, wk, x, tm )
           call calc_sys_residual( tm, src, R )
           lp = 0.5d0*sum( R(1:NR)**2 )
           nt%w(l,j,k) = wsave - hw
           call net_eval_hod_multi( nt, wk, x, tm )
           call calc_sys_residual( tm, src, R )
           lm = 0.5d0*sum( R(1:NR)**2 )
           nt%w(l,j,k) = wsave
           fd = ( lp - lm )/( 2.d0*hw )
           an = g%nabla(l,j,k)
           emax = max( emax, abs(an-fd) );  gmax = max( gmax, abs(fd) )
           nchk = nchk + 1
        end do
     end do
  end do

  write(*,'(a)') " residual with one-, two- and three-factor terms,"
  write(*,'(a)') " including a squared factor and a repeated component"
  write(*,'(a,i0,a,e13.5)') " adjoint vs FD of the loss, ", nchk, &
       " weights, max rel = ", emax/max(gmax,1.d-300)
  if ( emax/max(gmax,1.d-300) > 1.d-6 ) then
     write(*,'(a)') " FAILED"
     stop 1
  end if
  write(*,'(a)') " passed"

  call grad_free( g );  call twork_free( tw );  call work_free( wk )
  call net_free( nt );  deallocate( tm, sm, tdum )

CONTAINS

  SUBROUTINE term1( ir, c1, i1, c )
    implicit none
    integer,intent(IN) :: ir, c1, i1
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir;  sys_coeff(sys_nterm) = c
    sys_nfac(sys_nterm) = 1
    sys_fcomp(1,sys_nterm) = c1;  sys_find(1,sys_nterm) = i1
  END SUBROUTINE term1

  SUBROUTINE term2( ir, c1, i1, c2, i2, c )
    implicit none
    integer,intent(IN) :: ir, c1, i1, c2, i2
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir;  sys_coeff(sys_nterm) = c
    sys_nfac(sys_nterm) = 2
    sys_fcomp(1,sys_nterm) = c1;  sys_find(1,sys_nterm) = i1
    sys_fcomp(2,sys_nterm) = c2;  sys_find(2,sys_nterm) = i2
  END SUBROUTINE term2

  SUBROUTINE term3( ir, c1, i1, c2, i2, c3, i3, c )
    implicit none
    integer,intent(IN) :: ir, c1, i1, c2, i2, c3, i3
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir;  sys_coeff(sys_nterm) = c
    sys_nfac(sys_nterm) = 3
    sys_fcomp(1,sys_nterm) = c1;  sys_find(1,sys_nterm) = i1
    sys_fcomp(2,sys_nterm) = c2;  sys_find(2,sys_nterm) = i2
    sys_fcomp(3,sys_nterm) = c3;  sys_find(3,sys_nterm) = i3
  END SUBROUTINE term3

  SUBROUTINE term4( ir, c1, i1, c2, i2, c3, i3, c4, i4, c )
    implicit none
    integer,intent(IN) :: ir, c1, i1, c2, i2, c3, i3, c4, i4
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir;  sys_coeff(sys_nterm) = c
    sys_nfac(sys_nterm) = 4
    sys_fcomp(1,sys_nterm) = c1;  sys_find(1,sys_nterm) = i1
    sys_fcomp(2,sys_nterm) = c2;  sys_find(2,sys_nterm) = i2
    sys_fcomp(3,sys_nterm) = c3;  sys_find(3,sys_nterm) = i3
    sys_fcomp(4,sys_nterm) = c4;  sys_find(4,sys_nterm) = i4
  END SUBROUTINE term4

end program verify_product_adj
