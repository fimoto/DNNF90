! Does a step along the negative gradient decrease the collocation loss?
!
! The gradient of this loss agrees with central differences to 1e-9, so
! it is the gradient of something.  The question left is whether that
! something is the quantity the trainer reports and descends: if a small
! step along the negative gradient raises the loss, the two are not the
! same object, and no amount of tuning will help.
!
! The test is the simplest one available: evaluate the loss, take the
! gradient, step by -eta*grad for a few decreasing eta, and print the
! loss at each.  For a correct gradient the loss must fall for small
! enough eta, and fall at a rate approaching eta*|grad|^2.
program verify_descent
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, alpha_index
  use net_module, only: net_t, net_init, net_free, work_t, work_init, &
       work_free, net_eval_hod_multi
  use train_module, only: twork_t, grad_t, twork_init, grad_init, &
       twork_free, grad_free, net_forward_point, net_backward_point_multi
  use global_variables, only: sys_nres, sys_nterm, sys_res, sys_coeff, &
       sys_nfac, sys_fcomp, sys_find, sys_has_src, sys_wres
  use pinn_module, only: calc_sys_residual, set_sys_seed
  implicit none
  integer,parameter :: D0 = 2, KMAX = 2, NC = 5, W = 8, NPT = 60
  real(8),parameter :: EPSC = 1.d0, MU = 0.5d0, DIF = 0.1d0, &
       NU = 0.1d0, CF = 0.2d0
  type(net_t) :: nt
  type(work_t) :: wk
  type(twork_t) :: tw
  type(grad_t) :: g
  real(8),allocatable :: tm(:,:), sm(:,:), tdum(:), wsave(:,:,:)
  real(8) :: xs(D0,NPT), src(NC,NPT), R(NC)
  real(8) :: l0, l1, eta, gn
  integer :: sd(D0,1), dims(4), l, j, k, n, it
  integer :: ix, iy, ixx, iyy

  sd = 0
  call init_hod_tables( D0, KMAX, 0, sd )
  dims = (/ D0, W, W, NC /)
  call net_init( nt, 4, dims )
  call work_init( wk, nt );  call twork_init( tw, nt );  call grad_init( g, nt )
  allocate( tm(NC,NUM_alpha), sm(NC,NUM_alpha), tdum(NUM_alpha) )
  allocate( wsave(nt%nlayer,nt%ndmax,0:nt%ndmax) )

  ix  = alpha_index( (/ 1,0 /) );  iy  = alpha_index( (/ 0,1 /) )
  ixx = alpha_index( (/ 2,0 /) );  iyy = alpha_index( (/ 0,2 /) )

  sys_nres = 5;  sys_nterm = 0;  sys_has_src = .false.
  sys_wres = 1.d0;  sys_nfac = 1;  sys_fcomp = 0;  sys_find = 1
  call t1( 1, 1, ixx, 1.d0 );  call t1( 1, 1, iyy, 1.d0 )
  call t1( 1, 2, 1  , 1.d0/EPSC )
  call t2( 2, 2, ix , 3, 1  , 1.d0 )
  call t2( 2, 2, iy , 4, 1  , 1.d0 )
  call t2( 2, 2, ix , 1, ix , -MU )
  call t2( 2, 2, iy , 1, iy , -MU )
  call t2( 2, 2, 1  , 1, ixx, -MU )
  call t2( 2, 2, 1  , 1, iyy, -MU )
  call t1( 2, 2, ixx, -DIF );  call t1( 2, 2, iyy, -DIF )
  call t2( 3, 3, ix , 3, 1  , 1.d0 )
  call t2( 3, 3, iy , 4, 1  , 1.d0 )
  call t1( 3, 5, ix , 1.d0 )
  call t1( 3, 3, ixx, -NU );  call t1( 3, 3, iyy, -NU )
  call t2( 3, 1, ix , 2, 1  , CF )
  call t2( 4, 4, ix , 3, 1  , 1.d0 )
  call t2( 4, 4, iy , 4, 1  , 1.d0 )
  call t1( 4, 5, iy , 1.d0 )
  call t1( 4, 4, ixx, -NU );  call t1( 4, 4, iyy, -NU )
  call t2( 4, 1, iy , 2, 1  , CF )
  call t1( 5, 3, ix , 1.d0 );  call t1( 5, 4, iy , 1.d0 )

  call random_seed()
  call random_number( nt%w );  nt%w = 0.6d0*( nt%w - 0.5d0 )
  do n = 1, NPT
     call random_number( xs(1,n) );  call random_number( xs(2,n) )
     xs(1,n) = 6.d0*xs(1,n) - 3.d0;  xs(2,n) = 6.d0*xs(2,n) - 3.d0
     call sources( xs(1,n), xs(2,n), src(:,n) )
  end do

  l0 = loss()
  g%nabla = 0.d0
  do n = 1, NPT
     call net_forward_point( nt, tw, xs(:,n), tdum )
     call net_eval_hod_multi( nt, wk, xs(:,n), tm )
     call calc_sys_residual( tm, src(:,n), R )
     call set_sys_seed( tm, R, 1.d0, sm )
     call net_backward_point_multi( nt, tw, sm, g )
  end do
  gn = 0.d0
  do l = 2, nt%nlayer
     do j = 1, nt%ndim(l)
        do k = 0, nt%ndim(l-1)
           gn = gn + g%nabla(l,j,k)**2
        end do
     end do
  end do

  write(*,'(a,e13.5,a,e13.5)') " loss ", l0, "   |grad|^2 ", gn
  write(*,'(a)') " a step of -eta*grad should lower it by about eta*|grad|^2"
  wsave = nt%w
  eta = 1.d-2
  do it = 1, 6
     nt%w = wsave
     do l = 2, nt%nlayer
        do j = 1, nt%ndim(l)
           do k = 0, nt%ndim(l-1)
              nt%w(l,j,k) = wsave(l,j,k) - eta*g%nabla(l,j,k)
           end do
        end do
     end do
     l1 = loss()
     write(*,'(a,e10.2,a,e13.5,a,e13.5,a,e13.5)') "   eta ", eta, &
          "  loss ", l1, "  change ", l1-l0, "  predicted ", -eta*gn
     eta = eta*0.1d0
  end do
  nt%w = wsave

  call grad_free( g );  call twork_free( tw );  call work_free( wk )
  call net_free( nt );  deallocate( tm, sm, tdum, wsave )

CONTAINS

  REAL(8) FUNCTION loss()
    implicit none
    integer :: nn
    loss = 0.d0
    do nn = 1, NPT
       call net_eval_hod_multi( nt, wk, xs(:,nn), tm )
       call calc_sys_residual( tm, src(:,nn), R )
       loss = loss + 0.5d0*sum( sys_wres(1:NC)*R(1:NC)**2 )
    end do
  END FUNCTION loss

  SUBROUTINE t1( ir, c1, i1, c )
    implicit none
    integer,intent(IN) :: ir, c1, i1
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir;  sys_coeff(sys_nterm) = c
    sys_nfac(sys_nterm) = 1
    sys_fcomp(1,sys_nterm) = c1;  sys_find(1,sys_nterm) = i1
  END SUBROUTINE t1

  SUBROUTINE t2( ir, c1, i1, c2, i2, c )
    implicit none
    integer,intent(IN) :: ir, c1, i1, c2, i2
    real(8),intent(IN) :: c
    sys_nterm = sys_nterm + 1
    sys_res(sys_nterm) = ir;  sys_coeff(sys_nterm) = c
    sys_nfac(sys_nterm) = 2
    sys_fcomp(1,sys_nterm) = c1;  sys_find(1,sys_nterm) = i1
    sys_fcomp(2,sys_nterm) = c2;  sys_find(2,sys_nterm) = i2
  END SUBROUTINE t2

  SUBROUTINE sources( xa, ya, so )
    implicit none
    real(8),intent(IN) :: xa, ya
    real(8),intent(OUT) :: so(NC)
    real(8) :: s, c, S2, C2, rho, rx, ry, rxx, ryy
    real(8) :: px, py, pxx, pyy, uu, vv, ux, uy, vx, vy
    real(8) :: uxx, uyy, vxx, vyy, ppx, ppy, drgp
    s = sin(xa);  c = cos(xa);  S2 = sin(ya);  C2 = cos(ya)
    px = c*S2;  py = s*C2;  pxx = -s*S2;  pyy = -s*S2
    rho = c*C2 + 0.5d0
    rx = -s*C2;  ry = -c*S2
    rxx = -c*C2;  ryy = -c*C2
    uu = -c*S2;  vv = s*C2
    ux = s*S2;  uy = -c*C2;  vx = c*C2;  vy = -s*S2
    uxx = c*S2;  uyy = c*S2;  vxx = -s*C2;  vyy = -s*C2
    ppx = sin(2.d0*xa)/2.d0;  ppy = sin(2.d0*ya)/2.d0
    drgp = rx*px + rho*pxx + ry*py + rho*pyy
    so(1) = pxx + pyy + rho/EPSC
    so(2) = uu*rx + vv*ry - MU*drgp - DIF*( rxx + ryy )
    so(3) = uu*ux + vv*uy + ppx - NU*( uxx + uyy ) + CF*rho*px
    so(4) = uu*vx + vv*vy + ppy - NU*( vxx + vyy ) + CF*rho*py
    so(5) = 0.d0
  END SUBROUTINE sources

end program verify_descent
