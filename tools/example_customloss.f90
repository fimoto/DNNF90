!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (example_customloss.f90) is part of DNNF90.
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
! Custom losses through the seed interface.
!
! The library contract is loss agnostic: net_grad_point returns the
! weight gradient of ANY differentiable functional of the carried
! derivative slots, given its seed dL/dT_alpha (an |A|-dimensional
! calculation on the caller side).  The keyword-driven trainer offers
! three convenience forms (MATH, MATH_HOD, PINN), but a loss outside
! that menu is one seed away, not outside DNNF90.
!
! Demonstration: robust regression that the trainer menu cannot express,
!   L = sum_i huber_delta( N(x_i) - y_i )  +  lam3 * sum_i |d3N(x_i)|^2
! (Huber value loss against one gross outlier, plus the Sobolev-type
! third-derivative penalty), trained with plain gradient descent.
! The quadratic value loss is run with the same data and penalty for
! comparison: the outlier drags it, while the Huber fit resists.
program example_customloss
  use multi_index_bell_module
  use net_module
  use train_module
  implicit none
  integer,parameter :: NP = 16
  real(8),parameter :: delta = 0.10d0, lam3 = 0.001d0, lr = 0.05d0, beta = 0.9d0
  integer :: dummy(1,1), dims(4), n, ep, icase, l, jj, ii
  real(8) :: rmse_quad = 0.d0, rmse_huber = 0.d0
  type(tabset_t) :: ts
  type(net_t) :: nt
  type(work_t) :: wk
  type(twork_t) :: tw
  type(grad_t) :: g
  real(8) :: x(NP), y(NP), tv(4), seed(4), r, rmse, w0saved(4*12*12)
  real(8),allocatable :: wsave(:,:,:), vel(:,:,:)

  call tabset_init( ts, 1, 3, 0, dummy )     ! D0=1, K=3 dense: slots 0..3
  dims = (/ 1, 10, 10, 1 /)
  call net_init( nt, 4, dims, ts )
  call work_init( wk, nt );  call twork_init( tw, nt );  call grad_init( g, nt )
  ! The library leaves the weights to the caller (Init_w is a trainer
  ! concern); a deterministic Glorot-like fill breaks the symmetry.
  do l=2,nt%nlayer
     do jj=1,nt%ndim(l)
        do ii=0,nt%ndim(l-1)
           nt%w(l,jj,ii) = 0.7d0*sin( dble(97*l+13*jj+29*ii) ) &
                           /sqrt( dble(nt%ndim(l-1)+1) )
        end do
     end do
  end do
  allocate( wsave(nt%nlayer,nt%ndmax,0:nt%ndmax) );  wsave = nt%w
  allocate( vel(nt%nlayer,nt%ndmax,0:nt%ndmax) );    vel = 0.d0

  do n=1,NP                                  ! noisy parabola + one outlier
     x(n) = -1.d0 + 2.d0*(n-1)/dble(NP-1)
     y(n) = x(n)**2 + 1.d0 + 0.02d0*sin( 37.d0*n )
  end do
  y(6) = y(6) + 2.d0                         ! gross outlier

  do icase=1,2
     nt%w = wsave;  vel = 0.d0
     do ep=1,20000
        call grad_zero( g )
        do n=1,NP
           call net_eval_hod( nt, wk, (/ x(n) /), tv )
           r = tv(1) - y(n)
           seed = 0.d0
           if ( icase==1 ) then
              seed(1) = r                            ! quadratic:  d(r^2/2)/dT
           else
              seed(1) = max( -delta, min( delta, r ) ) ! Huber:  clipped residual
           end if
           seed(4) = 2.d0*lam3*tv(4)                 ! d(lam3*T3^2)/dT3
           call net_grad_point( nt, tw, (/ x(n) /), seed, g )
        end do
        ! heavy-ball step on the mean gradient
        vel  = beta*vel - lr*( g%nabla/dble(NP) )
        nt%w = nt%w + vel
     end do
     rmse = 0.d0
     do n=1,NP
        if ( n==6 ) cycle                    ! judge against the clean truth
        call net_eval_hod( nt, wk, (/ x(n) /), tv )
        rmse = rmse + ( tv(1) - (x(n)**2+1.d0) )**2
     end do
     rmse = sqrt( rmse/dble(NP-1) )
     if ( icase==1 ) then
        write(*,"(a,f8.4)") "quadratic value loss (trainer menu):  clean-RMSE =", rmse
        rmse_quad = rmse
     else
        write(*,"(a,f8.4)") "Huber value loss (custom seed)     :  clean-RMSE =", rmse
        rmse_huber = rmse
     end if
  end do
  write(*,"(a)") "same library, same propagation: only the seed changed."

  ! Release what the demonstration built: this file is a model of the
  ! seed interface, so it shows the teardown as well as the setup.
  call grad_free( g );  call twork_free( tw );  call work_free( wk )
  call net_free( nt );  call tabset_free( ts )

  ! The claim of the example is that the Huber seed resists the outlier
  ! that drags the quadratic seed.  If that ordering ever reverses, the
  ! demonstration is no longer demonstrating anything.
  if ( rmse_huber < rmse_quad ) then
     write(*,"(a)") "customloss_example: ALL PASSED"
  else
     write(*,"(a)") "customloss_example: FAILED, the Huber seed did not resist"
     stop 1
  end if
end program
