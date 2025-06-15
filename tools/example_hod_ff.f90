!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (example_hod_ff.f90) is part of DNNF90.
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
! (MIT License; see LICENSE at the repository root.)
!
! Demonstration: a machine learning force field trained on energies,
! forces AND analytic Hessians, which is the capability that
! distinguishes DNNF90 from descriptor pipelines such as n2p2 (those
! provide dG/dR for force training but not d2G/dR2, and their networks
! do not propagate second derivatives).
!
! System      one dimensional Morse chain, all pairs inside a cutoff
! Descriptors n2p2 compatible radial symmetry functions (symfunc_module)
! Network     one atomic network E = sum_i N(G(i)), dense tables K=2
! Losses      (a) energies + forces          (baseline, what n2p2 can do)
!             (b) energies + forces + Hessians
! Validation  force constants and phonon frequencies of the equilibrium
!             chain against the analytic Morse results, after checking
!             every hand-written derivative of this file against central
!             differences (verify_chain_rule)
!
! Caveat      the cutoff used is the cosine one, whose second derivative
!             jumps at rc.  That is admissible here only because no pair
!             of the sampled configurations comes near rc, which the file
!             checks; a production coupling wants a C^2 cutoff.
!
! The Hessian of the model,
!   H_pq = sum_i [ sum_ab N_ab(G_i) dG_a/dx_p dG_b/dx_q
!                + sum_a  N_a(G_i)  d2G_a/dx_p dx_q ],
! and the seeding of dL/dN_a and dL/dN_ab both come from the same carried
! derivative slots, so Hessian training costs one ordinary adjoint pass
! per atom and configuration.
!
! Run from anywhere (self contained, no input files):  ./hod_ff_example.out
PROGRAM example_hod_ff
  use multi_index_bell_module, only: init_hod_tables, NUM_alpha, alpha_list
  use net_module
  use train_module
  use symfunc_module
  use kalman_module, only: kalman_t, kf_init, kf_free, kf_update_grad, &
       kf_num_weights
  implicit none

  ! ---- system: Morse chain ----
  integer,parameter :: NA = 6          ! atoms
  integer,parameter :: MTRAIN = 60     ! training configurations
  real(8),parameter :: DM = 1.d0, AM = 1.5d0, R0 = 1.d0, RCUT = 2.6d0
  real(8),parameter :: DISP = 0.10d0   ! large displacement amplitude
  real(8),parameter :: DISPS = 0.03d0  ! small displacements near equilibrium

  ! Observation scales of the filter.  The energy is the reference at 1;
  ! forces and Hessian entries are larger in magnitude, and without a
  ! scale the Hessian observations would dominate the covariance.
  real(8),parameter :: SC_F = 0.3d0
  real(8),parameter :: SC_H = 0.1d0
  integer,parameter :: KSWEEP = 40

  ! Which observables each filtered network sees.  The absolute energy of
  ! a force field has no physical meaning: it is fixed by the choice of
  ! reference, and only differences between configurations matter.  Forces
  ! determine the energy surface up to exactly that one constant, so a fit
  ! without the energy observable loses nothing physical.  These switches
  ! exist to test that statement: set them and compare.
  !               energy   force   Hessian
  logical,parameter :: OBS1(3) = (/ .true.,  .true., .false. /)   ! E+F
  logical,parameter :: OBS2(3) = (/ .true.,  .true., .true.  /)   ! E+F+H

  ! ---- descriptors ----
  integer,parameter :: NG = 3
  type(symset_t) :: ss

  ! ---- network ----
  integer,parameter :: NL = 4
  integer :: dims(NL) = (/ NG, 10, 10, 1 /)

  type(net_t)   :: netEF, netH, netKEF, netKH
  type(kalman_t) :: kfEF, kfH
  type(work_t)  :: wk
  type(twork_t) :: tw
  type(grad_t)  :: gEF, gH, gK

  real(8) :: xtr(NA,MTRAIN), etr(MTRAIN), ftr(NA,MTRAIN), htr(NA,NA,MTRAIN)
  real(8) :: x0(NA), h_ref(NA,NA), h_fit(NA,NA)
  real(8) :: w_ref(NA), w_fit_EF(NA), w_fit_H(NA)
  integer :: n, ep, dummy(1,1), islot0, islot1(NG), islot2(NG,NG)
  real(8) :: r1, r2, eEF, eH, fEF, fH, hEF, hH
  real(8) :: r3, r4, eKEF, eKH, fKEF, fKH, hKEF, hKH
  real(8) :: w_kef(NA), w_kh(NA), ov1, ov2
  integer :: sweep

  ! ---- tables: dense, three inputs, order two ----
  call init_hod_tables( NG, 2, 0, dummy )
  call locate_slots( islot0, islot1, islot2 )

  ss%ng = NG
  allocate( ss%eta(NG), ss%rs(NG) )
  ss%eta = (/ 4.d0, 4.d0, 1.5d0 /)
  ss%rs  = (/ 1.0d0, 1.6d0, 0.0d0 /)
  ss%rc  = RCUT

  ! ---- reference data from the analytic Morse chain ----
  do n=1,NA
     x0(n) = dble(n-1)*R0
  end do
  call gen_data
  call morse_hessian( x0, h_ref )

  ! ---- two networks from the same deterministic start ----
  call net_init( netEF, NL, dims );  call det_init( netEF )
  call net_init( netH,  NL, dims );  call det_init( netH )
  call work_init( wk, netEF )
  call twork_init( tw, netEF )
  call grad_init( gEF, netEF )
  call grad_init( gH,  netH )

  ! Two more networks from the same deterministic start, trained by the
  ! filter instead of by gradient descent: one on energies and forces,
  ! one that also sees the Hessians.
  call net_init( netKEF, NL, dims );  call det_init( netKEF )
  call net_init( netKH,  NL, dims );  call det_init( netKH )
  call grad_init( gK, netKEF )
  call kf_init( kfEF, netKEF, 1.d-2, 0.995d0, 0.9999d0 )
  call kf_init( kfH,  netKH,  1.d-4, 0.999d0, 0.99999d0 )

  ! ---- training ----
  do ep=1,12000
     r1 = merge( 2.d-3, 3.d-4, ep <= 8000 )     ! two-phase learning rate
     call grad_zero( gEF )
     call epoch_pass( netEF, gEF, 0.d0 )        ! energies + forces only
     call opt_adam_step( netEF, gEF, r1, 0.9d0, 0.999d0, 1.d-8, MTRAIN, ep )
     call grad_zero( gH )
     call epoch_pass( netH, gH, 0.2d0 )         ! + Hessians (per-element weight)
     call opt_adam_step( netH, gH, r1, 0.9d0, 0.999d0, 1.d-8, MTRAIN, ep )
  end do

  ! ---- the same two fits by the extended Kalman filter ----
  ! One sweep presents every observable of every configuration once, so a
  ! handful of sweeps replaces the thousands of epochs the gradient
  ! method needs.
  do sweep=1,KSWEEP
     call kalman_pass( netKEF, kfEF, gK, OBS1, ov1 )
     call kalman_pass( netKH,  kfH,  gK, OBS2, ov2 )
  end do
  write(*,'(a,f8.5)') "worst mode overlap of the last sweep : ", ov2

  ! ---- the chain rule of this file, against finite differences ----
  ! Everything below rests on three hand-written derivative assemblies:
  ! the descriptor derivatives, the model Hessian built from the carried
  ! slots, and the seed of the Hessian loss.  A wrong factor in any of
  ! them still trains to a plausible looking force field, so each is
  ! checked against the definition before the physics is reported.
  call verify_chain_rule( netH )
  call verify_eig_seed( netKH, gK )

  ! ---- validation on the equilibrium chain ----
  call model_errors( netEF, eEF, fEF )
  call model_errors( netH,  eH,  fH )
  hEF = hessian_rms( netEF )
  hH  = hessian_rms( netH )
  write(*,'(a)') "training-set RMS errors        dE(shifted)      force     Hessian"
  write(*,'(a,3e12.3)') "  E+F trained (n2p2-style)      :", eEF, fEF, hEF
  write(*,'(a,3e12.3)') "  E+F+H trained (this work)     :", eH,  fH, hH
  call model_errors( netKEF, eKEF, fKEF );  hKEF = hessian_rms( netKEF )
  call model_errors( netKH,  eKH,  fKH  );  hKH  = hessian_rms( netKH )
  write(*,'(a,3e12.3)') "  E+F   by Kalman filter        :", eKEF, fKEF, hKEF
  write(*,'(a,3e12.3)') "  E+F+H by Kalman filter        :", eKH,  fKH,  hKH

  call phonons( h_ref, w_ref )
  call model_hessian( netEF, x0, h_fit ); call phonons( h_fit, w_fit_EF )
  r1 = maxval( abs( h_fit - h_ref ) ) / maxval( abs( h_ref ) )
  call model_hessian( netH,  x0, h_fit ); call phonons( h_fit, w_fit_H )
  r2 = maxval( abs( h_fit - h_ref ) ) / maxval( abs( h_ref ) )

  write(*,'(a)') "equilibrium force constants, max relative error:"
  write(*,'(a,f8.4)') "  E+F trained   : ", r1
  write(*,'(a,f8.4)') "  E+F+H trained : ", r2
  call model_hessian( netKEF, x0, h_fit ); call phonons( h_fit, w_kef )
  r3 = maxval( abs( h_fit - h_ref ) ) / maxval( abs( h_ref ) )
  call model_hessian( netKH,  x0, h_fit ); call phonons( h_fit, w_kh )
  r4 = maxval( abs( h_fit - h_ref ) ) / maxval( abs( h_ref ) )
  write(*,'(a,f8.4)') "  E+F   Kalman  : ", r3
  write(*,'(a,f8.4)') "  E+F+H Kalman  : ", r4
  write(*,'(a)') "phonon frequencies (analytic | E+F | E+F+H):"
  do n=2,NA
     write(*,'(2x,3f12.6)') w_ref(n), w_fit_EF(n), w_fit_H(n)
  end do
  ! artifacts for the n2p2 bridge (tools/example_n2p2_bridge.cpp)
  call net_save( netH, 'ff_weight.dat' )
  call write_bridge_ref
  write(*,'(a,f7.2,a,f7.2,a)') "max phonon error: E+F ", &
       100.d0*maxval(abs(w_fit_EF(2:NA)-w_ref(2:NA))/w_ref(2:NA)), &
       " %   E+F+H ", &
       100.d0*maxval(abs(w_fit_H(2:NA)-w_ref(2:NA))/w_ref(2:NA)), " %"

CONTAINS

  !> Central differences against the three assemblies of this file.
  SUBROUTINE verify_chain_rule( nt )
    implicit none
    type(net_t),intent(INOUT) :: nt
    real(8),parameter :: hx = 1.d-5, hw = 1.d-6
    real(8) :: xp(NA), Gp(NG), Gm(NG), G0(NG)
    real(8) :: dGd(NG,NA), d2Gd(NG,NA,NA), dGp(NG,NA), dGm(NG,NA)
    real(8) :: d2scr(NG,NA,NA)   ! second derivatives of the perturbed point
    real(8) :: hnum(NA,NA), hana(NA,NA)
    real(8) :: e1, e2, wsave, num, ana, dmax, dscale
    integer :: i, k, p, q, l, jj, ii, nfail
    type(grad_t) :: gchk

    nfail = 0
    write(*,'(a)') "chain-rule checks (central differences)"

    ! (1) descriptor derivatives of one atom against differences of G
    dmax = 0.d0; dscale = 0.d0
    call symfunc_eval( ss, NA, xtr(:,3), 2, G0, dGd, d2Gd )
    do p=1,NA
       xp = xtr(:,3); xp(p) = xp(p) + hx
       call symfunc_eval( ss, NA, xp, 2, Gp, dGp, d2scr )
       xp = xtr(:,3); xp(p) = xp(p) - hx
       call symfunc_eval( ss, NA, xp, 2, Gm, dGm, d2scr )
       do k=1,NG
          num = ( Gp(k) - Gm(k) )/(2.d0*hx)
          dmax = max( dmax, abs( num - dGd(k,p) ) )
          dscale = max( dscale, abs(dGd(k,p)) )
       end do
       do k=1,NG
          do q=1,NA
             num = ( dGp(k,q) - dGm(k,q) )/(2.d0*hx)
             dmax = max( dmax, abs( num - d2Gd(k,q,p) ) )
             dscale = max( dscale, abs(d2Gd(k,q,p)) )
          end do
       end do
    end do
    write(*,'(a,e11.3)') "  dG/dx and d2G/dx2 vs differences of G : ", &
         dmax/max(dscale,1.d-12)
    if ( dmax/max(dscale,1.d-12) > 1.d-6 ) nfail = nfail + 1

    ! (2) the model Hessian against differences of the model energy
    call model_hessian( nt, xtr(:,3), hana )
    do p=1,NA
       do q=1,NA
          xp = xtr(:,3); xp(p) = xp(p) + hx; xp(q) = xp(q) + hx
          e1 = model_energy( nt, xp )
          xp = xtr(:,3); xp(p) = xp(p) + hx; xp(q) = xp(q) - hx
          e2 = model_energy( nt, xp )
          hnum(p,q) = e1 - e2
          xp = xtr(:,3); xp(p) = xp(p) - hx; xp(q) = xp(q) + hx
          hnum(p,q) = hnum(p,q) - model_energy( nt, xp )
          xp = xtr(:,3); xp(p) = xp(p) - hx; xp(q) = xp(q) - hx
          hnum(p,q) = ( hnum(p,q) + model_energy( nt, xp ) )/(4.d0*hx*hx)
       end do
    end do
    write(*,'(a,e11.3)') "  model Hessian vs differences of E     : ", &
         maxval(abs(hnum-hana))/max(maxval(abs(hana)),1.d-12)
    if ( maxval(abs(hnum-hana))/max(maxval(abs(hana)),1.d-12) > 1.d-5 ) &
         nfail = nfail + 1

    ! (3) the seeded gradient of the Hessian loss against differences of L
    call grad_init( gchk, nt )
    call grad_zero( gchk )
    call epoch_pass( nt, gchk, 0.2d0 )
    dmax = 0.d0; dscale = 0.d0
    do l=2,nt%nlayer
       do jj=1,nt%ndim(l)
          do ii=0,nt%ndim(l-1)
             wsave = nt%w(l,jj,ii)
             nt%w(l,jj,ii) = wsave + hw
             e1 = loss_value( nt, 0.2d0 )
             nt%w(l,jj,ii) = wsave - hw
             e2 = loss_value( nt, 0.2d0 )
             nt%w(l,jj,ii) = wsave
             num = ( e1 - e2 )/(2.d0*hw)
             ana = gchk%nabla(l,jj,ii)
             dmax = max( dmax, abs(num-ana) )
             dscale = max( dscale, abs(ana) )
          end do
       end do
    end do
    write(*,'(a,e11.3)') "  dL/dw with Hessian term vs diff of L  : ", &
         dmax/max(dscale,1.d-12)
    if ( dmax/max(dscale,1.d-12) > 1.d-6 ) nfail = nfail + 1
    call grad_free( gchk )

    ! (4) the cutoff function used here is the cosine one, whose second
    ! derivative jumps at rc: fc''(rc) = +0.5(pi/rc)^2 while fc'' = 0
    ! beyond.  A Hessian-trained model is therefore only C^1 across the
    ! cutoff, and this demonstration is only sound because no pair of the
    ! training set comes near rc.  That is an invariant of the sampling,
    ! not of the method, so it is checked rather than assumed.  A
    ! production coupling wants a C^2 cutoff, for instance a polynomial
    ! or the tanh^3 form.
    dmax = huge(1.d0)
    do i=1,MTRAIN
       do p=1,NA-1
          do q=p+1,NA
             e1 = abs( xtr(p,i) - xtr(q,i) )
             if ( e1 < ss%rc ) dmax = min( dmax, ss%rc - e1 )
             if ( e1 > ss%rc ) dmax = min( dmax, e1 - ss%rc )
          end do
       end do
    end do
    write(*,'(a,f9.4)') "  closest approach of any pair to rc    : ", dmax
    if ( dmax < 0.1d0 ) then
       write(*,'(a)') "  a pair sits within 0.1 of the cutoff: with the"
       write(*,'(a)') "  cosine cutoff the model Hessian is discontinuous"
       write(*,'(a)') "  there, so this sampling needs a C^2 cutoff"
       nfail = nfail + 1
    end if

    if ( nfail == 0 ) then
       write(*,'(a)') "  ALL PASSED"
    else
       write(*,'(a,i0,a)') "  ", nfail, " CHECK(S) FAILED"
       stop
    end if
  END SUBROUTINE verify_chain_rule

  !> Model energy of one configuration.
  REAL(8) FUNCTION model_energy( nt, x )
    implicit none
    type(net_t),intent(IN) :: nt
    real(8),intent(IN) :: x(NA)
    real(8) :: Gd(NG), dGd(NG,NA), d2Gd(NG,NA,NA), tval(NUM_alpha)
    integer :: i
    model_energy = 0.d0
    do i=1,NA
       call symfunc_eval( ss, NA, x, i, Gd, dGd, d2Gd )
       call net_eval_hod( nt, wk, Gd, tval )
       model_energy = model_energy + tval(islot0)
    end do
  END FUNCTION model_energy

  !> The loss epoch_pass differentiates, evaluated directly.
  REAL(8) FUNCTION loss_value( nt, wh )
    implicit none
    type(net_t),intent(IN) :: nt
    real(8),intent(IN) :: wh
    real(8),parameter :: we = 1.d0, wf = 1.d0
    real(8) :: Gd(NG), dGd(NG,NA), d2Gd(NG,NA,NA), tval(NUM_alpha)
    real(8) :: emod, fmod(NA), hmod(NA,NA), L
    integer :: mcfg, i, a, b, p, q
    L = 0.d0
    do mcfg=1,MTRAIN
       emod = 0.d0; fmod = 0.d0; hmod = 0.d0
       do i=1,NA
          call symfunc_eval( ss, NA, xtr(:,mcfg), i, Gd, dGd, d2Gd )
          call net_eval_hod( nt, wk, Gd, tval )
          emod = emod + tval(islot0)
          do a=1,NG
             fmod(:) = fmod(:) - tval(islot1(a))*dGd(a,:)
          end do
          if ( wh > 0.d0 ) then
             do p=1,NA
                do q=1,NA
                   do a=1,NG
                      hmod(p,q) = hmod(p,q) + tval(islot1(a))*d2Gd(a,p,q)
                      do b=1,NG
                         hmod(p,q) = hmod(p,q) &
                              + tval(islot2(a,b))*dGd(a,p)*dGd(b,q)
                      end do
                   end do
                end do
             end do
          end if
       end do
       L = L + 0.5d0*we*( emod - etr(mcfg) )**2 &
             + 0.5d0*wf*sum( ( fmod - ftr(:,mcfg) )**2 )
       if ( wh > 0.d0 ) &
          L = L + 0.5d0*wh*sum( ( hmod - htr(:,:,mcfg) )**2 )
    end do
    loss_value = L
  END FUNCTION loss_value

  !> Reference for the mixed n2p2/DNNF90 binary: descriptors, value and
  !! derivative slots of atom 1 at equilibrium, 17 digits.
  SUBROUTINE write_bridge_ref
    implicit none
    real(8) :: Gd(NG), dGd(NG,NA), d2Gd(NG,NA,NA), tval(NUM_alpha)
    integer :: ur
    call symfunc_eval( ss, NA, x0, 1, Gd, dGd, d2Gd )
    call net_eval_hod( netH, wk, Gd, tval )
    ur = 90
    open( ur, file='ff_bridge_ref.dat', status='replace' )
    write(ur,'(2i6)') NG, NUM_alpha
    write(ur,'(100e26.17)') ss%eta(1:NG), ss%rs(1:NG), ss%rc
    write(ur,'(100e26.17)') Gd(1:NG)
    write(ur,'(100e26.17)') tval(1:NUM_alpha)
    close(ur)
  END SUBROUTINE write_bridge_ref

  !> One pass over all configurations: accumulate the gradient of the
  !! loss with energy weight 1, force weight 1 and Hessian weight wh.
  SUBROUTINE epoch_pass( nt, g, wh )
    implicit none
    type(net_t),intent(IN) :: nt
    type(grad_t),intent(INOUT) :: g
    real(8),intent(IN) :: wh
    real(8),parameter :: we = 1.d0, wf = 1.d0
    real(8) :: Gd(NG), dGd(NG,NA), d2Gd(NG,NA,NA)
    real(8) :: Gall(NG,NA), dGall(NG,NA,NA), d2Gall(NG,NA,NA,NA)
    real(8) :: tval(NUM_alpha), tall(NUM_alpha,NA), seed(NUM_alpha)
    real(8) :: emod, fmod(NA), hmod(NA,NA)
    real(8) :: re, rf(NA), rh(NA,NA), rr
    integer :: mcfg, i, a, b, p, q

    do mcfg=1,MTRAIN
       ! descriptors and model E, F, H of this configuration
       emod = 0.d0; fmod = 0.d0; hmod = 0.d0
       do i=1,NA
          call symfunc_eval( ss, NA, xtr(:,mcfg), i, Gd, dGd, d2Gd )
          Gall(:,i) = Gd; dGall(:,i,:) = dGd; d2Gall(:,i,:,:) = d2Gd
          call net_eval_hod( nt, wk, Gd, tval )
          tall(:,i) = tval
          emod = emod + tval(islot0)
          do a=1,NG
             fmod(:) = fmod(:) - tval(islot1(a))*dGd(a,:)
          end do
          do p=1,NA
             do q=1,NA
                do a=1,NG
                   hmod(p,q) = hmod(p,q) + tval(islot1(a))*d2Gd(a,p,q)
                   do b=1,NG
                      hmod(p,q) = hmod(p,q) &
                           + tval(islot2(a,b))*dGd(a,p)*dGd(b,q)
                   end do
                end do
             end do
          end do
       end do
       re = emod - etr(mcfg)
       rf = fmod - ftr(:,mcfg)
       rh = hmod - htr(:,:,mcfg)

       ! seeds per atom and one adjoint pass each
       do i=1,NA
          seed = 0.d0
          seed(islot0) = we*re
          do a=1,NG
             seed(islot1(a)) = -wf*dot_product( rf, dGall(a,i,:) )
             if ( wh > 0.d0 ) then
                do p=1,NA
                   seed(islot1(a)) = seed(islot1(a)) &
                        + wh*dot_product( rh(p,:), d2Gall(a,i,p,:) )
                end do
                do b=a,NG
                   ! slot alpha = e_a + e_b collects both ordered pairs
                   rr = 0.d0
                   do p=1,NA
                      rr = rr + dot_product( rh(p,:), dGall(b,i,:) )*dGall(a,i,p)
                      if ( b /= a ) &
                        rr = rr + dot_product( rh(p,:), dGall(a,i,:) )*dGall(b,i,p)
                   end do
                   seed(islot2(a,b)) = wh*rr
                end do
             end if
          end do
          call net_grad_point( nt, tw, Gall(:,i), seed, g )
       end do
    end do
  END SUBROUTINE epoch_pass

  SUBROUTINE model_errors( nt, erms, frms )
    implicit none
    type(net_t),intent(IN) :: nt
    real(8),intent(OUT) :: erms, frms
    real(8) :: Gd(NG), dGd(NG,NA), d2Gd(NG,NA,NA), tval(NUM_alpha)
    real(8) :: emod, fmod(NA)
    integer :: mcfg, i, a
    real(8) :: dev(MTRAIN), dbar
    erms = 0.d0; frms = 0.d0
    do mcfg=1,MTRAIN
       emod = 0.d0; fmod = 0.d0
       do i=1,NA
          call symfunc_eval( ss, NA, xtr(:,mcfg), i, Gd, dGd, d2Gd )
          call net_eval_hod( nt, wk, Gd, tval )
          emod = emod + tval(islot0)
          do a=1,NG
             fmod(:) = fmod(:) - tval(islot1(a))*dGd(a,:)
          end do
       end do
       dev(mcfg) = emod - etr(mcfg)
       frms = frms + sum( ( fmod - ftr(:,mcfg) )**2 )
    end do
    ! The absolute energy of a force field is fixed by the choice of
    ! reference and carries no physics; only differences between
    ! configurations do.  Removing the mean of the residual measures
    ! exactly those differences, and is the error a fit without an energy
    ! observable should be judged by: forces determine the surface up to
    ! that one constant and nothing else.
    dbar = sum( dev )/dble(MTRAIN)
    erms = sqrt( sum( ( dev - dbar )**2 )/dble(MTRAIN) )
    frms = sqrt( frms/dble(MTRAIN*NA) )
  END SUBROUTINE model_errors

  !> RMS residual of the model Hessian over the training configurations.
  !! This is the quantity the Hessian term of the loss minimizes, so it
  !! is the direct test of whether that term does what it claims.
  REAL(8) FUNCTION hessian_rms( nt )
    implicit none
    type(net_t),intent(IN) :: nt
    real(8) :: hm(NA,NA), s
    integer :: mcfg
    s = 0.d0
    do mcfg=1,MTRAIN
       call model_hessian( nt, xtr(:,mcfg), hm )
       s = s + sum( ( hm - htr(:,:,mcfg) )**2 )
    end do
    hessian_rms = sqrt( s/dble(MTRAIN*NA*NA) )
  END FUNCTION hessian_rms

  SUBROUTINE model_hessian( nt, x, h )
    implicit none
    type(net_t),intent(IN) :: nt
    real(8),intent(IN) :: x(NA)
    real(8),intent(OUT) :: h(NA,NA)
    real(8) :: Gd(NG), dGd(NG,NA), d2Gd(NG,NA,NA), tval(NUM_alpha)
    integer :: i, a, b, p, q
    h = 0.d0
    do i=1,NA
       call symfunc_eval( ss, NA, x, i, Gd, dGd, d2Gd )
       call net_eval_hod( nt, wk, Gd, tval )
       do p=1,NA
          do q=1,NA
             do a=1,NG
                h(p,q) = h(p,q) + tval(islot1(a))*d2Gd(a,p,q)
                do b=1,NG
                   h(p,q) = h(p,q) + tval(islot2(a,b))*dGd(a,p)*dGd(b,q)
                end do
             end do
          end do
       end do
    end do
  END SUBROUTINE model_hessian

  !> Check the eigenvalue seed against central differences.
  !!
  !! The filter observes eigenvalues of the model Hessian, so it needs
  !! d lambda_k / dw.  That is one more hand-written derivative assembly,
  !! and like the three checked in verify_chain_rule it would train to a
  !! plausible looking model if a factor were wrong.  Here the adjoint
  !! built from the eigenvalue seed is compared with differences of the
  !! eigenvalue itself.
  SUBROUTINE verify_eig_seed( nt, g )
    implicit none
    type(net_t),intent(INOUT) :: nt
    type(grad_t),intent(INOUT) :: g
    real(8),parameter :: hw = 1.d-6
    real(8) :: Gall(NG, NA), dGall(NG, NA, NA), d2Gall(NG, NA, NA, NA)
    real(8) :: seed(NUM_alpha), cv(NG), qf
    real(8) :: hm(NA, NA), lam(NA), vec(NA, NA)
    real(8) :: lp(NA), lm(NA), vdum(NA, NA), wsave, num, ana, emax, gmax
    integer :: i, a, b, p, l, jj, ii, kmode

    kmode = 3                     ! an interior mode, away from the edges
    call setup_cfg( nt, 1, Gall, dGall, d2Gall )
    call model_hessian( nt, xtr(:,1), hm )
    call jacobi_eig( hm, lam, vec )

    ! analytic: accumulate the adjoint of lambda_kmode over the atoms
    call grad_zero( g )
    do i=1,NA
       do a=1,NG
          cv(a) = dot_product( vec(:,kmode), dGall(a,i,:) )
       end do
       seed = 0.d0
       do a=1,NG
          qf = 0.d0
          do p=1,NA
             qf = qf + vec(p,kmode) &
                  *dot_product( d2Gall(a,i,p,:), vec(:,kmode) )
          end do
          seed(islot1(a)) = qf
          do b=1,NG
             seed(islot2(a,b)) = seed(islot2(a,b)) + cv(a)*cv(b)
          end do
       end do
       call net_grad_point( nt, tw, Gall(:,i), seed, g )
    end do

    emax = 0.d0;  gmax = 0.d0
    do l=2,nt%nlayer
       do jj=1,nt%ndim(l)
          do ii=0,nt%ndim(l-1)
             wsave = nt%w(l,jj,ii)
             nt%w(l,jj,ii) = wsave + hw
             call model_hessian( nt, xtr(:,1), hm )
             call jacobi_eig( hm, lp, vdum )
             nt%w(l,jj,ii) = wsave - hw
             call model_hessian( nt, xtr(:,1), hm )
             call jacobi_eig( hm, lm, vdum )
             nt%w(l,jj,ii) = wsave
             num = ( lp(kmode) - lm(kmode) )/(2.d0*hw)
             ana = g%nabla(l,jj,ii)
             emax = max( emax, abs(num-ana) )
             gmax = max( gmax, abs(ana) )
          end do
       end do
    end do
    write(*,'(a,e12.4)') "  d lambda / dw vs differences of lambda : ", &
         emax/max(gmax,1.d-12)
    write(*,'(a,6f10.5)') "  model spectrum of configuration 1     : ", lam
  END SUBROUTINE verify_eig_seed

  !> descriptors of one configuration, shared by the checks and the sweep
  SUBROUTINE setup_cfg( nt, mcfg, Gall, dGall, d2Gall )
    implicit none
    type(net_t),intent(IN) :: nt
    integer,intent(IN) :: mcfg
    real(8),intent(OUT) :: Gall(NG,NA), dGall(NG,NA,NA), d2Gall(NG,NA,NA,NA)
    real(8) :: Gd(NG), dGd(NG, NA), d2Gd(NG, NA, NA)
    integer :: i
    do i=1,NA
       call symfunc_eval( ss, NA, xtr(:,mcfg), i, Gd, dGd, d2Gd )
       Gall(:,i) = Gd; dGall(:,i,:) = dGd; d2Gall(:,i,:,:) = d2Gd
    end do
  END SUBROUTINE setup_cfg

  !> Pair the modes of two spectra by eigenvector overlap.
  !!
  !! Sorting the eigenvalues and pairing by position is wrong here.  The
  !! model Hessian of a partly trained network can carry a negative
  !! eigenvalue, which the ascending sort puts first and which pushes the
  !! translation mode out of slot one; and two modes of the model can
  !! cross during training, after which position k of the model is a
  !! different physical mode from position k of the target.  Either way the
  !! filter is handed a residual between unrelated modes.
  !!
  !! The translation mode is identified by its overlap with the uniform
  !! vector rather than by its position, and excluded on both sides.  The
  !! remaining modes are paired greedily by |v_model . v_target|, which is
  !! sign independent as eigenvectors are only defined up to sign.  The
  !! greedy pass is not guaranteed to be the optimal assignment, but it is
  !! deterministic and the overlaps here are close to one; the smallest
  !! overlap of the pairing is reported so a poor match is visible rather
  !! than silent.
  SUBROUTINE match_modes( vec_m, vec_t, map, itr_t, ovmin )
    implicit none
    real(8),intent(IN) :: vec_m(NA,NA), vec_t(NA,NA)
    integer,intent(OUT) :: map(NA), itr_t
    real(8),intent(OUT) :: ovmin
    logical :: used(NA)
    real(8) :: uvec(NA), best, ov
    integer :: j, k, jbest, itr_m

    uvec = 1.d0/sqrt( dble(NA) )

    itr_t = 1;  best = -1.d0
    do j=1,NA
       ov = abs( dot_product( vec_t(:,j), uvec ) )
       if ( ov > best ) then
          best = ov;  itr_t = j
       end if
    end do
    itr_m = 1;  best = -1.d0
    do j=1,NA
       ov = abs( dot_product( vec_m(:,j), uvec ) )
       if ( ov > best ) then
          best = ov;  itr_m = j
       end if
    end do

    used = .false.
    used(itr_m) = .true.        ! the model's translation mode is not observed
    map = 0
    ovmin = 1.d0
    do k=1,NA
       if ( k == itr_t ) cycle
       best = -1.d0;  jbest = 0
       do j=1,NA
          if ( used(j) ) cycle
          ov = abs( dot_product( vec_m(:,j), vec_t(:,k) ) )
          if ( ov > best ) then
             best = ov;  jbest = j
          end if
       end do
       map(k) = jbest
       used(jbest) = .true.
       ovmin = min( ovmin, best )
    end do
  END SUBROUTINE match_modes

  !> One sweep of the extended Kalman filter over the training set.
  !!
  !! The filter presents one scalar observable at a time and applies a
  !! rank-one update, so it needs the Jacobian of that observable.  An
  !! observable of a force field is a sum over the atoms of a
  !! configuration, so the row is accumulated by calling net_grad_point
  !! once per atom with the seed of that observable and then handed to
  !! kf_update_grad with the residual.  The seeds are the same ones the
  !! gradient pass builds; only their use differs.
  !!
  !! Observables per configuration: the energy, the NA force components,
  !! and the upper triangle of the Hessian when wh_on is true.  The scale
  !! argument balances their magnitudes and plays the part the loss
  !! weights play in the gradient method.
  SUBROUTINE kalman_pass( nt, kf, g, obs, ovworst )
    implicit none
    type(net_t),intent(INOUT) :: nt
    type(kalman_t),intent(INOUT) :: kf
    type(grad_t),intent(INOUT) :: g
    !> obs(1) energy, obs(2) forces, obs(3) curvature
    logical,intent(IN) :: obs(3)
    !> Smallest eigenvector overlap of the mode pairing over this sweep.
    !! Close to one means the pairing is unambiguous; a small value warns
    !! that the model and target modes no longer correspond.
    real(8),intent(OUT) :: ovworst
    real(8) :: Gd(NG), dGd(NG,NA), d2Gd(NG,NA,NA)
    real(8) :: Gall(NG,NA), dGall(NG,NA,NA), d2Gall(NG,NA,NA,NA)
    real(8) :: tval(NUM_alpha), seed(NUM_alpha)
    real(8) :: emod, fmod(NA), hmod(NA,NA)
    real(8) :: lam_m(NA), lam_t(NA), vec_m(NA,NA), vec_t(NA,NA)
    real(8) :: cv(NG), qf, ovm
    integer :: mcfg, i, a, b, p, q, k, km, kmap(NA), itr_t

    ovworst = 1.d0
    do mcfg=1,MTRAIN
       ! ---- descriptors and the model E, F, H of this configuration ----
       emod = 0.d0; fmod = 0.d0; hmod = 0.d0
       do i=1,NA
          call symfunc_eval( ss, NA, xtr(:,mcfg), i, Gd, dGd, d2Gd )
          Gall(:,i) = Gd; dGall(:,i,:) = dGd; d2Gall(:,i,:,:) = d2Gd
          call net_eval_hod( nt, wk, Gd, tval )
          emod = emod + tval(islot0)
          do a=1,NG
             fmod(:) = fmod(:) - tval(islot1(a))*dGd(a,:)
          end do
          if ( obs(3) ) then
             do p=1,NA
                do q=1,NA
                   do a=1,NG
                      hmod(p,q) = hmod(p,q) + tval(islot1(a))*d2Gd(a,p,q)
                      do b=1,NG
                         hmod(p,q) = hmod(p,q) &
                              + tval(islot2(a,b))*dGd(a,p)*dGd(b,q)
                      end do
                   end do
                end do
             end do
          end if
       end do

       ! ---- the energy, when it is observed at all ----
       if ( obs(1) ) then
       call grad_zero( g )
       do i=1,NA
          seed = 0.d0
          seed(islot0) = 1.d0
          call net_grad_point( nt, tw, Gall(:,i), seed, g )
       end do
       call kf_update_grad( nt, kf, g, etr(mcfg) - emod, 1.d0 )
       end if

       ! ---- the force components, one short ----
       if ( obs(2) ) then
       ! sum_p f_p = 0 holds exactly for a translation invariant model, so
       ! the last component is determined by the others and presenting it
       ! would give the filter an exactly dependent observation.
       do p=1,NA-1
          call grad_zero( g )
          do i=1,NA
             seed = 0.d0
             do a=1,NG
                seed(islot1(a)) = -dGall(a,i,p)
             end do
             call net_grad_point( nt, tw, Gall(:,i), seed, g )
          end do
          call kf_update_grad( nt, kf, g, ftr(p,mcfg) - fmod(p), SC_F )
       end do
       end if

       ! ---- the curvature, observed through the spectrum ----
       ! The entries H_pq are not an independent set: translation
       ! invariance makes sum_q H_pq vanish exactly, and the model builds
       ! all of them from three first-order and six second-order slots per
       ! atom, so their Jacobian rows are strongly dependent.  A sequential
       ! filter loses the positive definiteness of its covariance on such
       ! observations, which is what the guard in kalman_module reports.
       !
       ! The eigenvalues are an independent set of the right size, and they
       ! are also the quantity the physics asks about: the phonon
       ! frequencies are their square roots.  For a simple eigenvalue,
       !   d lambda_k / d(anything) = v_k^T (dH/d anything) v_k,
       ! so the seed of observation k is the seed of H contracted twice
       ! with v_k.  Writing c_a = sum_p v_p dG_a/dx_p, the second-order
       ! part factorizes into c_a c_b and the first-order part is the
       ! quadratic form of d2G_a.
       if ( obs(3) ) then
          call jacobi_eig( hmod, lam_m, vec_m )
          call jacobi_eig( htr(:,:,mcfg), lam_t, vec_t )
          call match_modes( vec_m, vec_t, kmap, itr_t, ovm )
          ovworst = min( ovworst, ovm )
          do k=1,NA
             if ( k == itr_t ) cycle      ! the translation carries nothing
             km = kmap(k)                 ! the model mode that matches it
             call grad_zero( g )
             do i=1,NA
                do a=1,NG
                   cv(a) = dot_product( vec_m(:,km), dGall(a,i,:) )
                end do
                seed = 0.d0
                do a=1,NG
                   qf = 0.d0
                   do p=1,NA
                      qf = qf + vec_m(p,km) &
                           *dot_product( d2Gall(a,i,p,:), vec_m(:,km) )
                   end do
                   seed(islot1(a)) = qf
                   do b=1,NG
                      ! islot2(a,b) == islot2(b,a), so an off-diagonal pair
                      ! accumulates c_a c_b twice, which is the ordered sum
                      seed(islot2(a,b)) = seed(islot2(a,b)) + cv(a)*cv(b)
                   end do
                end do
                call net_grad_point( nt, tw, Gall(:,i), seed, g )
             end do
             call kf_update_grad( nt, kf, g, lam_t(k) - lam_m(km), SC_H )
          end do
       end if
    end do
  END SUBROUTINE kalman_pass

  ! ---------- reference physics ----------

  REAL(8) FUNCTION morse_e( x )
    implicit none
    real(8),intent(IN) :: x(NA)
    integer :: i, j
    real(8) :: r, u
    morse_e = 0.d0
    do i=1,NA-1
       do j=i+1,NA
          r = abs( x(i)-x(j) )
          if ( r < RCUT ) then
             u = 1.d0 - exp( -AM*(r-R0) )
             morse_e = morse_e + DM*( u*u - 1.d0 )
          end if
       end do
    end do
  END FUNCTION morse_e

  SUBROUTINE morse_fh( x, f, h )
    implicit none
    real(8),intent(IN) :: x(NA)
    real(8),intent(OUT) :: f(NA), h(NA,NA)
    integer :: i, j
    real(8) :: r, s, ex, de, d2e
    f = 0.d0;  h = 0.d0
    do i=1,NA-1
       do j=i+1,NA
          r = abs( x(i)-x(j) )
          if ( r < RCUT ) then
             s  = sign( 1.d0, x(i)-x(j) )
             ex = exp( -AM*(r-R0) )
             de  = 2.d0*DM*AM*ex*( 1.d0 - ex )       ! dV/dr
             d2e = 2.d0*DM*AM*AM*ex*( 2.d0*ex - 1.d0 ) ! d2V/dr2
             f(i) = f(i) - de*s
             f(j) = f(j) + de*s
             h(i,i) = h(i,i) + d2e
             h(j,j) = h(j,j) + d2e
             h(i,j) = h(i,j) - d2e
             h(j,i) = h(j,i) - d2e
          end if
       end do
    end do
  END SUBROUTINE morse_fh

  SUBROUTINE morse_hessian( x, h )
    implicit none
    real(8),intent(IN) :: x(NA)
    real(8),intent(OUT) :: h(NA,NA)
    real(8) :: f(NA)
    call morse_fh( x, f, h )
  END SUBROUTINE morse_hessian

  SUBROUTINE gen_data
    implicit none
    integer :: mcfg, n, iseed
    real(8) :: f(NA)
    iseed = 12345
    do mcfg=1,MTRAIN
       do n=1,NA
          ! phonon-aware sampling: the first configuration is the
          ! equilibrium chain itself, half of the rest stay close to it
          if ( mcfg == 1 ) then
             xtr(n,mcfg) = x0(n)
          else if ( mod(mcfg,2) == 0 ) then
             xtr(n,mcfg) = x0(n) + DISPS*( 2.d0*lcg(iseed) - 1.d0 )
          else
             xtr(n,mcfg) = x0(n) + DISP*( 2.d0*lcg(iseed) - 1.d0 )
          end if
       end do
       etr(mcfg) = morse_e( xtr(:,mcfg) )
       call morse_fh( xtr(:,mcfg), f, htr(:,:,mcfg) )
       ftr(:,mcfg) = f
    end do
  END SUBROUTINE gen_data

  REAL(8) FUNCTION lcg( is )
    implicit none
    integer,intent(INOUT) :: is
    is = mod( is*1103515245 + 12345, 2147483647 )
    if ( is < 0 ) is = is + 2147483647
    lcg = dble(is)/2147483647.d0
  END FUNCTION lcg

  SUBROUTINE det_init( nt )
    implicit none
    type(net_t),intent(INOUT) :: nt
    integer :: l, j, i
    do l=2,nt%nlayer
       do j=1,nt%ndim(l)
          do i=0,nt%ndim(l-1)
             nt%w(l,j,i) = 0.3d0*sin( 1.7d0*l + 0.9d0*j + 0.3d0*i ) + 0.02d0
          end do
       end do
    end do
  END SUBROUTINE det_init

  SUBROUTINE locate_slots( i0, i1, i2 )
    implicit none
    integer,intent(OUT) :: i0, i1(NG), i2(NG,NG)
    integer :: ia, a, b, alp(NG)
    i0 = 0; i1 = 0; i2 = 0
    do ia=1,NUM_alpha
       alp(1:NG) = alpha_list(1:NG,ia)
       if ( sum(alp) == 0 ) i0 = ia
       do a=1,NG
          if ( alp(a) == 1 .and. sum(alp) == 1 ) i1(a) = ia
          if ( alp(a) == 2 .and. sum(alp) == 2 ) then
             i2(a,a) = ia
          end if
          do b=a+1,NG
             if ( alp(a)==1 .and. alp(b)==1 .and. sum(alp)==2 ) then
                i2(a,b) = ia; i2(b,a) = ia
             end if
          end do
       end do
    end do
  END SUBROUTINE locate_slots

  !> Eigenfrequencies of a symmetric Hessian (unit masses), by Jacobi.
  !> Eigenvalues and eigenvectors of a symmetric matrix, by Jacobi.
  !! Needed because the filter observes eigenvalues, and the derivative
  !! of an eigenvalue needs its eigenvector.
  SUBROUTINE jacobi_eig( h, lam, vec )
    implicit none
    real(8),intent(IN) :: h(NA,NA)
    real(8),intent(OUT) :: lam(NA), vec(NA,NA)
    real(8) :: a(NA,NA), th, c, s, hp, hq
    integer :: p, q, k, sweep, ord(NA), i, j
    real(8) :: tmp(NA)
    a = h
    vec = 0.d0
    do p=1,NA
       vec(p,p) = 1.d0
    end do
    do sweep=1,100
       do p=1,NA-1
          do q=p+1,NA
             if ( abs(a(p,q)) < 1.d-14 ) cycle
             th = 0.5d0*atan2( 2.d0*a(p,q), a(q,q)-a(p,p) )
             c = cos(th); s = sin(th)
             do k=1,NA
                hp = a(k,p); hq = a(k,q)
                a(k,p) = c*hp - s*hq
                a(k,q) = s*hp + c*hq
             end do
             do k=1,NA
                hp = a(p,k); hq = a(q,k)
                a(p,k) = c*hp - s*hq
                a(q,k) = s*hp + c*hq
             end do
             do k=1,NA
                hp = vec(k,p); hq = vec(k,q)
                vec(k,p) = c*hp - s*hq
                vec(k,q) = s*hp + c*hq
             end do
          end do
       end do
    end do
    do p=1,NA
       lam(p) = a(p,p)
    end do
    ! ascending order, carrying the vectors with the values
    do i=1,NA
       ord(i) = i
    end do
    do i=1,NA-1
       do j=i+1,NA
          if ( lam(ord(j)) < lam(ord(i)) ) then
             k = ord(i); ord(i) = ord(j); ord(j) = k
          end if
       end do
    end do
    tmp = lam
    a   = vec
    do i=1,NA
       lam(i)   = tmp(ord(i))
       vec(:,i) = a(:,ord(i))
    end do
  END SUBROUTINE jacobi_eig

  SUBROUTINE phonons( h, w )
    implicit none
    real(8),intent(IN) :: h(NA,NA)
    real(8),intent(OUT) :: w(NA)
    real(8) :: a(NA,NA), th, c, s, hp, hq
    integer :: p, q, k, sweep
    a = h
    do sweep=1,100
       do p=1,NA-1
          do q=p+1,NA
             if ( abs(a(p,q)) < 1.d-14 ) cycle
             th = 0.5d0*atan2( 2.d0*a(p,q), a(q,q)-a(p,p) )
             c = cos(th); s = sin(th)
             do k=1,NA
                hp = a(k,p); hq = a(k,q)
                a(k,p) = c*hp - s*hq
                a(k,q) = s*hp + c*hq
             end do
             do k=1,NA
                hp = a(p,k); hq = a(q,k)
                a(p,k) = c*hp - s*hq
                a(q,k) = s*hp + c*hq
             end do
          end do
       end do
    end do
    do p=1,NA
       w(p) = sqrt( max( a(p,p), 0.d0 ) )
    end do
    call sort_asc( w )
  END SUBROUTINE phonons

  SUBROUTINE sort_asc( w )
    implicit none
    real(8),intent(INOUT) :: w(NA)
    integer :: p, q
    real(8) :: t
    do p=1,NA-1
       do q=p+1,NA
          if ( w(q) < w(p) ) then
             t = w(p); w(p) = w(q); w(q) = t
          end if
       end do
    end do
  END SUBROUTINE sort_asc

END PROGRAM example_hod_ff
