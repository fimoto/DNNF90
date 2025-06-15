!
!  DNNF90
!  Copyright (C) 2026  Fumihiro Imoto
!
!  This file (validation_module.f90) is part of DNNF90.
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
! Calc errors in order to update weights
MODULE validation_module

  use parallel_module
  use global_variables
  use multi_index_bell_module
  use io_module

  use pinn_module, only: calc_pinn_residual, calc_sys_residual
  use lib_net_module, only: lnet_forward_value, lnet_forward_hod, &
       lnet_forward_hod_multi, lnet_sync_weights
  implicit none

  PRIVATE
  PUBLIC :: alloc_for_validation
  PUBLIC :: prep_outfiles, perform_validation, write_outfile
  PUBLIC :: Ntrain_set
  PUBLIC :: Nval_set
  PUBLIC :: Cost_train_set,Cost_train
  PUBLIC :: Cost_val_set,Cost_val
  PUBLIC :: Cost_input_set,Cost_input
  !
  PUBLIC:: SE_train_set,SE_train
  PUBLIC :: SE_val_set,SE_val
  PUBLIC :: SE_input_set,SE_input
  !
  PUBLIC :: AE_train_set,AE_train
  PUBLIC :: AE_val_set,AE_val
  PUBLIC :: AE_input_set,AE_input
  
  
  
  
  !*** New variables for validation & weight update
  integer,allocatable :: Ntrain_set(:)
  integer,allocatable :: Nval_set(:)
  real(8),allocatable :: Cost_train_set(:,:),Cost_train(:)
  real(8),allocatable :: Cost_val_set(:,:),Cost_val(:)
  real(8),allocatable :: Cost_input_set(:,:),Cost_input(:)
  !
  real(8),allocatable :: SE_train_set(:,:),SE_train(:)
  real(8),allocatable :: SE_val_set(:,:),SE_val(:)
  real(8),allocatable :: SE_input_set(:,:),SE_input(:)
  !
  real(8),allocatable :: AE_train_set(:,:),AE_train(:)
  real(8),allocatable :: AE_val_set(:,:),AE_val(:)
  real(8),allocatable :: AE_input_set(:,:),AE_input(:)
  
  
  
CONTAINS
  
  SUBROUTINE alloc_for_validation
    implicit none
    allocate( Ntrain_set(Ntot_train_set) )
    allocate( Nval_set(Ntot_train_set) )
    allocate( Cost_train_set(Ntot_train_set,0:2),Cost_train(0:2) )
    allocate( Cost_val_set(Ntot_train_set,0:2),Cost_val(0:2) )
    allocate( Cost_input_set(Ntot_train_set,0:2),Cost_input(0:2) )
    !
    allocate( SE_train_set(Ntot_train_set,2),SE_train(2) )
    allocate( SE_val_set(Ntot_train_set,2),SE_val(2) )
    allocate( SE_input_set(Ntot_train_set,2),SE_input(2) )
    !
    allocate( AE_train_set(Ntot_train_set,2),AE_train(2) )
    allocate( AE_val_set(Ntot_train_set,2),AE_val(2) )
    allocate( AE_input_set(Ntot_train_set,2),AE_input(2) )
  END SUBROUTINE alloc_for_validation
  
  SUBROUTINE prep_outfiles
    implicit none
    open(100,file=file_history,status='replace')
    write(100,'(a)') "# DNNF90 training history: one row per validation event"
    write(100,'(a)') "#  1:epoch  2:patience"
    write(100,'(a)') "#  3:cost_train    4:cost_val    5:cost_input   (per point;"
    write(100,'(a)') "#    the composite objective, Loss_term weights applied --"
    write(100,'(a)') "#    the same sum the optimizer descends)"
    write(100,'(a)') "#  6:cost_train_f  7:cost_train_df  8:cost_val_f  9:cost_val_df"
    write(100,'(a)') "# 10:cost_input_f 11:cost_input_df"
    write(100,'(a)') "# 12:rmse_train_f 13:rmse_train_df 14:rmse_val_f 15:rmse_val_df"
    write(100,'(a)') "# 16:rmse_input_f 17:rmse_input_df"
    write(100,'(a)') "# 18:mae_train_f  19:mae_train_df  20:mae_val_f  21:mae_val_df"
    write(100,'(a)') "# 22:mae_input_f  23:mae_input_df"
    if ( Ntot_train_set > 1 ) then
       write(100,'(a,i0,a)') "# 24..: the same 21 metrics for each loss term in turn (", &
            Ntot_train_set, " terms: columns 24-44 are term 1, 45-65 term 2, ...)"
    end if
    close(100)
  END SUBROUTINE prep_outfiles

  
  SUBROUTINE perform_validation
    implicit none
    integer :: i, j, is, ie, n, itmp, it, iv
    integer :: iflag_train, iflag_validation
    real(8) :: vpred
    real(8),allocatable :: tvec(:)
    real(8) :: rtmp1, rtmp2
    integer :: ii
    real(8) :: rdiff, rtmp_hod
    ! Membership of the training and validation sets, built once.  It used
    ! to be decided by scanning ind_train and ind_validation for every
    ! point of every term, which costs NUM_input*(NUM_train+NUM_validation)
    ! comparisons per validation event: sixty million on the ten
    ! dimensional case.  The trainer already avoids the same pattern when
    ! it draws a batch.
    integer,allocatable :: member(:)
    real(8),allocatable :: tmval(:,:)   ! one row per field component
    real(8),allocatable :: rsys_v(:), syssrc_v(:)
    
    ! The optimizer has just moved the weights, so the library network
    ! is refreshed before any cost is measured.
    if ( .not. allocated(tvec) ) allocate( tvec(max(NUM_alpha,1)) )
    call lnet_sync_weights( weight )

    allocate( member(NUM_input) );  member = 0
    allocate( tmval(max(ndim(Nlayer),1),max(NUM_alpha,1)) );  tmval = 0.d0
    allocate( rsys_v(max(sys_nres,1)), syssrc_v(max(sys_nres,1)) )
    rsys_v = 0.d0;  syssrc_v = 0.d0
    do it=1,NUM_train
       member(ind_train(it)) = 1
    end do
    do it=1,NUM_validation
       member(ind_validation(it)) = 2
    end do

    Ntrain_set(:)=0
    Cost_train_set(:,:)=0.d0; Cost_train(:)=0.d0
    SE_train_set(:,:)=0.d0; SE_train(:)=0.d0
    AE_train_set(:,:)=0.d0; AE_train(:)=0.d0
    !
    Nval_set(:)=0
    Cost_val_set(:,:)=0.d0; Cost_val(:)=0.d0
    SE_val_set(:,:)=0.d0; SE_val(:)=0.d0
    AE_val_set(:,:)=0.d0; AE_val(:)=0.d0

    rtmp1=0.d0; rtmp2=0.d0

    ! calc functional derivative (START)

    ! PINN residual cost: Cost(:,2) = (1/2) sum R^2 over collocation points.
    ! If Exact_solution 1, SE/AE(:,1) report |u-u_exact| but do NOT enter Cost
    ! (early stopping must not use knowledge of the exact solution).
    do j=1,Ntot_train_set
       if ( form_train(j)=="PINN" ) then
          is = label_start(j)
          ie = label_end(j)
          do n=is,ie
             ! A system has one residual per component and the cost is
             ! the sum of their squares.  Evaluating only the scalar
             ! residual, as this did, reports a number unrelated to what
             ! the training minimizes, and that number is what the
             ! early-stopping rule reads.
             if ( sys_nterm > 0 ) then
                call lnet_forward_hod_multi( descriptor_input(n,1:ndim(1)), &
                     tmval )
                syssrc_v(1:sys_nres) = sys_src_input(n,1:sys_nres)
                call calc_sys_residual( tmval, syssrc_v, rsys_v )
                ! weighted the same way the seed is, so that the cost
                ! reported is the objective being minimized
                rtmp1 = sum( sys_wres(1:sys_nres)*rsys_v(1:sys_nres)**2 )
                rdiff = sqrt( rtmp1 )
             else
                call lnet_forward_hod( descriptor_input(n,1:ndim(1)), tvec )
                call calc_pinn_residual( tvec, pinn_src(n), rdiff )
                rtmp1 = rdiff**2
             end if
             iflag_train      = merge( 1, 0, member(n)==1 )
             iflag_validation = merge( 1, 0, member(n)==2 )
             rtmp2 = 0.d0
             if ( iswitch_pinn_exact == 1 ) then
                if ( sys_nterm > 0 ) then
                   rtmp2 = sum( abs( tmval(1:ndim(Nlayer),1) &
                        - response_input(n,1:ndim(Nlayer)) ) ) &
                        / dble(ndim(Nlayer))
                else
                   rtmp2 = abs( tvec(1)-response_input(n,1) )
                end if
             end if
             ! SE accumulates squares and AE absolute values: rtmp1 is
             ! the square and belongs only to the SE columns, while the
             ! AE columns take abs(rdiff), or the MAE of a collocation
             ! term would be a mean square under a different name.
             if ( iflag_train==1 ) then
                Cost_train_set(j,2) = Cost_train_set(j,2) + 0.5d0*rtmp1
                SE_train_set(j,2) = SE_train_set(j,2) + rtmp1
                AE_train_set(j,2) = AE_train_set(j,2) + abs(rdiff)
                SE_train_set(j,1) = SE_train_set(j,1) + rtmp2**2
                AE_train_set(j,1) = AE_train_set(j,1) + rtmp2
             end if
             if ( iflag_validation==1 ) then
                Cost_val_set(j,2) = Cost_val_set(j,2) + 0.5d0*rtmp1
                SE_val_set(j,2) = SE_val_set(j,2) + rtmp1
                AE_val_set(j,2) = AE_val_set(j,2) + abs(rdiff)
                SE_val_set(j,1) = SE_val_set(j,1) + rtmp2**2
                AE_val_set(j,1) = AE_val_set(j,1) + rtmp2
             end if
          end do
       end if
    end do

    ! High-order derivative cost for MATH_HOD sets:
    !   Cost(:,2) = (1/2) sum_{alpha>0} lambda_{|alpha|} (T-y)^2   (weighted)
    !   SE/AE(:,2) accumulate the unweighted derivative errors
    do j=1,Ntot_train_set
       if ( form_train(j)=="MATH_HOD" ) then
          is = label_start(j)
          ie = label_end(j)
          do n=is,ie
             call lnet_forward_hod( descriptor_input(n,1:ndim(1)), tvec )
             rtmp1=0.d0; rtmp2=0.d0; rtmp_hod=0.d0
             do ii=2,NUM_alpha
                rdiff = tvec(ii) - hod_target_input(n,ii)
                rtmp1 = rtmp1 + rdiff**2
                rtmp2 = rtmp2 + abs(rdiff)
                rtmp_hod = rtmp_hod + lambda_hod(alpha_deg(ii))*rdiff**2
             end do
             iflag_train      = merge( 1, 0, member(n)==1 )
             iflag_validation = merge( 1, 0, member(n)==2 )
             if ( iflag_train==1 ) then
                Cost_train_set(j,2) = Cost_train_set(j,2) + 0.5d0*rtmp_hod
                SE_train_set(j,2) = SE_train_set(j,2) + rtmp1
                AE_train_set(j,2) = AE_train_set(j,2) + rtmp2
             end if
             if ( iflag_validation==1 ) then
                Cost_val_set(j,2) = Cost_val_set(j,2) + 0.5d0*rtmp_hod
                SE_val_set(j,2) = SE_val_set(j,2) + rtmp1
                AE_val_set(j,2) = AE_val_set(j,2) + rtmp2
             end if
          end do
       end if
    end do

    !For training data
    ! Feedforward
    do it=1,NUM_train
       itmp=0
       do j=1,Ntot_train_set
          if ( (label_start(j)<=ind_train(it)).and.(ind_train(it)<=label_end(j)) ) then
             itmp=j
          end if
       end do
       if ( itmp == 0 ) then
          write(*,*) "perform_validation: training index", ind_train(it), &
               " lies in no loss term"
          stop
       end if
       Ntrain_set(itmp) = Ntrain_set(itmp)+1
       ! The leading index of the staging buffers is a single-point
       ! placeholder.
       ! One prediction per field component.  The loop below runs over
       ! the components, so it needs the prediction of each: comparing
       ! every target against the first component makes the reported cost
       ! of a system meaningless, and it is the cost the early-stopping
       ! rule reads.
       if ( ndim(Nlayer) > 1 ) then
          call lnet_forward_hod_multi( &
               descriptor_input(ind_train(it),1:ndim(1)), tmval )
       else
          call lnet_forward_value( descriptor_input(ind_train(it),1:ndim(1)), &
               vpred )
       end if
       do i=1,ndim(Nlayer)
          if ( form_train(itmp)=="PINN" ) cycle   ! residual cost handled above
          if ( ndim(Nlayer) > 1 ) vpred = tmval(i,1)
          ! sys_wcomp weights each component the way the training seed
          ! does, so the reported cost is the objective being minimized
          rtmp1 = sys_wcomp(i)*( vpred-response_input(ind_train(it),i) )**2
          rtmp2 = abs( vpred-response_input(ind_train(it),i) )
          Cost_train_set(itmp,1) = Cost_train_set(itmp,1) + 0.5d0*rtmp1
          SE_train_set(itmp,1) = SE_train_set(itmp,1) + rtmp1
          AE_train_set(itmp,1) = AE_train_set(itmp,1) + rtmp2
          !
          !                rbuf1(itmp) = rbuf1(itmp) + zmat(Nlayer,i)
          !                rbuf2(itmp) = rbuf2(itmp) + response_input(ind_train(it),i)
       end do
    end do

    
    ! For validation data
    ! Feedforward
    do iv=1,NUM_validation
       itmp=0
       do j=1,Ntot_train_set
          if ( (label_start(j)<=ind_validation(iv)).and.(ind_validation(iv)<=label_end(j)) ) then
             itmp=j
          end if
       end do
       if ( itmp == 0 ) then
          write(*,*) "perform_validation: validation index", &
               ind_validation(iv), " lies in no loss term"
          stop
       end if
       Nval_set(itmp) = Nval_set(itmp)+1
       
       ! The validation points are read straight from the input arrays;
       ! the leading index of the staging buffers is a single-point
       ! placeholder.
       if ( ndim(Nlayer) > 1 ) then
          call lnet_forward_hod_multi( &
               descriptor_input(ind_validation(iv),1:ndim(1)), tmval )
       else
          call lnet_forward_value( &
               descriptor_input(ind_validation(iv),1:ndim(1)), vpred )
       end if
       do i=1,ndim(Nlayer)
          if ( form_train(itmp)=="PINN" ) cycle   ! residual cost handled above
          if ( ndim(Nlayer) > 1 ) vpred = tmval(i,1)
          rtmp1 = sys_wcomp(i)*( vpred-response_input(ind_validation(iv),i) )**2
          rtmp2 = abs( vpred-response_input(ind_validation(iv),i) )
          Cost_val_set(itmp,1) = Cost_val_set(itmp,1) + 0.5d0*rtmp1
          SE_val_set(itmp,1) = SE_val_set(itmp,1) + rtmp1
          AE_val_set(itmp,1) = AE_val_set(itmp,1) + rtmp2
       end do
    end do

    ! The composite objective the optimizer descends carries the
    ! Loss_term weight of every set (the seed multiplies by
    ! gd_ratio(j), and so does the L-BFGS full cost).  The cost that
    ! the history, the patience rule and the best-weight selection read
    ! must be the same object, or a run is stopped and its "best"
    ! chosen against a number the training is not minimizing -- with a
    ! small collocation weight the unweighted residual dominated the
    ! display and the descent looked far slower than it was.  The
    ! per-part columns (:,1) and (:,2) and the SE/AE metrics stay
    ! unweighted: they are diagnostics of the parts, not the objective.
    do j=1,Ntot_train_set
       if ( form_train(j)=="PINN" ) then
          Cost_train_set(j,0) = gd_ratio(j)*Cost_train_set(j,2)
       else if ( form_train(j)=="MATH_HOD" ) then
          Cost_train_set(j,0) = gd_ratio(j)*( lambda_hod(0)*Cost_train_set(j,1) + Cost_train_set(j,2) )
       else
          Cost_train_set(j,0) = gd_ratio(j)*Cost_train_set(j,1)
       end if
    end do
    ! sum over Ntot_train_set
    do j=1,2
       Cost_train(j) = sum( Cost_train_set(:,j) )
       SE_train(j) = sum( SE_train_set(:,j) )
       AE_train(j) = sum( AE_train_set(:,j) )
    end do
    Cost_train(0) = sum( Cost_train_set(:,0) )


    do j=1,Ntot_train_set
       if ( form_train(j)=="PINN" ) then
          Cost_val_set(j,0) = gd_ratio(j)*Cost_val_set(j,2)
       else if ( form_train(j)=="MATH_HOD" ) then
          Cost_val_set(j,0) = gd_ratio(j)*( lambda_hod(0)*Cost_val_set(j,1) + Cost_val_set(j,2) )
       else
          Cost_val_set(j,0) = gd_ratio(j)*Cost_val_set(j,1)
       end if
    end do
    ! sum over Ntot_train_set
    do j=1,2
       Cost_val(j) = sum( Cost_val_set(:,j) )
       SE_val(j) = sum( SE_val_set(:,j) )
       AE_val(j) = sum( AE_val_set(:,j) )
    end do
    Cost_val(0) = sum( Cost_val_set(:,0) )
    

    ! Get *_input_set(Ntot_train_set,2), *input(2)
    do j=1,Ntot_train_set
       Cost_input_set(j,:) = Cost_train_set(j,:) + Cost_val_set(j,:)
       SE_input_set(j,:) = SE_train_set(j,:) + SE_val_set(j,:)
       AE_input_set(j,:) = AE_train_set(j,:) + AE_val_set(j,:)
    end do
    do j=1,2
       Cost_input(j) = sum( Cost_input_set(:,j) )
       SE_input(j) = sum( SE_input_set(:,j) )
       AE_input(j) = sum( AE_input_set(:,j) )
    end do
    Cost_input(0) = sum( Cost_input_set(:,0) )

    deallocate( member )
    deallocate( tmval, rsys_v, syssrc_v )
  END SUBROUTINE perform_validation
  
  
  
  SUBROUTINE write_outfile( loop,patience )
    implicit none
    integer,intent(IN) :: loop,patience

    real(8) :: v(21)
    real(8) :: vs(Ntot_train_set, 21)
    integer :: jj
    !
    v(1)  = Cost_train(0)/NUM_train
    v(2)  = val_safe( Cost_val(0), NUM_validation )
    v(3)  = Cost_input(0)/NUM_input
    v(4)  = Cost_train(1)/NUM_train
    v(5)  = Cost_train(2)/NUM_train
    v(6)  = val_safe( Cost_val(1), NUM_validation )
    v(7)  = val_safe( Cost_val(2), NUM_validation )
    v(8)  = Cost_input(1)/NUM_input
    v(9)  = Cost_input(2)/NUM_input
    v(10) = sqrt(SE_train(1)/NUM_train)
    v(11) = sqrt(SE_train(2)/NUM_train)
    v(12) = sqrt( val_safe( SE_val(1), NUM_validation ) )
    v(13) = sqrt( val_safe( SE_val(2), NUM_validation ) )
    v(14) = sqrt(SE_input(1)/NUM_input)
    v(15) = sqrt(SE_input(2)/NUM_input)
    v(16) = AE_train(1)/NUM_train
    v(17) = AE_train(2)/NUM_train
    v(18) = val_safe( AE_val(1), NUM_validation )
    v(19) = val_safe( AE_val(2), NUM_validation )
    v(20) = AE_input(1)/NUM_input
    v(21) = AE_input(2)/NUM_input
    !
    vs(:,1)  = Cost_train_set(:,0)/Ntrain_set(:)
    do jj=1,Ntot_train_set
       vs(jj,2)  = val_safe( Cost_val_set(jj,0), Nval_set(jj) )
       vs(jj,6)  = val_safe( Cost_val_set(jj,1), Nval_set(jj) )
       vs(jj,7)  = val_safe( Cost_val_set(jj,2), Nval_set(jj) )
       vs(jj,12) = sqrt( val_safe( SE_val_set(jj,1), Nval_set(jj) ) )
       vs(jj,13) = sqrt( val_safe( SE_val_set(jj,2), Nval_set(jj) ) )
       vs(jj,18) = val_safe( AE_val_set(jj,1), Nval_set(jj) )
       vs(jj,19) = val_safe( AE_val_set(jj,2), Nval_set(jj) )
    end do
    vs(:,3)  = Cost_input_set(:,0)/Ndata_train_set(1:Ntot_train_set)
    vs(:,4)  = Cost_train_set(:,1)/Ntrain_set(:)
    vs(:,5)  = Cost_train_set(:,2)/Ntrain_set(:)
    vs(:,8)  = Cost_input_set(:,1)/Ndata_train_set(1:Ntot_train_set)
    vs(:,9)  = Cost_input_set(:,2)/Ndata_train_set(1:Ntot_train_set)
    vs(:,10) = sqrt(SE_train_set(:,1)/Ntrain_set(:))
    vs(:,11) = sqrt(SE_train_set(:,2)/Ntrain_set(:))
    vs(:,14) = sqrt(SE_input_set(:,1)/Ndata_train_set(1:Ntot_train_set))
    vs(:,15) = sqrt(SE_input_set(:,2)/Ndata_train_set(1:Ntot_train_set))
    vs(:,16) = AE_train_set(:,1)/Ntrain_set(:)
    vs(:,17) = AE_train_set(:,2)/Ntrain_set(:)
    vs(:,20) = AE_input_set(:,1)/Ndata_train_set(1:Ntot_train_set)
    vs(:,21) = AE_input_set(:,2)/Ndata_train_set(1:Ntot_train_set)
    !
    open(100,file=file_history,position='append')
    if ( Ntot_train_set > 1 ) then
       write(100,'(i0,2x,i0,1000(2x,e13.5))') istart_step, patience, v, transpose(vs)
    else
       write(100,'(i0,2x,i0,21(2x,e13.5))') istart_step, patience, v
    end if
    close(100)

    call write_data("weight_best")
!    first_time = .false.
    
  END SUBROUTINE write_outfile
  

  !> x/n, or zero when there is nothing to average over (absent
  !! validation points would otherwise print NaN in the history).
  REAL(8) FUNCTION val_safe( x, n )
    implicit none
    real(8),intent(IN) :: x
    integer,intent(IN) :: n
    if ( n > 0 ) then
       val_safe = x/dble(n)
    else
       val_safe = 0.d0
    end if
  END FUNCTION val_safe

END MODULE validation_module

