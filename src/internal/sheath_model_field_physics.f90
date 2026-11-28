! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone modules; status-returning public facade in sheath_model.
!> Zhao residuals, Sagdeev integrals, and physical profile admissibility.
!! 根の探索順・選択方針を持たず、与えられた状態の物理量と成立条件を評価する。
submodule(sheath_model_field) sheath_model_field_physics
  use sheath_model_admissibility, only: validate_zhao_profile
  use sheath_model_core, only: integrate_zhao_rho, type_a_connection_residual, zhao_balance_residuals
  use sheath_model_ions, only: ion_density_ratio
  implicit none

contains

  module subroutine evaluate_charge_residual( &
      params, branch, target_field_hat, &
      y, residual, valid &
      )
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: target_field_hat, y(3)
    real(dp), intent(out) :: residual(3)
    logical, intent(out) :: valid

    real(dp) :: phi0_v, phi_m_v, density_m3

    residual = 0.0_dp
    call decode_unknowns(params, branch, y, phi0_v, phi_m_v, density_m3, valid)
    if (.not. valid) return
    call evaluate_physical_field_residual(params, branch, target_field_hat, phi0_v, phi_m_v, density_m3, residual, valid)
  end subroutine evaluate_charge_residual

  ! Physical coordinates preserve exact energy-bin endpoints through final acceptance.
  module subroutine evaluate_physical_field_residual(params, branch, target_field_hat, phi0_v, phi_m_v, &
      density_m3, residual, valid)
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: target_field_hat, phi0_v, phi_m_v, density_m3
    real(dp), intent(out) :: residual(3)
    logical, intent(out) :: valid
    real(dp) :: phi0_hat, phi_m_hat, density_hat
    real(dp) :: neutrality, field_squared

    residual = 0.0_dp
    valid = all(ieee_is_finite([phi0_v, phi_m_v, density_m3])) .and. density_m3 > 0.0_dp
    if (.not. valid) return

    phi0_hat = phi0_v/params%potential_scale_v
    phi_m_hat = phi_m_v/params%potential_scale_v
    density_hat = density_m3/params%density_scale_m3
    if (.not. ion_accessible(params, max(phi0_hat, 0.0_dp))) then
      valid = .false.
      return
    end if

    call zhao_balance_residuals(params, branch, phi0_v, phi_m_v, density_m3, neutrality)
    select case (branch)
    case ('A')
      field_squared = -2.0_dp*integrate_zhao_rho(params, branch, 'lower', phi_m_hat, phi0_hat, &
          phi0_hat, phi_m_hat, density_hat)
      ! Scale by the leading depth^(3/2) so a collapsed upper segment is not a root.
      residual(3) = type_a_connection_residual(params, phi0_hat, phi_m_hat, density_hat)
    case ('B', 'C')
      field_squared = 2.0_dp*integrate_zhao_rho(params, branch, 'monotonic', phi0_hat, 0.0_dp, &
          phi0_hat, phi_m_hat, density_hat)
    case default
      valid = .false.
      return
    end select
    residual(1) = neutrality/params%density_scale_m3
    residual(2) = (field_squared - target_field_hat*target_field_hat)/max(1.0_dp, target_field_hat*target_field_hat)
    valid = all(ieee_is_finite(residual))
  end subroutine evaluate_physical_field_residual

  module subroutine validate_field_root_profile( &
      params, root, target_field_hat, &
      status, message &
      )
    type(zhao_params_type), intent(in) :: params
    type(zhao_field_root), intent(inout) :: root
    real(dp), intent(in) :: target_field_hat
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    real(dp) :: boundary_e2, band

    call validate_zhao_profile(params, root%branch, root%phi0_v/params%potential_scale_v, &
        root%phi_m_v/params%potential_scale_v, root%ambient_electron_density_m3/params%density_scale_m3, &
        root%minimum_field_squared_hat, boundary_e2, status, message, band, root%upstream_field_deficit_v_m)
    if (status /= SHEATH_OK) return
    root%upstream_negative_band_v = band*params%potential_scale_v

    if (abs(boundary_e2 - target_field_hat**2) > 1e-7_dp*max(1.0_dp, target_field_hat**2)) then
      status = SHEATH_NUMERICAL_FAILURE
      message = 'The profile does not reproduce the prescribed field.'
    end if
  end subroutine validate_field_root_profile

  pure logical function ion_accessible(params, phi_hat) result(accessible)
    type(zhao_params_type), intent(in) :: params
    real(dp), intent(in) :: phi_hat

    accessible = ieee_is_finite(ion_density_ratio(phi_hat*params%potential_scale_v, &
        0.5_dp*params%t_swe_ev*params%mach**2, params%ion_pressure_factor*params%t_swi_ev))
  end function ion_accessible

end submodule sheath_model_field_physics
