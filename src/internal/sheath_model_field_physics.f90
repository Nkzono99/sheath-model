! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone modules; status-returning public facade in sheath_model.
!> Zhao residuals, Sagdeev integrals, and physical profile admissibility.
!! 根の探索順・選択方針を持たず、与えられた状態の物理量と成立条件を評価する。
submodule(sheath_model_field) sheath_model_field_physics
  use sheath_model_admissibility, only: validate_zhao_profile
  use sheath_model_core, only: integrate_zhao_rho, &
                               zhao_residuals_type_a, zhao_residuals_type_b, zhao_residuals_type_c
  implicit none

contains

  module procedure encode_field_unknowns

  y = 0.0_dp
  valid = density_m3 > 0.0_dp .and. params%n_phe_ref_m3 > 0.0_dp .and. params%t_phe_ev > 0.0_dp
  if (.not. valid) return
  select case (branch)
  case ('A')
    valid = phi_m_v < min(phi0_v, 0.0_dp)
    if (.not. valid) return
    y(1) = log((phi0_v - phi_m_v)/params%t_phe_ev)
    y(2) = log(-phi_m_v/params%t_phe_ev)
    y(3) = log(density_m3/params%n_phe_ref_m3)
  case ('B')
    valid = phi0_v > 0.0_dp
    if (.not. valid) return
    y(1) = log(phi0_v/params%t_phe_ev)
    y(2) = log(density_m3/params%n_phe_ref_m3)
  case ('C')
    valid = phi0_v < 0.0_dp
    if (.not. valid) return
    y(1) = log(-phi0_v/params%t_phe_ev)
    y(2) = log(density_m3/params%n_phe_ref_m3)
  case default
    valid = .false.
  end select
  valid = valid .and. all(ieee_is_finite(y))
  end procedure encode_field_unknowns

  module procedure decode_field_unknowns

  phi0_v = 0.0_dp
  phi_m_v = 0.0_dp
  density_m3 = 0.0_dp
  valid = all(ieee_is_finite(y))
  if (.not. valid .or. y(1) < -50.0_dp .or. y(1) > log(200.0_dp)) then
    valid = .false.
    return
  end if
  select case (branch)
  case ('A')
    if (y(2) < -50.0_dp .or. y(2) > log(200.0_dp) .or. &
        y(3) < -30.0_dp .or. y(3) > log(1.0e6_dp)) then
      valid = .false.
      return
    end if
    phi_m_v = -params%t_phe_ev*exp(y(2))
    phi0_v = phi_m_v + params%t_phe_ev*exp(y(1))
    density_m3 = params%n_phe_ref_m3*exp(y(3))
  case ('B')
    if (y(2) < -30.0_dp .or. y(2) > log(1.0e6_dp)) then
      valid = .false.
      return
    end if
    phi0_v = params%t_phe_ev*exp(y(1))
    phi_m_v = phi0_v
    density_m3 = params%n_phe_ref_m3*exp(y(2))
  case ('C')
    if (y(2) < -30.0_dp .or. y(2) > log(1.0e6_dp)) then
      valid = .false.
      return
    end if
    phi0_v = -params%t_phe_ev*exp(y(1))
    phi_m_v = phi0_v
    density_m3 = params%n_phe_ref_m3*exp(y(2))
  case default
    valid = .false.
  end select
  valid = valid .and. all(ieee_is_finite([phi0_v, phi_m_v, density_m3]))
  end procedure decode_field_unknowns

  module procedure evaluate_charge_residual

  real(dp) :: phi0_v, phi_m_v, density_m3, phi0_hat, phi_m_hat, density_hat
  real(dp) :: raw(3), integral, field_squared, field_residual_scale
  real(dp) :: x3(3), x2(2)
  logical :: integral_ok

  residual = 0.0_dp
  call decode_field_unknowns( &
    params, branch, y, phi0_v, phi_m_v, density_m3, valid &
    )
  if (.not. valid) return
  phi0_hat = phi0_v/params%t_phe_ev
  phi_m_hat = phi_m_v/params%t_phe_ev
  density_hat = density_m3/params%n_phe_ref_m3
  if (.not. ion_accessible(params, max(phi0_hat, 0.0_dp))) then
    valid = .false.
    return
  end if

  select case (branch)
  case ('A')
    x3 = [phi0_v, phi_m_v, density_m3]
    call zhao_residuals_type_a(params, x3, raw)
    call integrate_field_rho_hat( &
      params, branch, 'lower', phi_m_hat, phi0_hat, phi0_hat, phi_m_hat, &
      density_hat, integral, integral_ok &
      )
    if (.not. integral_ok) then
      valid = .false.
      return
    end if
    field_squared = -2.0_dp*integral
    field_residual_scale = max(1.0_dp, target_field_hat*target_field_hat)
    residual(1) = raw(1)/params%n_phe_ref_m3
    residual(2) = (field_squared - target_field_hat*target_field_hat)/field_residual_scale
    residual(3) = raw(3)
  case ('B', 'C')
    x2 = [phi0_v, density_m3]
    if (branch == 'B') then
      call zhao_residuals_type_b(params, x2, raw(1:2))
    else
      call zhao_residuals_type_c(params, x2, raw(1:2))
    end if
    call integrate_field_rho_hat( &
      params, branch, 'monotonic', phi0_hat, 0.0_dp, phi0_hat, phi_m_hat, &
      density_hat, integral, integral_ok &
      )
    if (.not. integral_ok) then
      valid = .false.
      return
    end if
    field_squared = 2.0_dp*integral
    field_residual_scale = max(1.0_dp, target_field_hat*target_field_hat)
    residual(1) = raw(1)/params%n_phe_ref_m3
    residual(2) = (field_squared - target_field_hat*target_field_hat)/field_residual_scale
  case default
    valid = .false.
    return
  end select
  valid = all(ieee_is_finite(residual))
  end procedure evaluate_charge_residual

  subroutine integrate_field_rho_hat( &
    params, branch, side, lower_phi_hat, upper_phi_hat, phi0_hat, phi_m_hat, &
    density_hat, integral, success &
    )
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    character(len=*), intent(in) :: side
    real(dp), intent(in) :: lower_phi_hat, upper_phi_hat, phi0_hat, phi_m_hat, density_hat
    real(dp), intent(out) :: integral
    logical, intent(out) :: success

    integral = integrate_zhao_rho(params, branch, side, lower_phi_hat, upper_phi_hat, &
                                  phi0_hat, phi_m_hat, density_hat)
    success = ieee_is_finite(integral)
  end subroutine integrate_field_rho_hat

  module procedure validate_field_root_profile

  real(dp) :: boundary_e2
  call validate_zhao_profile(params, root%branch, root%phi0_v/params%t_phe_ev, &
                             root%phi_m_v/params%t_phe_ev, root%ambient_electron_density_m3/params%n_phe_ref_m3, &
                             root%minimum_field_squared_hat, boundary_e2, status, message)
  if (status /= sheath_ok) return
  if (abs(boundary_e2 - target_field_hat**2) > 1e-7_dp*max(1.0_dp, target_field_hat**2)) then
    status = sheath_numerical_failure
    message = 'The profile does not reproduce the prescribed field.'
  end if
  end procedure validate_field_root_profile

  pure logical function ion_accessible(params, phi_hat) result(accessible)
    type(zhao_params_type), intent(in) :: params
    real(dp), intent(in) :: phi_hat

    accessible = params%tau > 0.0_dp .and. params%mach > 0.0_dp .and. &
                 1.0_dp - 2.0_dp*phi_hat/(params%tau*params%mach*params%mach) > 0.0_dp
  end function ion_accessible

end submodule sheath_model_field_physics
