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

  integer, parameter :: energy_quadrature_panels = 128
  real(dp), parameter :: profile_negative_tolerance = 1.0e-7_dp

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

  module procedure evaluate_root_field_energy

  real(dp) :: phi0_hat, phi_m_hat, density_hat, energy_hat, segment_energy_hat
  logical :: success

  status = sheath_numerical_failure
  message = ''
  root%field_energy_j_m2 = huge(1.0_dp)
  phi0_hat = root%phi0_v/params%t_phe_ev
  phi_m_hat = root%phi_m_v/params%t_phe_ev
  density_hat = root%ambient_electron_density_m3/params%n_phe_ref_m3
  if (.not. all(ieee_is_finite([phi0_hat, phi_m_hat, density_hat])) .or. &
      density_hat <= 0.0_dp .or. params%lambda_d_phe_ref_m <= 0.0_dp) then
    message = 'prescribed-field Zhao field-energy normalization is invalid.'
    return
  end if

  energy_hat = 0.0_dp
  select case (root%branch)
  case ('A')
    call integrate_field_energy_hat( &
      params, root%branch, 'lower', phi_m_hat, phi0_hat, phi0_hat, phi_m_hat, &
      density_hat, segment_energy_hat, success &
      )
    if (.not. success) then
      message = 'prescribed-field Zhao-A lower field-energy integral failed.'
      return
    end if
    energy_hat = energy_hat + segment_energy_hat
    call integrate_field_energy_hat( &
      params, root%branch, 'upper', phi_m_hat, 0.0_dp, phi0_hat, phi_m_hat, &
      density_hat, segment_energy_hat, success &
      )
    if (.not. success) then
      message = 'prescribed-field Zhao-A upper field-energy integral failed.'
      return
    end if
    energy_hat = energy_hat + segment_energy_hat
  case ('B', 'C')
    call integrate_field_energy_hat( &
      params, root%branch, 'monotonic', phi0_hat, 0.0_dp, phi0_hat, phi_m_hat, &
      density_hat, energy_hat, success &
      )
    if (.not. success) then
      message = 'prescribed-field Zhao monotonic field-energy integral failed.'
      return
    end if
  case default
    message = 'prescribed-field Zhao field-energy root has an unknown branch.'
    return
  end select
  root%field_energy_j_m2 = 0.5_dp*eps0*params%t_phe_ev*params%t_phe_ev* &
                           energy_hat/params%lambda_d_phe_ref_m
  if (.not. ieee_is_finite(root%field_energy_j_m2) .or. root%field_energy_j_m2 < 0.0_dp) then
    root%field_energy_j_m2 = huge(1.0_dp)
    message = 'prescribed-field Zhao field energy is invalid.'
    return
  end if
  status = sheath_ok
  end procedure evaluate_root_field_energy

  subroutine integrate_field_energy_hat( &
    params, branch, side, start_phi_hat, end_phi_hat, phi0_hat, phi_m_hat, &
    density_hat, energy_hat, success &
    )
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    character(len=*), intent(in) :: side
    real(dp), intent(in) :: start_phi_hat, end_phi_hat, phi0_hat, phi_m_hat, density_hat
    real(dp), intent(out) :: energy_hat
    logical, intent(out) :: success

    real(dp) :: t, phi_hat, jacobian, rho_integral, field_squared, summand, weight, h
    real(dp) :: field_squared_scale
    real(dp) :: energy_integrand(0:energy_quadrature_panels)
    integer :: point
    logical :: integral_ok, point_ok(0:energy_quadrature_panels)

    energy_hat = 0.0_dp
    success = .false.
    if (.not. all(ieee_is_finite([ &
                                 start_phi_hat, end_phi_hat, phi0_hat, phi_m_hat, density_hat &
                                 ])) .or. density_hat <= 0.0_dp) return
    h = 1.0_dp/real(energy_quadrature_panels, dp)
    field_squared_scale = max(1.0_dp, phi0_hat*phi0_hat, phi_m_hat*phi_m_hat)
    energy_integrand = 0.0_dp
    point_ok = .false.
    do point = 0, energy_quadrature_panels
      t = real(point, dp)*h
      phi_hat = start_phi_hat + (end_phi_hat - start_phi_hat)*sin(0.5_dp*pi*t)**2
      jacobian = (end_phi_hat - start_phi_hat)*0.5_dp*pi*sin(pi*t)
      if (branch == 'A') then
        call integrate_field_rho_hat( &
          params, branch, side, phi_m_hat, phi_hat, phi0_hat, phi_m_hat, &
          density_hat, rho_integral, integral_ok &
          )
        field_squared = -2.0_dp*rho_integral
      else
        call integrate_field_rho_hat( &
          params, branch, side, phi_hat, 0.0_dp, phi0_hat, phi_m_hat, &
          density_hat, rho_integral, integral_ok &
          )
        field_squared = 2.0_dp*rho_integral
      end if
      if (.not. integral_ok) cycle
      if (field_squared < -profile_negative_tolerance*field_squared_scale) cycle
      energy_integrand(point) = sqrt(max(0.0_dp, field_squared))*abs(jacobian)
      point_ok(point) = .true.
    end do
    if (.not. all(point_ok)) return

    ! Composite Simpson integral over the potential path.
    do point = 0, energy_quadrature_panels
      summand = energy_integrand(point)
      if (point == 0 .or. point == energy_quadrature_panels) then
        weight = 1.0_dp
      else if (mod(point, 2) == 0) then
        weight = 2.0_dp
      else
        weight = 4.0_dp
      end if
      energy_hat = energy_hat + weight*summand
    end do
    energy_hat = energy_hat*h/3.0_dp
    success = ieee_is_finite(energy_hat) .and. energy_hat >= 0.0_dp
  end subroutine integrate_field_energy_hat

  pure logical function ion_accessible(params, phi_hat) result(accessible)
    type(zhao_params_type), intent(in) :: params
    real(dp), intent(in) :: phi_hat

    accessible = params%tau > 0.0_dp .and. params%mach > 0.0_dp .and. &
                 1.0_dp - 2.0_dp*phi_hat/(params%tau*params%mach*params%mach) > 0.0_dp
  end function ion_accessible

end submodule sheath_model_field_physics
