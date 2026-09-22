! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone modules; status-returning public facade in sheath_model.
!> Zhao residuals, Sagdeev integrals, and physical profile admissibility.
!! 根の探索順・選択方針を持たず、与えられた状態の物理量と成立条件を評価する。
submodule(sheath_model_field) sheath_model_field_physics
  use sheath_model_core, only: evaluate_zhao_rho_hat, &
                               zhao_residuals_type_a, zhao_residuals_type_b, zhao_residuals_type_c
  implicit none

  integer, parameter :: rho_quadrature_panels = 256
  integer, parameter :: energy_quadrature_panels = 128
  integer, parameter :: profile_validation_samples = 32
  real(dp), parameter :: profile_negative_tolerance = 1.0e-7_dp
  real(dp), parameter :: profile_endpoint_tolerance = 1.0e-5_dp

contains

  module procedure encode_field_unknowns

  y = 0.0_dp
  valid = density_m3 > 0.0_dp .and. params%n_phe_ref_m3 > 0.0_dp .and. params%t_phe_ev > 0.0_dp
  if (.not. valid) return
  select case (branch)
  case ('A')
    valid = phi0_v > 0.0_dp .and. phi_m_v < 0.0_dp
    if (.not. valid) return
    y(1) = log(phi0_v/params%t_phe_ev)
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
    phi0_v = params%t_phe_ev*exp(y(1))
    phi_m_v = -params%t_phe_ev*exp(y(2))
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
    if (field_squared < -1.0e-10_dp) then
      valid = .false.
      return
    end if
    field_residual_scale = max(1.0_dp, target_field_hat*target_field_hat)
    residual(1) = raw(1)/params%n_phe_ref_m3
    residual(2) = (max(0.0_dp, field_squared) - target_field_hat*target_field_hat)/field_residual_scale
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
    if (field_squared < -1.0e-10_dp) then
      valid = .false.
      return
    end if
    field_residual_scale = max(1.0_dp, target_field_hat*target_field_hat)
    residual(1) = raw(1)/params%n_phe_ref_m3
    residual(2) = (max(0.0_dp, field_squared) - target_field_hat*target_field_hat)/field_residual_scale
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

    real(dp) :: t, phi_hat, jacobian, rho_hat, summand, weight, h
    integer :: point

    integral = 0.0_dp
    success = .false.
    if (.not. all(ieee_is_finite([ &
                                 lower_phi_hat, upper_phi_hat, phi0_hat, phi_m_hat, density_hat &
                                 ])) .or. density_hat <= 0.0_dp) return
    h = 1.0_dp/real(rho_quadrature_panels, dp)
    do point = 0, rho_quadrature_panels
      t = real(point, dp)*h
      phi_hat = lower_phi_hat + (upper_phi_hat - lower_phi_hat)*sin(0.5_dp*pi*t)**2
      jacobian = (upper_phi_hat - lower_phi_hat)*0.5_dp*pi*sin(pi*t)
      if (.not. ion_accessible(params, phi_hat)) return
      call evaluate_zhao_rho_hat( &
        params, branch, side, phi_hat, phi0_hat, phi_m_hat, density_hat, rho_hat &
        )
      if (.not. ieee_is_finite(rho_hat)) return
      summand = rho_hat*jacobian
      if (point == 0 .or. point == rho_quadrature_panels) then
        weight = 1.0_dp
      else if (mod(point, 2) == 0) then
        weight = 2.0_dp
      else
        weight = 4.0_dp
      end if
      integral = integral + weight*summand
    end do
    integral = integral*h/3.0_dp
    success = ieee_is_finite(integral)
  end subroutine integrate_field_rho_hat

  module procedure validate_field_root_profile

  real(dp) :: phi0_hat, phi_m_hat, density_hat, phi_hat, fraction
  real(dp) :: integral, field_squared, interface_field_squared, upper_endpoint_field_squared
  real(dp) :: minimum_field_squared, field_squared_scale
  integer :: point
  logical :: integral_ok

  status = sheath_numerical_failure
  message = ''
  root%minimum_field_squared_hat = huge(1.0_dp)
  phi0_hat = root%phi0_v/params%t_phe_ev
  phi_m_hat = root%phi_m_v/params%t_phe_ev
  density_hat = root%ambient_electron_density_m3/params%n_phe_ref_m3
  field_squared_scale = max(1.0_dp, target_field_hat*target_field_hat)
  if (.not. all(ieee_is_finite([ &
                               phi0_hat, phi_m_hat, density_hat, field_squared_scale &
                               ])) .or. density_hat <= 0.0_dp) then
    message = 'prescribed-field Zhao profile normalization is invalid.'
    return
  end if

  minimum_field_squared = huge(1.0_dp)
  interface_field_squared = huge(1.0_dp)
  upper_endpoint_field_squared = 0.0_dp
  select case (root%branch)
  case ('A')
    do point = 0, profile_validation_samples
      fraction = real(point, dp)/real(profile_validation_samples, dp)
      phi_hat = phi_m_hat + fraction*(phi0_hat - phi_m_hat)
      call integrate_field_rho_hat( &
        params, 'A', 'lower', phi_m_hat, phi_hat, phi0_hat, phi_m_hat, &
        density_hat, integral, integral_ok &
        )
      if (.not. integral_ok) then
        message = 'prescribed-field Zhao lower profile integration failed.'
        return
      end if
      field_squared = -2.0_dp*integral
      minimum_field_squared = min(minimum_field_squared, field_squared)
      if (point == profile_validation_samples) interface_field_squared = field_squared
    end do
    do point = 0, profile_validation_samples
      fraction = real(point, dp)/real(profile_validation_samples, dp)
      phi_hat = phi_m_hat + fraction*(0.0_dp - phi_m_hat)
      call integrate_field_rho_hat( &
        params, 'A', 'upper', phi_m_hat, phi_hat, phi0_hat, phi_m_hat, &
        density_hat, integral, integral_ok &
        )
      if (.not. integral_ok) then
        message = 'prescribed-field Zhao upper profile integration failed.'
        return
      end if
      field_squared = -2.0_dp*integral
      minimum_field_squared = min(minimum_field_squared, field_squared)
      if (point == profile_validation_samples) upper_endpoint_field_squared = field_squared
    end do
  case ('B', 'C')
    do point = 0, profile_validation_samples
      fraction = real(point, dp)/real(profile_validation_samples, dp)
      phi_hat = phi0_hat + fraction*(0.0_dp - phi0_hat)
      call integrate_field_rho_hat( &
        params, root%branch, 'monotonic', phi_hat, 0.0_dp, phi0_hat, phi_m_hat, &
        density_hat, integral, integral_ok &
        )
      if (.not. integral_ok) then
        message = 'prescribed-field Zhao monotonic profile integration failed.'
        return
      end if
      field_squared = 2.0_dp*integral
      minimum_field_squared = min(minimum_field_squared, field_squared)
      if (point == 0) interface_field_squared = field_squared
    end do
  case default
    message = 'prescribed-field Zhao profile has an unknown branch.'
    return
  end select

  if (.not. all(ieee_is_finite([ &
                               minimum_field_squared, interface_field_squared, upper_endpoint_field_squared &
                               ]))) then
    message = 'prescribed-field Zhao profile field is non-finite.'
    return
  end if
  root%minimum_field_squared_hat = minimum_field_squared
  if (minimum_field_squared < -profile_negative_tolerance*field_squared_scale) then
    status = sheath_no_physical_solution
    message = 'prescribed-field Zhao profile requires an imaginary electric field.'
    return
  end if
  if (abs(interface_field_squared - target_field_hat*target_field_hat) > &
      profile_endpoint_tolerance*field_squared_scale) then
    message = 'prescribed-field Zhao profile does not reproduce the interface field.'
    return
  end if
  if (root%branch == 'A' .and. &
      abs(upper_endpoint_field_squared) > profile_endpoint_tolerance*field_squared_scale) then
    message = 'prescribed-field Zhao-A upper profile does not reach zero upstream field.'
    return
  end if
  status = sheath_ok
  end procedure validate_field_root_profile

  module procedure evaluate_root_potential_energy

  real(dp) :: phi0_hat, phi_m_hat, density_hat, energy_hat, segment_energy_hat
  logical :: success

  status = sheath_numerical_failure
  message = ''
  root%potential_energy_j_m2 = huge(1.0_dp)
  phi0_hat = root%phi0_v/params%t_phe_ev
  phi_m_hat = root%phi_m_v/params%t_phe_ev
  density_hat = root%ambient_electron_density_m3/params%n_phe_ref_m3
  if (.not. all(ieee_is_finite([phi0_hat, phi_m_hat, density_hat])) .or. &
      density_hat <= 0.0_dp .or. params%lambda_d_phe_ref_m <= 0.0_dp) then
    message = 'prescribed-field Zhao potential-energy normalization is invalid.'
    return
  end if

  energy_hat = 0.0_dp
  select case (root%branch)
  case ('A')
    call integrate_field_field_energy_hat( &
      params, root%branch, 'lower', phi_m_hat, phi0_hat, phi0_hat, phi_m_hat, &
      density_hat, segment_energy_hat, success &
      )
    if (.not. success) then
      message = 'prescribed-field Zhao-A lower potential-energy integral failed.'
      return
    end if
    energy_hat = energy_hat + segment_energy_hat
    call integrate_field_field_energy_hat( &
      params, root%branch, 'upper', phi_m_hat, 0.0_dp, phi0_hat, phi_m_hat, &
      density_hat, segment_energy_hat, success &
      )
    if (.not. success) then
      message = 'prescribed-field Zhao-A upper potential-energy integral failed.'
      return
    end if
    energy_hat = energy_hat + segment_energy_hat
  case ('B', 'C')
    call integrate_field_field_energy_hat( &
      params, root%branch, 'monotonic', phi0_hat, 0.0_dp, phi0_hat, phi_m_hat, &
      density_hat, energy_hat, success &
      )
    if (.not. success) then
      message = 'prescribed-field Zhao monotonic potential-energy integral failed.'
      return
    end if
  case default
    message = 'prescribed-field Zhao potential-energy root has an unknown branch.'
    return
  end select
  root%potential_energy_j_m2 = -0.5_dp*eps0*params%t_phe_ev*params%t_phe_ev* &
                               energy_hat/params%lambda_d_phe_ref_m
  if (.not. ieee_is_finite(root%potential_energy_j_m2) .or. root%potential_energy_j_m2 > 0.0_dp) then
    root%potential_energy_j_m2 = huge(1.0_dp)
    message = 'prescribed-field Zhao potential energy is invalid.'
    return
  end if
  status = sheath_ok
  end procedure evaluate_root_potential_energy

  subroutine integrate_field_field_energy_hat( &
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

    ! Preserve the original Simpson accumulation order for reproducible root ranking.
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
  end subroutine integrate_field_field_energy_hat

  pure logical function ion_accessible(params, phi_hat) result(accessible)
    type(zhao_params_type), intent(in) :: params
    real(dp), intent(in) :: phi_hat

    accessible = params%tau > 0.0_dp .and. params%mach > 0.0_dp .and. &
                 1.0_dp - 2.0_dp*phi_hat/(params%tau*params%mach*params%mach) > 0.0_dp
  end function ion_accessible

end submodule sheath_model_field_physics
