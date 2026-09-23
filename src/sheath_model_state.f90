! SPDX-License-Identifier: MIT
!> Stateless physical evaluation at specified potentials, independent of a time integrator.
module sheath_model_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
  use sheath_model_constants, only: dp, i32, qe, eps0, electron_mass, proton_mass, lower_ascii
  use sheath_model_status, only: SHEATH_OK, SHEATH_INVALID_ARGUMENT, SHEATH_NUMERICAL_FAILURE
  use sheath_model_photoelectrons, only: photoelectron_source, validate_photoelectrons, photoelectron_density
  use sheath_model_core, only: zhao_params_type, neutral_electron_density, integrate_zhao_rho, evaluate_zhao_fluxes, &
      evaluate_zhao_rho_hat
  use sheath_model_admissibility, only: validate_zhao_profile
  implicit none
  private
  public :: zhao_plasma_input, zhao_state_result, evaluate_sheath_state, prepare_plasma_params

  !> Fixed upstream ion state, electron distribution and outward boundary photoelectron source.
  !! SI units except electron temperature [eV]. Normal drifts are positive inward.
  type :: zhao_plasma_input
    real(dp) :: ion_density_m3 = 8.7e6_dp
    real(dp) :: electron_temperature_ev = 12.0_dp
    real(dp) :: electron_drift_mps = 4.0529988897111727e5_dp
    real(dp) :: ion_drift_mps = 4.0529988897111727e5_dp
    real(dp) :: ion_mass_kg = proton_mass
    real(dp) :: electron_mass_kg = electron_mass
    type(photoelectron_source) :: photoelectrons
  end type

  !> Evaluation of a trial state, not necessarily a complete physical sheath solution.
  !! Signed E^2 integrals use V^2/m^2. electric_field_v_m is NaN when boundary E^2 is negative.
  !! admissible requires physical_status=SHEATH_OK, including the Type-A upper connection.
  type :: zhao_state_result
    logical :: evaluated = .false., admissible = .false.
    character(len=1) :: branch = ' '
    real(dp) :: boundary_potential_v = 0.0_dp, minimum_potential_v = 0.0_dp
    real(dp) :: ambient_electron_density_m3 = 0.0_dp
    real(dp) :: neutrality_residual_m3 = 0.0_dp
    real(dp) :: boundary_field_squared_v2_m2 = 0.0_dp
    real(dp) :: connection_residual_v2_m2 = 0.0_dp
    real(dp) :: electric_field_v_m = 0.0_dp
    real(dp) :: electron_inward_flux_m2_s = 0.0_dp, ion_inward_flux_m2_s = 0.0_dp
    real(dp) :: photoelectron_outward_flux_m2_s = 0.0_dp
    real(dp) :: photoelectron_escape_flux_m2_s = 0.0_dp, photoelectron_return_flux_m2_s = 0.0_dp
    real(dp) :: net_current_a_m2 = 0.0_dp
    integer(i32) :: physical_status = SHEATH_INVALID_ARGUMENT
    character(len=256) :: physical_message = ''
  end type

contains

  !> Prepare internal scales and validate the fixed plasma/source input.
  !! Spectra use a binary numerical scale near T_e to preserve energy-bin edges; no Maxwellian fit is made.
  subroutine prepare_plasma_params(input, params, status, message)
    class(zhao_plasma_input), intent(in) :: input
    type(zhao_params_type), intent(out) :: params
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    real(dp) :: pe, captured
    params = zhao_params_type()
    status = SHEATH_INVALID_ARGUMENT
    message = 'Plasma inputs must be finite, with positive ion density, temperature, ion speed and masses.'
    if (.not. all(ieee_is_finite([input%ion_density_m3, input%electron_temperature_ev, &
        input%electron_drift_mps, input%ion_drift_mps, input%ion_mass_kg, input%electron_mass_kg]))) return
    if (min(input%ion_density_m3, input%electron_temperature_ev, input%ion_drift_mps, &
        input%ion_mass_kg, input%electron_mass_kg) <= 0.0_dp) return
    call validate_photoelectrons(input%photoelectrons, status, message)
    if (status /= SHEATH_OK) return
    params%photoelectrons = input%photoelectrons
    params%n_swi_inf_m3 = input%ion_density_m3
    params%density_scale_m3 = input%ion_density_m3
    params%t_swe_ev = input%electron_temperature_ev
    params%potential_scale_v = input%photoelectrons%potential_scale(input%electron_temperature_ev)
    params%v_d_electron_mps = input%electron_drift_mps
    params%v_d_ion_mps = input%ion_drift_mps
    params%m_i_kg = input%ion_mass_kg
    params%m_e_kg = input%electron_mass_kg
    params%v_swe_th_mps = sqrt(2.0_dp*qe*params%t_swe_ev/params%m_e_kg)
    params%velocity_scale_mps = sqrt(2.0_dp*qe*params%potential_scale_v/params%m_e_kg)
    params%cs_mps = sqrt(qe*params%t_swe_ev/params%m_i_kg)
    params%mach = params%v_d_ion_mps/params%cs_mps
    params%u = params%v_d_electron_mps/params%v_swe_th_mps
    params%tau = params%t_swe_ev/params%potential_scale_v
    params%length_scale_m = sqrt(eps0*params%potential_scale_v/(params%density_scale_m3*qe))
    call photoelectron_density(params%photoelectrons, params%m_e_kg, 0.0_dp, 0.0_dp, 0.0_dp, .false., pe, captured)
    params%emission_density_scale_m3 = 2.0_dp*pe
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Plasma/source normalization is non-finite or underflowed.'
    if (.not. all(ieee_is_finite([params%v_swe_th_mps, params%velocity_scale_mps, params%cs_mps, &
        params%mach, params%u, params%tau, params%length_scale_m, params%emission_density_scale_m3]))) return
    if (min(params%v_swe_th_mps, params%velocity_scale_mps, params%cs_mps, params%mach, &
        params%tau, params%length_scale_m) <= 0.0_dp) return
    status = SHEATH_OK
    message = ''
  end subroutine

  !> Evaluate B/C at boundary_potential_v, or A at boundary_potential_v and minimum_potential_v [V].
  !! The upstream electron normalization follows neutrality. B permits the flat phi_H=0 endpoint.
  !! SHEATH_OK means finite evaluation; inspect output%admissible and connection_residual_v2_m2 for a solution.
  !! Negative E^2 is retained, not clipped. minimum_potential_v is required only for A and forbidden for B/C.
  !! Optional electron_normalization_m3 permits a B/C rounding-cell correction only between the neutral densities
  !! at the immediately neighboring binary64 potentials. All fields/fluxes use that same density; no tolerance is relaxed.
  subroutine evaluate_sheath_state(input, branch, boundary_potential_v, output, status, message, &
      minimum_potential_v, electron_normalization_m3)
    class(zhao_plasma_input), intent(in) :: input
    character(len=*), intent(in) :: branch
    real(dp), intent(in) :: boundary_potential_v
    type(zhao_state_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    real(dp), intent(in), optional :: minimum_potential_v
    real(dp), intent(in), optional :: electron_normalization_m3
    type(zhao_params_type) :: p
    real(dp) :: phi0, phim, density, factor, minimum_e2, boundary_e2
    real(dp) :: left_phi, right_phi, left_density, right_density, minimum, rho
    character(len=1) :: selected
    output = zhao_state_result()
    call prepare_plasma_params(input, p, status, message)
    if (status /= SHEATH_OK) return
    status = SHEATH_INVALID_ARGUMENT
    message = 'Invalid branch or potentials: A requires phi_m<min(phi_H,0); B phi_H>=0; C phi_H<0.'
    if (.not. ieee_is_finite(boundary_potential_v)) return
    phi0 = boundary_potential_v/p%potential_scale_v
    if (.not. ieee_is_finite(phi0)) return
    select case (trim(lower_ascii(branch)))
    case ('a')
      selected = 'A'
      if (.not. present(minimum_potential_v)) return
      if (.not. ieee_is_finite(minimum_potential_v)) return
      if (minimum_potential_v >= min(boundary_potential_v, 0.0_dp)) return
      phim = minimum_potential_v/p%potential_scale_v
    case ('b')
      selected = 'B'
      if (boundary_potential_v < 0.0_dp .or. present(minimum_potential_v)) return
      phim = 0.0_dp
    case ('c')
      selected = 'C'
      if (boundary_potential_v >= 0.0_dp .or. present(minimum_potential_v)) return
      phim = phi0
    case default
      return
    end select
    if (.not. ieee_is_finite(phim)) return
    message = 'Trial potential blocks the cold ion beam.'
    if (1.0_dp - 2.0_dp*max(phi0, 0.0_dp)/(p%tau*p%mach**2) <= 0.0_dp) return
    density = neutral_electron_density(p, selected, boundary_potential_v, phim*p%potential_scale_v)
    if (present(electron_normalization_m3)) then
      message = 'Density correction is limited to the B/C neighboring-potential neutrality interval.'
      if (selected == 'A' .or. boundary_potential_v == 0.0_dp) return
      if (.not. ieee_is_finite(electron_normalization_m3)) return
      left_phi = nearest(boundary_potential_v, -1.0_dp)
      right_phi = nearest(boundary_potential_v, 1.0_dp)
      minimum = min(left_phi, 0.0_dp)
      left_density = neutral_electron_density(p, selected, left_phi, minimum)
      minimum = min(right_phi, 0.0_dp)
      right_density = neutral_electron_density(p, selected, right_phi, minimum)
      if (.not. all(ieee_is_finite([left_density, right_density]))) return
      if (electron_normalization_m3 < min(left_density, right_density) .or. &
          electron_normalization_m3 > max(left_density, right_density)) return
      density = electron_normalization_m3
    end if
    output%branch = selected
    output%boundary_potential_v = boundary_potential_v
    output%minimum_potential_v = phim*p%potential_scale_v
    output%ambient_electron_density_m3 = density
    call evaluate_zhao_rho_hat(p, selected, 'upper', 0.0_dp, phi0, phim, density/p%density_scale_m3, rho)
    output%neutrality_residual_m3 = rho*p%density_scale_m3
    factor = (p%potential_scale_v/p%length_scale_m)**2
    if (selected == 'A') then
      output%boundary_field_squared_v2_m2 = -2.0_dp*factor*integrate_zhao_rho(p, selected, 'lower', &
          phim, phi0, phi0, phim, density/p%density_scale_m3)
      output%connection_residual_v2_m2 = -2.0_dp*factor*integrate_zhao_rho(p, selected, 'upper', &
          phim, 0.0_dp, phi0, phim, density/p%density_scale_m3)
    else
      output%boundary_field_squared_v2_m2 = 2.0_dp*factor*integrate_zhao_rho(p, selected, 'monotonic', &
          phi0, 0.0_dp, phi0, phim, density/p%density_scale_m3)
    end if
    output%electric_field_v_m = ieee_value(0.0_dp, ieee_quiet_nan)
    if (output%boundary_field_squared_v2_m2 >= 0.0_dp) then
      output%electric_field_v_m = sqrt(output%boundary_field_squared_v2_m2)
      if (selected == 'C') then
        output%electric_field_v_m = -output%electric_field_v_m
      end if
    end if
    call evaluate_zhao_fluxes(p, boundary_potential_v, output%minimum_potential_v, density, &
        output%electron_inward_flux_m2_s, output%ion_inward_flux_m2_s, output%photoelectron_outward_flux_m2_s, &
        output%photoelectron_escape_flux_m2_s, output%photoelectron_return_flux_m2_s)
    output%net_current_a_m2 = qe*(output%electron_inward_flux_m2_s - output%ion_inward_flux_m2_s &
        - output%photoelectron_escape_flux_m2_s)
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Trial state has non-finite integrals, density, or fluxes.'
    if (.not. all(ieee_is_finite([density, output%neutrality_residual_m3, &
        output%boundary_field_squared_v2_m2, output%connection_residual_v2_m2, &
        output%electron_inward_flux_m2_s, output%ion_inward_flux_m2_s, output%photoelectron_outward_flux_m2_s, &
        output%photoelectron_escape_flux_m2_s, output%photoelectron_return_flux_m2_s, output%net_current_a_m2]))) return
    call validate_zhao_profile(p, selected, phi0, phim, density/p%density_scale_m3, minimum_e2, boundary_e2, &
        output%physical_status, output%physical_message)
    if (output%physical_status == SHEATH_NUMERICAL_FAILURE) then
      message = output%physical_message
      return
    end if
    output%evaluated = .true.
    output%admissible = output%physical_status == SHEATH_OK
    status = SHEATH_OK
    message = ''
  end subroutine
end module
