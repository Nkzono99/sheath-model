! SPDX-License-Identifier: MIT
!> Conditions that an outer stationary sheath imposes on a simulator's matching plane.
!! The matching plane is the outer model's boundary (z=0); the simulator domain lies below it.
!! Electrons crossing the plane outward, of any origin, form the outer model's emitted source.
!! Potentials are relative to upstream 0 V; number fluxes are positive in their named direction.
module sheath_model_matching
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
  use sheath_model_constants, only: dp, i32, qe, pi
  use sheath_model_status, only: SHEATH_OK, SHEATH_INVALID_ARGUMENT, SHEATH_NUMERICAL_FAILURE
  use sheath_model_core, only: zhao_params_type, evaluate_zhao_fluxes
  use sheath_model_state, only: prepare_plasma_params, boundary_field
  use sheath_model_equilibrium, only: sheath_equilibrium_input, sheath_equilibrium_result, &
      prepare_equilibrium_params => prepare_params
  use sheath_model_field, only: prescribed_field_input, prescribed_field_result
  implicit none
  private
  public :: sheath_matching_conditions, evaluate_matching_conditions
  public :: plane_electron_access_energy_ev, plane_ion_access_energy_ev
  public :: plane_electron_escape_threshold_ev, plane_ion_escape_threshold_ev
  public :: plane_electron_inward_flux_m2_s

  !> Plane potential/field, path extrema, upstream reservoirs and fluxes on the matching plane.
  !! A simulator combines the path extrema with its local plane potential: electrons need
  !! min(plane, electron_path_minimum_v) and ions max(plane, ion_path_maximum_v) as the outer extrema.
  type :: sheath_matching_conditions
    logical :: valid = .false.
    character(len=1) :: branch = ' '
    !> phi_H [V] and E(H+)=-dphi/dz [V/m]; E is positive for A/B and negative for C.
    real(dp) :: plane_potential_v = 0.0_dp
    real(dp) :: plane_field_v_m = 0.0_dp
    !> Potential to pass between the plane and upstream besides the plane itself [V]:
    !! the interior minimum phi_m for A, otherwise upstream 0 V.
    real(dp) :: electron_path_minimum_v = 0.0_dp
    real(dp) :: ion_path_maximum_v = 0.0_dp
    !> Upstream drifting Maxwellian; density is its normalization, not the total upstream density.
    real(dp) :: electron_reservoir_density_m3 = 0.0_dp
    real(dp) :: electron_temperature_ev = 0.0_dp
    real(dp) :: electron_drift_mps = 0.0_dp
    real(dp) :: electron_inward_flux_m2_s = 0.0_dp
    !> Upstream ion beam with the inward entry speed.
    real(dp) :: ion_reservoir_density_m3 = 0.0_dp
    real(dp) :: ion_entry_speed_mps = 0.0_dp
    real(dp) :: ion_temperature_ev = 0.0_dp
    real(dp) :: ion_inward_flux_m2_s = 0.0_dp
    !> Emitted electrons leaving the plane, split into escaping and returning parts.
    real(dp) :: emitted_outward_flux_m2_s = 0.0_dp
    real(dp) :: emitted_escape_flux_m2_s = 0.0_dp
    real(dp) :: emitted_return_flux_m2_s = 0.0_dp
    !> Conventional current along +z [A/m^2]; near zero for a J=0 root.
    real(dp) :: net_current_a_m2 = 0.0_dp
    !> Accepted upstream E^2<0 band [V] and field deficit [V/m] of drifting A/C J=0 roots.
    real(dp) :: upstream_negative_band_v = 0.0_dp
    real(dp) :: upstream_field_deficit_v_m = 0.0_dp
  end type sheath_matching_conditions

  !> Build matching conditions from a J=0 or prescribed-field root and the input it was solved for.
  !! The root's potentials and electron normalization are re-evaluated with the input; no search is made.
  interface evaluate_matching_conditions
    module procedure matching_from_equilibrium, matching_from_field
  end interface evaluate_matching_conditions

contains

  subroutine matching_from_equilibrium(input, root, conditions, status, message)
    class(sheath_equilibrium_input), intent(in) :: input
    type(sheath_equilibrium_result), intent(in) :: root
    type(sheath_matching_conditions), intent(out) :: conditions
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(zhao_params_type) :: p
    status = SHEATH_INVALID_ARGUMENT
    message = 'Matching conditions need a valid equilibrium root.'
    if (.not. root%valid) return
    call prepare_equilibrium_params(input, p, status, message)
    if (status /= SHEATH_OK) return
    call fill_conditions(p, root%branch, root%surface_potential_v, root%minimum_potential_v, &
        root%ambient_electron_density_m3, conditions, status, message)
    if (status /= SHEATH_OK) return
    conditions%upstream_negative_band_v = root%upstream_negative_band_v
    conditions%upstream_field_deficit_v_m = root%upstream_field_deficit_v_m
  end subroutine matching_from_equilibrium

  subroutine matching_from_field(input, root, conditions, status, message)
    type(prescribed_field_input), intent(in) :: input
    type(prescribed_field_result), intent(in) :: root
    type(sheath_matching_conditions), intent(out) :: conditions
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(zhao_params_type) :: p
    status = SHEATH_INVALID_ARGUMENT
    message = 'Matching conditions need a valid prescribed-field root.'
    if (.not. root%valid) return
    call prepare_plasma_params(input, p, status, message)
    if (status /= SHEATH_OK) return
    call fill_conditions(p, root%branch, root%boundary_potential_v, root%minimum_potential_v, &
        root%ambient_electron_density_m3, conditions, status, message)
    if (status /= SHEATH_OK) return
    conditions%upstream_negative_band_v = root%upstream_negative_band_v
    conditions%upstream_field_deficit_v_m = root%upstream_field_deficit_v_m
  end subroutine matching_from_field

  !> Evaluate fields and fluxes with the same internal expressions as the closures.
  subroutine fill_conditions(p, branch, phi0_v, phim_v, density_m3, conditions, status, message)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: phi0_v, phim_v, density_m3
    type(sheath_matching_conditions), intent(inout) :: conditions
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    real(dp) :: field_squared, field, electron, ion, outward, escape, returning
    status = SHEATH_INVALID_ARGUMENT
    message = 'Matching conditions need branch A, B, or C with finite potentials and positive density.'
    if (branch /= 'A' .and. branch /= 'B' .and. branch /= 'C') return
    if (.not. all(ieee_is_finite([phi0_v, phim_v, density_m3])) .or. density_m3 <= 0.0_dp) return
    call boundary_field(p, branch, phi0_v/p%potential_scale_v, phim_v/p%potential_scale_v, &
        density_m3/p%density_scale_m3, field_squared, field)
    call evaluate_zhao_fluxes(p, phi0_v, phim_v, density_m3, electron, ion, outward, escape, returning)
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Matching plane field or fluxes are non-finite.'
    if (.not. all(ieee_is_finite([field, electron, ion, outward, escape, returning]))) return
    message = 'Matching plane fluxes have unphysical signs.'
    if (electron <= 0.0_dp .or. ion <= 0.0_dp .or. min(outward, escape, returning) < 0.0_dp) return

    conditions%branch = branch
    conditions%plane_potential_v = phi0_v
    conditions%plane_field_v_m = field
    conditions%electron_path_minimum_v = 0.0_dp
    if (branch == 'A') conditions%electron_path_minimum_v = phim_v
    conditions%ion_path_maximum_v = 0.0_dp
    conditions%electron_reservoir_density_m3 = density_m3
    conditions%electron_temperature_ev = p%t_swe_ev
    conditions%electron_drift_mps = p%v_d_electron_mps
    conditions%electron_inward_flux_m2_s = electron
    conditions%ion_reservoir_density_m3 = p%n_swi_inf_m3
    conditions%ion_entry_speed_mps = p%v_d_ion_mps
    conditions%ion_temperature_ev = p%t_swi_ev
    conditions%ion_inward_flux_m2_s = ion
    conditions%emitted_outward_flux_m2_s = outward
    conditions%emitted_escape_flux_m2_s = escape
    conditions%emitted_return_flux_m2_s = returning
    conditions%net_current_a_m2 = qe*(electron - ion - escape)
    conditions%valid = .true.
    status = SHEATH_OK
    message = ''
  end subroutine fill_conditions


  !> Normal energy [eV] an upstream electron needs at infinity to reach a plane at local potential plane_potential_v:
  !! -min(plane_potential_v, electron_path_minimum_v, 0). Zero when no potential repels it.
  pure real(dp) function plane_electron_access_energy_ev(conditions, plane_potential_v) result(energy_ev)
    type(sheath_matching_conditions), intent(in) :: conditions
    real(dp), intent(in) :: plane_potential_v
    energy_ev = -min(plane_potential_v, conditions%electron_path_minimum_v, 0.0_dp)
  end function plane_electron_access_energy_ev

  !> Normal energy [eV] an upstream ion needs at infinity to reach a plane at local potential plane_potential_v:
  !! max(plane_potential_v, ion_path_maximum_v, 0).
  pure real(dp) function plane_ion_access_energy_ev(conditions, plane_potential_v) result(energy_ev)
    type(sheath_matching_conditions), intent(in) :: conditions
    real(dp), intent(in) :: plane_potential_v
    energy_ev = max(plane_potential_v, conditions%ion_path_maximum_v, 0.0_dp)
  end function plane_ion_access_energy_ev

  !> Normal energy [eV] at the plane above which an electron leaving outward passes the outer sheath:
  !! plane_potential_v - min(plane_potential_v, electron_path_minimum_v). Below it the electron returns with the same
  !! normal energy.
  pure real(dp) function plane_electron_escape_threshold_ev(conditions, plane_potential_v) result(energy_ev)
    type(sheath_matching_conditions), intent(in) :: conditions
    real(dp), intent(in) :: plane_potential_v
    energy_ev = plane_potential_v - min(plane_potential_v, conditions%electron_path_minimum_v)
  end function plane_electron_escape_threshold_ev

  !> Normal energy [eV] at the plane above which an ion leaving outward passes the outer sheath:
  !! max(plane_potential_v, ion_path_maximum_v) - plane_potential_v.
  pure real(dp) function plane_ion_escape_threshold_ev(conditions, plane_potential_v) result(energy_ev)
    type(sheath_matching_conditions), intent(in) :: conditions
    real(dp), intent(in) :: plane_potential_v
    energy_ev = max(plane_potential_v, conditions%ion_path_maximum_v) - plane_potential_v
  end function plane_ion_escape_threshold_ev

  !> Number flux [1/(m^2 s)] of upstream electrons reaching a plane at local potential plane_potential_v: the
  !! inward-drifting Maxwellian of the conditions restricted to the access energy,
  !! N_e v_th/(2 sqrt(pi)) [exp(-(b-u)^2) + sqrt(pi) u erfc(b-u)] with v_th = sqrt(2 e T_e/m_e),
  !! b = sqrt(K_access/T_e) and u = v_d/v_th. At the plane potential of the conditions it equals
  !! electron_inward_flux_m2_s. NaN for a nonpositive temperature or mass.
  pure real(dp) function plane_electron_inward_flux_m2_s(conditions, plane_potential_v, electron_mass_kg) result(flux)
    type(sheath_matching_conditions), intent(in) :: conditions
    real(dp), intent(in) :: plane_potential_v, electron_mass_kg
    real(dp) :: thermal_speed, b, u
    flux = ieee_value(0.0_dp, ieee_quiet_nan)
    if (.not. (conditions%electron_temperature_ev > 0.0_dp .and. electron_mass_kg > 0.0_dp)) return
    thermal_speed = sqrt(2.0_dp*qe*conditions%electron_temperature_ev/electron_mass_kg)
    b = sqrt(plane_electron_access_energy_ev(conditions, plane_potential_v)/conditions%electron_temperature_ev)
    u = conditions%electron_drift_mps/thermal_speed
    flux = conditions%electron_reservoir_density_m3*thermal_speed/(2.0_dp*sqrt(pi))* &
        (exp(-(b - u)**2) + sqrt(pi)*u*erfc(b - u))
  end function plane_electron_inward_flux_m2_s

end module sheath_model_matching
