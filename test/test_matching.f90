! SPDX-License-Identifier: MIT
! Matching-plane conditions must agree with the closures, profiles and independent flux identities.
program test_matching
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan, ieee_is_nan
  use sheath_model
  use sheath_model_constants, only: electron_mass
  implicit none
  type(sheath_solver) :: solver
  real(dp), parameter :: qe = 1.602176634e-19_dp
  real(dp), parameter :: elevations(3) = [60.0_dp, 20.0_dp, 10.0_dp]
  character(len=1), parameter :: branches(3) = ['A', 'B', 'C']
  type(zhao_equilibrium_input) :: input
  type(sheath_profile_result) :: profile
  type(sheath_equilibrium_result) :: root
  type(sheath_matching_conditions) :: conditions
  type(sheath_state_result) :: state
  type(prescribed_field_input) :: field_input
  type(prescribed_field_result) :: field_root
  type(prescribed_field_result), allocatable :: field_roots(:)
  type(emission_source) :: source
  integer(i32) :: status
  integer :: i
  real(dp), allocatable :: edges(:), breakpoints(:)
  real(dp) :: flux(145), tolerance_v, current_scale
  character(len=512) :: message

  do i = 1, 3
    input = zhao_equilibrium_input(branch=branches(i), sun_elevation_deg=elevations(i), electron_drift_mode='zero')
    call solver%solve_profile(input, profile, status, message)
    call ok('profile '//branches(i))
    root = profile%equilibrium
    call evaluate_matching_conditions(input, root, conditions, status, message)
    call ok('J=0 matching '//branches(i))
    call check(conditions%valid .and. conditions%branch == branches(i), 'matching branch')
    call check(conditions%plane_potential_v == root%surface_potential_v, 'plane potential is the root wall potential')
    ! Same expressions as the closure: fluxes reproduce the root exactly up to rounding.
    call near(conditions%electron_inward_flux_m2_s, root%electron_inward_flux_m2_s, 1e-13_dp, 'electron flux')
    call near(conditions%ion_inward_flux_m2_s, root%ion_inward_flux_m2_s, 1e-13_dp, 'ion flux')
    call near(conditions%emitted_escape_flux_m2_s, root%emitted_escape_flux_m2_s, 1e-13_dp, 'escape flux')
    call near(conditions%emitted_outward_flux_m2_s, &
        conditions%emitted_escape_flux_m2_s + conditions%emitted_return_flux_m2_s, 1e-14_dp, 'emitted flux split')
    current_scale = qe*max(conditions%electron_inward_flux_m2_s, conditions%ion_inward_flux_m2_s)
    call check(abs(conditions%net_current_a_m2) <= 1e-8_dp*current_scale, 'J=0 on the plane')
    call check(conditions%electron_reservoir_density_m3 == root%ambient_electron_density_m3, 'electron reservoir')
    call check(conditions%ion_reservoir_density_m3 == input%ion_density_m3, 'ion reservoir')
    ! Field and path extrema against the independently integrated profile.
    ! The profile integrates the first integral by quadrature, independently of the closed-form boundary field.
    call near(conditions%plane_field_v_m, profile%electric_field_v_m(1), 1e-4_dp, 'plane field')
    if (branches(i) == 'C') then
      call check(conditions%plane_field_v_m < 0.0_dp, 'C field sign')
    else
      call check(conditions%plane_field_v_m > 0.0_dp, 'A/B field sign')
    end if
    ! Profiles stop at a potential cutoff, so the upstream end is compared within it.
    tolerance_v = 2.0_dp*solver%profile%potential_cutoff_v
    call check(abs(min(conditions%plane_potential_v, conditions%electron_path_minimum_v) - &
        minval(profile%potential_v)) <= tolerance_v, 'electron outer minimum')
    call check(abs(max(conditions%plane_potential_v, conditions%ion_path_maximum_v) - &
        maxval(profile%potential_v)) <= tolerance_v, 'ion outer maximum')
    if (branches(i) == 'A') then
      call check(conditions%electron_path_minimum_v == root%minimum_potential_v, 'A bottleneck')
    else
      call check(conditions%electron_path_minimum_v == 0.0_dp, 'B/C upstream access')
    end if
    ! Plane rules at the plane potential reproduce the root: the access-restricted Maxwellian gives the
    ! electron inward flux, and the escape threshold is the potential climb above the plane.
    call near(plane_electron_inward_flux_m2_s(conditions, conditions%plane_potential_v, electron_mass), &
        conditions%electron_inward_flux_m2_s, 1e-12_dp, 'plane inward flux identity '//branches(i))
    call check(plane_electron_access_energy_ev(conditions, conditions%plane_potential_v) == &
        -min(conditions%plane_potential_v, conditions%electron_path_minimum_v, 0.0_dp), 'electron access energy')
    call check(plane_ion_access_energy_ev(conditions, conditions%plane_potential_v) == &
        max(conditions%plane_potential_v, 0.0_dp), 'ion access energy')
    call check(plane_ion_escape_threshold_ev(conditions, conditions%plane_potential_v) == &
        max(conditions%plane_potential_v, 0.0_dp) - conditions%plane_potential_v, 'ion escape threshold')
    if (branches(i) == 'A') then
      call check(conditions%plane_potential_v > 0.0_dp .and. conditions%electron_path_minimum_v < 0.0_dp, 'A geometry')
      call near(plane_electron_escape_threshold_ev(conditions, conditions%plane_potential_v), &
          conditions%plane_potential_v - conditions%electron_path_minimum_v, 1e-15_dp, 'A escape threshold is the climb')
      ! The well, not the plane, sets the electron access of a Type A root.
      call near(plane_electron_inward_flux_m2_s(conditions, 0.5_dp*conditions%plane_potential_v, electron_mass), &
          conditions%electron_inward_flux_m2_s, 1e-15_dp, 'A access is set by the well')
    else if (conditions%plane_potential_v >= 0.0_dp) then
      call near(plane_electron_escape_threshold_ev(conditions, conditions%plane_potential_v), &
          conditions%plane_potential_v, 1e-15_dp, 'B escape threshold is the plane potential')
    else
      call check(plane_electron_escape_threshold_ev(conditions, conditions%plane_potential_v) == 0.0_dp, &
          'C has no escape barrier')
      ! A more negative local plane admits fewer upstream electrons.
      call check(plane_electron_inward_flux_m2_s(conditions, conditions%plane_potential_v - 1.0_dp, electron_mass) < &
          conditions%electron_inward_flux_m2_s, 'C inward flux falls with a more negative plane')
    end if
  end do
  conditions%electron_temperature_ev = 0.0_dp
  call check(ieee_is_nan(plane_electron_inward_flux_m2_s(conditions, 0.0_dp, electron_mass)), 'invalid temperature is NaN')

  ! Prescribed-field roots give the same conditions, with the plane field equal to the prescribed one.
  input = zhao_equilibrium_input(branch='B', sun_elevation_deg=20.0_dp, electron_drift_mode='zero')
  call solver%solve_equilibrium(input, root, status, message)
  call ok('B reference')
  field_input%branch = 'B'
  field_input%electron_drift_mps = 0.0_dp
  field_input%ion_entry_speed_mps = input%solar_wind_speed_mps*sin(20.0_dp*acos(-1.0_dp)/180.0_dp)
  field_input%emission = maxwellian_emission( &
      input%photoelectron_reference_density_m3*sin(20.0_dp*acos(-1.0_dp)/180.0_dp), 2.2_dp)
  call evaluate_sheath_state(field_input%plasma_input, 'B', root%surface_potential_v, state, status, message)
  call ok('B state')
  field_input%electric_field_v_m = state%electric_field_v_m
  call solver%solve_prescribed_field_candidates(field_input, field_roots, status, message)
  call ok('B prescribed field')
  field_root = field_roots(minloc(abs(field_roots%boundary_potential_v - root%surface_potential_v), dim=1))
  call check(abs(field_root%boundary_potential_v - root%surface_potential_v) < 1e-6_dp, 'field root recovers J=0 root')
  call evaluate_matching_conditions(field_input, field_root, conditions, status, message)
  call ok('field matching')
  call near(conditions%plane_field_v_m, field_input%electric_field_v_m, 1e-6_dp, 'prescribed plane field')
  call near(conditions%plane_potential_v, field_root%boundary_potential_v, 1e-15_dp, 'field plane potential')
  call near(conditions%electron_inward_flux_m2_s, field_root%electron_inward_flux_m2_s, 1e-13_dp, 'field electron flux')
  call near(conditions%emitted_return_flux_m2_s, field_root%emitted_return_flux_m2_s, 1e-13_dp, 'field return')
  call near(conditions%net_current_a_m2, field_root%net_current_a_m2, 1e-10_dp, 'field current')

  root%valid = .false.
  call evaluate_matching_conditions(input, root, conditions, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT .and. .not. conditions%valid, 'invalid root rejected')

  ! Outflow energy grid and observed sources.
  edges = log_energy_edges()
  call check(size(edges) == 146 .and. LOG_ENERGY_DEFAULT_BINS == 145 .and. size(edges) == LOG_ENERGY_DEFAULT_BINS + 1, &
      'default grid size')
  call check(log_energy_bin_count() == LOG_ENERGY_DEFAULT_BINS .and. log_energy_bin_count(decades=2, bins_per_decade=10) == 21 &
      .and. log_energy_bin_count(decades=0) == 0, 'bin count')
  call check(size(log_energy_edges(decades=2, bins_per_decade=10)) == 22, 'custom grid size matches bin count')
  call check(edges(1) == 0.0_dp .and. edges(2) == 1e-3_dp, 'default grid start')
  call near(edges(146), 1e3_dp, 1e-14_dp, 'default grid end')
  call check(all(edges(2:) > edges(:145)), 'ascending grid')
  call check(size(log_energy_edges(decades=0)) == 0, 'invalid grid')
  call check(energy_bin(edges, 0.0_dp) == 1 .and. energy_bin(edges, 5e-4_dp) == 1, 'first bin')
  call check(energy_bin(edges, 1e-3_dp) == 2, 'lower edge belongs to its bin')
  call check(energy_bin(edges, -1.0_dp) == 1, 'negative energy goes to the first bin')
  call check(energy_bin(edges, ieee_value(0.0_dp, ieee_quiet_nan)) == 1, 'NaN goes to the first bin')
  call check(energy_bin(edges, 1e3_dp) == 145 .and. energy_bin(edges, 1e6_dp) == 145, 'last bin')
  do i = 1, 145
    call check(energy_bin(edges, 0.5_dp*(edges(i) + edges(i + 1))) == i, 'midpoint bin')
  end do
  call check(energy_bin([0.0_dp], 1.0_dp) == 0, 'no bins')
  ! The O(1) log-grid bin agrees with the edge search everywhere, including on the edges themselves.
  do i = 1, 146
    call check(log_energy_bin(edges(i)) == energy_bin(edges, edges(i)), 'log bin on edge')
    call check(log_energy_bin(nearest(edges(i), -1.0_dp)) == energy_bin(edges, nearest(edges(i), -1.0_dp)), &
        'log bin just below an edge')
  end do
  do i = 1, 145
    call check(log_energy_bin(0.5_dp*(edges(i) + edges(i + 1))) == i, 'log bin midpoint')
  end do
  call check(log_energy_bin(0.0_dp) == 1 .and. log_energy_bin(-1.0_dp) == 1 .and. &
      log_energy_bin(ieee_value(0.0_dp, ieee_quiet_nan)) == 1, 'log bin low side')
  call check(log_energy_bin(1e3_dp) == 145 .and. log_energy_bin(1e6_dp) == 145, 'log bin high side')
  call check(log_energy_bin(1.0_dp, decades=0) == 0, 'log bin without a grid')
  breakpoints = log_energy_edges(lowest_edge_ev=0.5_dp, decades=2, bins_per_decade=10)
  do i = 1, size(breakpoints) - 1
    call check(log_energy_bin(0.5_dp*(breakpoints(i) + breakpoints(i + 1)), lowest_edge_ev=0.5_dp, decades=2, &
        bins_per_decade=10) == i, 'custom log bin midpoint')
    call check(log_energy_bin(breakpoints(i), lowest_edge_ev=0.5_dp, decades=2, bins_per_decade=10) == &
        energy_bin(breakpoints, breakpoints(i)), 'custom log bin edge')
  end do

  flux = 0.0_dp
  flux(10:12) = [1e10_dp, 0.0_dp, 2e10_dp]
  flux(40) = 5e9_dp
  call observed_outflow_source(edges, flux, source, status, message)
  call ok('observed source')
  call check(source%is_binned(), 'observed source is binned')
  breakpoints = source%search_breakpoints()
  call check(size(breakpoints) == 32, 'empty end bins trimmed, interior bins kept')
  call check(breakpoints(1) == edges(10) .and. breakpoints(32) == edges(41), 'trimmed edges')
  flux = 0.0_dp
  call observed_outflow_source(edges, flux, source, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'empty outflow rejected')
  flux(5) = -1.0_dp
  flux(6) = 2.0_dp
  call observed_outflow_source(edges, flux, source, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'negative outflow rejected')
  call observed_outflow_source(edges(:100), flux, source, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'edge count mismatch rejected')

  print *, 'Matching-plane conditions and outflow sources passed.'
contains
  subroutine ok(label)
    character(len=*), intent(in) :: label
    call check(status == SHEATH_OK, label)
  end subroutine
  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, 'FAIL: ', label, ' ', trim(message)
      error stop 1
    end if
  end subroutine
  subroutine near(actual, expected, tolerance, label)
    real(dp), intent(in) :: actual, expected, tolerance
    character(len=*), intent(in) :: label
    call check(ieee_is_finite(actual), label)
    call check(abs(actual - expected) <= tolerance*max(abs(expected), tiny(1.0_dp)), label)
  end subroutine
end program
