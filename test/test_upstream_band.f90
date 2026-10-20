program test_upstream_band
  ! Drifting reflected electrons: A/C roots are rejected by default and accepted, with the measured
  ! negative-E^2 band, when upstream_band_tolerance allows it. Reference values are algebraic roots
  ! solved independently in Python without the semi-infinite upstream check.
  use sheath_model
  use sheath_model_emission, only: DEFAULT_EMISSION
  implicit none
  real(dp), parameter :: pi = acos(-1.0_dp), qe = 1.602176634e-19_dp, me = 9.1093837139e-31_dp
  type(sheath_solver) :: strict, tolerant, narrow, floor, tight
  type(fixed_entry_equilibrium_input) :: input
  type(sheath_equilibrium_result) :: root, reference
  type(sheath_profile_result) :: profile
  type(sheath_matching_conditions) :: conditions
  type(prescribed_field_input) :: field_input
  type(prescribed_field_result) :: field_root
  type(sheath_matching_conditions) :: plane
  integer(i32) :: status
  character(len=256) :: message
  real(dp) :: source_density

  ! Regolith test conditions: n=5 cm^-3, Te=10 eV, 400 km/s normal wind, 4.5 uA/m^2 photoelectrons at 2.2 eV.
  source_density = 2.0_dp*sqrt(pi)*(4.5e-6_dp/qe)/sqrt(2.0_dp*qe*2.2_dp/me)
  input%plasma = plasma_input(ion_density_m3=5.0e6_dp, electron_temperature_ev=10.0_dp, &
      electron_drift_mps=4.0e5_dp, ion_entry_speed_mps=4.0e5_dp, &
      emission=maxwellian_emission(source_density, 2.2_dp))
  tolerant%search%upstream_band_tolerance = 0.15_dp
  narrow%search%upstream_band_tolerance = 0.02_dp
  ! Field floor: the upstream boundary sits where the sheath field has fallen to an external level.
  ! The solar-wind motional field |u x B| is about 2 mV/m for 400 km/s and 5 nT.
  floor%search%upstream_field_floor_v_m = 1.0e-3_dp
  tight%search%upstream_field_floor_v_m = 1.0e-4_dp

  input%branch = 'A'
  call strict%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_NO_PHYSICAL_SOLUTION .or. root%valid) error stop 'strict search must reject drifting A'

  call tolerant%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%branch /= 'A') error stop 'tolerant search must accept drifting A'
  if (abs(root%surface_potential_v - 6.0383_dp) > 1e-3_dp) error stop 'drifting A surface potential'
  if (abs(root%minimum_potential_v + 0.7862_dp) > 1e-3_dp) error stop 'drifting A minimum potential'
  if (root%upstream_negative_band_v < 0.040_dp .or. root%upstream_negative_band_v > 0.065_dp) &
      error stop 'drifting A negative band width'
  if (abs(root%net_current_a_m2) > 1e-8_dp*qe*root%ion_inward_flux_m2_s) error stop 'drifting A zero current'
  ! The plane flux helper carries the drift term: it reproduces the inward flux of a drifting root.
  call evaluate_matching_conditions(input, root, conditions, status, message)
  if (status /= SHEATH_OK) error stop 'drifting A matching conditions'
  if (abs(plane_electron_inward_flux_m2_s(conditions, conditions%plane_potential_v, input%plasma%electron_mass_kg) - &
      conditions%electron_inward_flux_m2_s) > 1e-12_dp*conditions%electron_inward_flux_m2_s) then
    error stop 'drifting A plane inward flux identity'
  end if

  call tolerant%build_profile(input, root, profile, status, message)
  if (status /= SHEATH_OK) error stop 'profile of an accepted drifting A root'
  if (profile%potential_v(size(profile%potential_v)) > -root%upstream_negative_band_v) &
      error stop 'profile must stop before the negative band'

  call narrow%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_NO_PHYSICAL_SOLUTION) error stop 'band wider than the tolerance must be rejected'

  ! The field deficit inside the A band is about 0.67 mV/m, below the motional field.
  call floor%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%branch /= 'A') error stop 'field floor must accept drifting A'
  if (abs(root%surface_potential_v - 6.0383_dp) > 1e-3_dp) error stop 'field-floor A surface potential'
  if (root%upstream_field_deficit_v_m < 5.0e-4_dp .or. root%upstream_field_deficit_v_m > 9.0e-4_dp) &
      error stop 'drifting A field deficit'
  if (root%upstream_negative_band_v < 0.040_dp .or. root%upstream_negative_band_v > 0.065_dp) &
      error stop 'field-floor A must still report the band width'
  call tight%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_NO_PHYSICAL_SOLUTION) error stop 'a deficit above the field floor must be rejected'

  ! The prescribed-field closure applies the same band criteria: imposing the boundary field of the
  ! accepted J=0 root recovers that root, and the strict search still rejects it.
  call floor%solve_equilibrium(input, root, status, message)
  call evaluate_matching_conditions(input, root, plane, status, message)
  if (status /= SHEATH_OK) error stop 'plane conditions of the drifting A root'
  field_input%plasma_input = input%plasma
  field_input%branch = 'A'
  field_input%electric_field_v_m = plane%plane_field_v_m
  call floor%solve_prescribed_field(field_input, field_root, status, message)
  if (status /= SHEATH_OK .or. field_root%branch /= 'A') error stop 'field floor must accept drifting A in the field closure'
  if (abs(field_root%boundary_potential_v - root%surface_potential_v) > 1e-6_dp) &
      error stop 'field closure must recover the J=0 surface potential'
  if (abs(field_root%net_current_a_m2) > 1e-6_dp*qe*root%ion_inward_flux_m2_s) &
      error stop 'field closure at the J=0 field must give zero current'
  if (abs(field_root%upstream_field_deficit_v_m - root%upstream_field_deficit_v_m) > 1e-6_dp .or. &
      abs(field_root%upstream_negative_band_v - root%upstream_negative_band_v) > 1e-6_dp) &
      error stop 'field closure must report the same upstream band'
  call strict%solve_prescribed_field(field_input, field_root, status, message)
  if (status == SHEATH_OK) error stop 'strict field closure must reject drifting A'

  input%branch = 'auto'
  call tolerant%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%branch /= 'A') error stop 'auto with tolerance selects A'

  input%branch = 'C'
  input%plasma%emission = DEFAULT_EMISSION
  call strict%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_NO_PHYSICAL_SOLUTION) error stop 'strict search must reject drifting C'
  call tolerant%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%branch /= 'C') error stop 'tolerant search must accept drifting C'
  if (abs(root%surface_potential_v + 7.3403_dp) > 1e-3_dp) error stop 'drifting C surface potential'
  if (root%upstream_negative_band_v <= 0.0_dp .or. root%upstream_negative_band_v > 0.005_dp) &
      error stop 'drifting C negative band width'
  ! The C band is shallow: its deficit is about 0.04 mV/m, so even the tight floor accepts it.
  call tight%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%branch /= 'C') error stop 'tight field floor must accept drifting C'
  if (root%upstream_field_deficit_v_m < 2.0e-5_dp .or. root%upstream_field_deficit_v_m > 6.0e-5_dp) &
      error stop 'drifting C field deficit'

  ! Without drift the tolerance changes nothing.
  input%branch = 'A'
  input%plasma%electron_drift_mps = 0.0_dp
  input%plasma%emission = maxwellian_emission(source_density, 2.2_dp)
  call strict%solve_equilibrium(input, reference, status, message)
  if (status /= SHEATH_OK) error stop 'zero-drift A'
  call tolerant%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%surface_potential_v /= reference%surface_potential_v .or. &
      root%upstream_negative_band_v /= 0.0_dp) error stop 'tolerance must not change zero-drift roots'
  call floor%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%surface_potential_v /= reference%surface_potential_v .or. &
      root%upstream_field_deficit_v_m /= 0.0_dp) error stop 'field floor must not change zero-drift roots'
  floor%search%upstream_field_floor_v_m = -1.0_dp
  call floor%solve_equilibrium(input, root, status, message)
  if (status == SHEATH_OK) error stop 'field floor must be nonnegative'

  tolerant%search%upstream_band_tolerance = 1.0_dp
  call tolerant%solve_equilibrium(input, root, status, message)
  if (status == SHEATH_OK) error stop 'tolerance must be below 1'

  print *, 'Upstream band checks passed.'
end program test_upstream_band
