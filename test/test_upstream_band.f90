program test_upstream_band
  ! Drifting reflected electrons: A/C roots are rejected by default and accepted, with the measured
  ! negative-E^2 band, when upstream_band_tolerance allows it. Reference values are algebraic roots
  ! solved independently in Python without the semi-infinite upstream check.
  use sheath_model
  use sheath_model_photoelectrons, only: DEFAULT_PHOTOELECTRONS
  implicit none
  real(dp), parameter :: pi = acos(-1.0_dp), qe = 1.602176634e-19_dp, me = 9.1093837139e-31_dp
  type(sheath_solver) :: strict, tolerant, narrow
  type(fixed_entry_equilibrium_input) :: input
  type(sheath_equilibrium_result) :: root, reference
  type(sheath_profile_result) :: profile
  integer(i32) :: status
  character(len=256) :: message
  real(dp) :: source_density

  ! Regolith test conditions: n=5 cm^-3, Te=10 eV, 400 km/s normal wind, 4.5 uA/m^2 photoelectrons at 2.2 eV.
  source_density = 2.0_dp*sqrt(pi)*(4.5e-6_dp/qe)/sqrt(2.0_dp*qe*2.2_dp/me)
  input%plasma = plasma_input(ion_density_m3=5.0e6_dp, electron_temperature_ev=10.0_dp, &
      electron_drift_mps=4.0e5_dp, ion_entry_speed_mps=4.0e5_dp, &
      photoelectrons=maxwellian_photoelectrons(source_density, 2.2_dp))
  tolerant%search%upstream_band_tolerance = 0.15_dp
  narrow%search%upstream_band_tolerance = 0.02_dp

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

  call tolerant%build_profile(input, root, profile, status, message)
  if (status /= SHEATH_OK) error stop 'profile of an accepted drifting A root'
  if (profile%potential_v(size(profile%potential_v)) > -root%upstream_negative_band_v) &
      error stop 'profile must stop before the negative band'

  call narrow%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_NO_PHYSICAL_SOLUTION) error stop 'band wider than the tolerance must be rejected'

  input%branch = 'auto'
  call tolerant%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%branch /= 'A') error stop 'auto with tolerance selects A'

  input%branch = 'C'
  input%plasma%photoelectrons = DEFAULT_PHOTOELECTRONS
  call strict%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_NO_PHYSICAL_SOLUTION) error stop 'strict search must reject drifting C'
  call tolerant%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%branch /= 'C') error stop 'tolerant search must accept drifting C'
  if (abs(root%surface_potential_v + 7.3403_dp) > 1e-3_dp) error stop 'drifting C surface potential'
  if (root%upstream_negative_band_v <= 0.0_dp .or. root%upstream_negative_band_v > 0.005_dp) &
      error stop 'drifting C negative band width'

  ! Without drift the tolerance changes nothing.
  input%branch = 'A'
  input%plasma%electron_drift_mps = 0.0_dp
  input%plasma%photoelectrons = maxwellian_photoelectrons(source_density, 2.2_dp)
  call strict%solve_equilibrium(input, reference, status, message)
  if (status /= SHEATH_OK) error stop 'zero-drift A'
  call tolerant%solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%surface_potential_v /= reference%surface_potential_v .or. &
      root%upstream_negative_band_v /= 0.0_dp) error stop 'tolerance must not change zero-drift roots'

  tolerant%search%upstream_band_tolerance = 1.0_dp
  call tolerant%solve_equilibrium(input, root, status, message)
  if (status == SHEATH_OK) error stop 'tolerance must be below 1'

  print *, 'Upstream band checks passed.'
end program test_upstream_band
