program test_warm_ions
  use sheath_model
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(zhao_equilibrium_input) :: input
  type(zhao_equilibrium_result) :: cold, warm
  type(zhao_density_result) :: density
  type(zhao_plasma_input) :: plasma
  type(zhao_state_result) :: state
  type(zhao_field_input) :: field_input
  type(zhao_field_result), allocatable :: candidates(:)
  real(dp) :: ratio, phi, critical, pressure, expected, gap
  real(dp), parameter :: ratios(5) = [0.6_dp, 0.9_dp, 1.0_dp, 1.2_dp, 2.0_dp]
  integer(i32) :: status
  integer :: i
  logical :: recovered
  character(len=256) :: message

  do i = 1, size(ratios)
    ratio = ratios(i)
    phi = 20.0_dp*(1.0_dp - 1.0_dp/ratio**2) - 5.0_dp*log(ratio)
    if (abs(ion_density_ratio(phi, 20.0_dp, 5.0_dp) - ratio) > 2e-13_dp) error stop 'ion energy/continuity'
    if (abs(ion_density_ratio(phi, 20.0_dp, 0.0_dp) - 1.0_dp/sqrt(1.0_dp - phi/20.0_dp)) > &
        1e-14_dp) error stop 'cold limit'
  end do
  critical = ion_critical_potential(20.0_dp, 5.0_dp)
  if (ion_density_ratio(critical, 20.0_dp, 5.0_dp) /= sqrt(8.0_dp)) error stop 'sonic endpoint'
  if (ieee_is_finite(ion_density_ratio(critical + 1e-8_dp, 20.0_dp, 5.0_dp))) error stop 'blocked warm ions'
  if (ieee_is_finite(ion_density_ratio(20.0_dp, 20.0_dp, 0.0_dp))) error stop 'divergent cold endpoint'
  if (ieee_is_finite(ion_density_ratio(0.0_dp, 20.0_dp, 40.0_dp))) error stop 'subsonic entry'
  gap = 1e-7_dp
  expected = 0.5_dp*(gap**2/2.0_dp - gap**3/3.0_dp)
  if (abs(ion_critical_potential(0.5_dp*(1.0_dp + gap), 1.0_dp) - expected) > 3e-24_dp) &
      error stop 'near-sonic cancellation'
  do i = 1, 3, 2
    pressure = real(i, dp)*150.0_dp
    expected = 7.5_dp*(1.0_dp - real(i, dp)*10.0_dp*log(1.0_dp + 1.0_dp/(real(i, dp)*10.0_dp)))
    if (abs(ion_critical_potential(0.5_dp*(15.0_dp + pressure), pressure) - expected) > 2e-13_dp) &
        error stop 'paper critical voltage'
  end do
  input = zhao_equilibrium_input(branch='A', electron_drift_mode='zero')
  call solve_equilibrium(input, cold, status, message)
  if (status /= SHEATH_OK) error stop 'cold reference'
  input%ion_temperature_ev = 12.0_dp
  input%ion_pressure_factor = 3.0_dp
  call solve_equilibrium(input, warm, status, message)
  if (status /= SHEATH_OK) then
    print *, trim(message)
    error stop 'warm equilibrium'
  end if
  if (abs(warm%minimum_potential_v - cold%minimum_potential_v) < 1e-6_dp) error stop 'pressure has no effect'
  if (abs(warm%net_current_a_m2) > 1e-12_dp) error stop 'warm current balance'
  call evaluate_density(input, warm, warm%surface_potential_v, density, status, message, side='lower')
  if (status /= SHEATH_OK .or. density%ion_m3 <= 0.0_dp) error stop 'warm public density'
  plasma%electron_drift_mps = 0.0_dp
  plasma%ion_temperature_ev = 12.0_dp
  plasma%ion_pressure_factor = 3.0_dp
  plasma%photoelectrons = maxwellian_photoelectrons(64e6_dp*sin(acos(-1.0_dp)/3.0_dp), 2.2_dp)
  call evaluate_sheath_state(plasma, 'A', warm%surface_potential_v, state, status, message, &
      minimum_potential_v=warm%minimum_potential_v)
  if (status /= SHEATH_OK .or. .not. state%admissible) error stop 'warm static closure'
  if (abs(state%net_current_a_m2) > 1e-12_dp) error stop 'warm static current'
  field_input%electron_drift_mps = plasma%electron_drift_mps
  field_input%ion_temperature_ev = plasma%ion_temperature_ev
  field_input%ion_pressure_factor = plasma%ion_pressure_factor
  field_input%photoelectrons = plasma%photoelectrons
  field_input%branch = 'A'
  field_input%electric_field_v_m = state%electric_field_v_m
  call solve_prescribed_field_candidates(field_input, candidates, status, message)
  if (status /= SHEATH_OK) error stop 'warm prescribed field'
  recovered = .false.
  do i = 1, size(candidates)
    if (abs(candidates(i)%boundary_potential_v - warm%surface_potential_v) < 1e-5_dp .and. &
        abs(candidates(i)%minimum_potential_v - warm%minimum_potential_v) < 1e-5_dp) recovered = .true.
  end do
  if (.not. recovered) error stop 'warm field closure failed to recover equilibrium'
  input%branch = 'B'
  input%sun_elevation_deg = 20.0_dp
  input%ion_temperature_ev = 1.0_dp
  call solve_equilibrium(input, warm, status, message)
  if (status /= SHEATH_OK) error stop 'warm B'
  input%branch = 'C'
  input%sun_elevation_deg = 10.0_dp
  call solve_equilibrium(input, warm, status, message)
  if (status /= SHEATH_OK) error stop 'warm C'
  input%ion_temperature_ev = -1.0_dp
  call solve_equilibrium(input, warm, status, message)
  if (status /= SHEATH_INVALID_ARGUMENT) error stop 'negative ion temperature'
  plasma%ion_drift_mps = 1.0_dp
  call evaluate_sheath_state(plasma, 'C', -1.0_dp, state, status, message)
  if (status /= SHEATH_INVALID_ARGUMENT) error stop 'subsonic input'
end program test_warm_ions
