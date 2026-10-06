program test_fixed_entry
  use sheath_model
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  implicit none
  type(fixed_entry_equilibrium_input) :: input
  type(zhao_equilibrium_input) :: solar
  type(zhao_equilibrium_result) :: root, reference
  type(zhao_density_result) :: density
  type(zhao_state_result) :: state
  type(zhao_profile_options) :: options
  type(zhao_profile_result) :: profile
  type(zhao_field_input) :: field
  type(zhao_field_result), allocatable :: candidates(:)
  real(dp), parameter :: elevations(3) = [60.0_dp, 20.0_dp, 10.0_dp]
  character(len=1), parameter :: branches(3) = ['A', 'B', 'C']
  real(dp) :: pressure, energy, critical, expected, phi
  integer(i32) :: status
  integer :: b, hot, i
  logical :: recovered
  character(len=256) :: message

  call solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%branch /= 'C') error stop 'fixed-entry defaults'
  if (root%photoelectron_escape_flux_m2_s /= 0.0_dp) error stop 'default emission'
  options%points_per_segment = 512
  do b = 1, 3
    solar = zhao_equilibrium_input(branch=branches(b), sun_elevation_deg=elevations(b), electron_drift_mode='zero')
    input%branch = branches(b)
    input%plasma%electron_drift_mps = 0.0_dp
    input%plasma%ion_drift_mps = solar%solar_wind_speed_mps*sin(elevations(b)*acos(-1.0_dp)/180.0_dp)
    input%plasma%photoelectrons = maxwellian_photoelectrons( &
        solar%photoelectron_reference_density_m3*sin(elevations(b)*acos(-1.0_dp)/180.0_dp), &
        solar%photoelectron_temperature_ev)
    do hot = 0, 1
      solar%ion_temperature_ev = real(hot, dp)
      if (b == 1) solar%ion_temperature_ev = 12.0_dp*real(hot, dp)
      solar%ion_pressure_factor = 3.0_dp
      input%plasma%ion_temperature_ev = solar%ion_temperature_ev
      input%plasma%ion_pressure_factor = solar%ion_pressure_factor
      call solve_equilibrium(solar, reference, status, message)
      if (status /= SHEATH_OK) error stop 'solar reference'
      call solve_equilibrium(input, root, status, message)
      if (status /= SHEATH_OK) then
        print *, branches(b), hot, trim(message)
        error stop 'fixed-entry equilibrium'
      end if
      if (abs(root%surface_potential_v - reference%surface_potential_v) > 2e-6_dp .or. &
          abs(root%minimum_potential_v - reference%minimum_potential_v) > 2e-6_dp) error stop 'physical input agreement'
      if (abs(root%ambient_electron_density_m3/reference%ambient_electron_density_m3 - 1.0_dp) > &
          2e-7_dp) error stop 'electron normalization agreement'
      expected = input%plasma%ion_density_m3*input%plasma%ion_drift_mps
      if (abs(root%ion_inward_flux_m2_s/expected - 1.0_dp) > 1e-14_dp) error stop 'fixed flux'
      if (abs(root%net_current_a_m2) > 1e-12_dp) error stop 'fixed current balance'
      if (b == 1) then
        call evaluate_density(input, root, root%surface_potential_v, density, status, message, side='lower')
      else
        call evaluate_density(input, root, root%surface_potential_v, density, status, message)
      end if
      if (status /= SHEATH_OK .or. density%ion_m3 <= 0.0_dp) error stop 'fixed density'
      call solve_profile(input, options, profile, status, message)
      if (status /= SHEATH_OK) error stop 'fixed profile'
      if (size(profile%z_m) < 2) error stop 'profile nodes'
      if (b == 1) then
        call evaluate_sheath_state(input%plasma, input%branch, root%surface_potential_v, state, status, message, &
            minimum_potential_v=root%minimum_potential_v)
      else
        call evaluate_sheath_state(input%plasma, input%branch, root%surface_potential_v, state, status, message)
      end if
      if (status /= SHEATH_OK .or. .not. state%admissible) error stop 'fixed static state'
      if (b == 1 .and. hot == 1) then
        field%zhao_plasma_input = input%plasma
        field%branch = input%branch
        field%electric_field_v_m = state%electric_field_v_m
        call solve_prescribed_field_candidates(field, candidates, status, message)
        if (status /= SHEATH_OK) error stop 'fixed field search'
        recovered = .false.
        do i = 1, size(candidates)
          if (abs(candidates(i)%boundary_potential_v - root%surface_potential_v) < 1e-5_dp .and. &
              abs(candidates(i)%minimum_potential_v - root%minimum_potential_v) < 1e-5_dp) recovered = .true.
        end do
        if (.not. recovered) error stop 'fixed field recovery'
      end if
    end do
  end do

  input%branch = 'auto'
  input%plasma%photoelectrons = maxwellian_photoelectrons(0.0_dp, 2.2_dp)
  input%plasma%ion_drift_mps = 400e3_dp
  call solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%branch /= 'C') error stop 'zero source'
  input%plasma%photoelectrons = binned_photoelectrons([0.0_dp, 1.0_dp, 3.0_dp], [0.0_dp, 0.0_dp])
  call solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. abs(root%surface_potential_v) <= 0.0_dp) error stop 'binned zero source'
  input%plasma%photoelectrons = binned_photoelectrons( &
      [0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], [2e10_dp, 5e9_dp, 1e9_dp])
  call solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%branch /= 'C') error stop 'binned emitting source'
  if (abs(root%photoelectron_escape_flux_m2_s/2.6e10_dp - 1.0_dp) > 1e-14_dp) error stop 'spectral flux preserved'
  if (abs(root%net_current_a_m2) > 1e-12_dp) error stop 'spectral equilibrium current'

  input%branch = 'B'
  input%plasma%photoelectrons = maxwellian_photoelectrons(1e7_dp, 2.2_dp)
  input%plasma%ion_drift_mps = sqrt(40.0_dp*1.602176634e-19_dp/input%plasma%ion_mass_kg)
  input%plasma%ion_temperature_ev = 5.0_dp
  input%plasma%ion_pressure_factor = 1.0_dp
  energy = 20.0_dp
  pressure = 5.0_dp
  critical = ion_critical_potential(energy, pressure)
  phi = 0.5_dp*(critical + energy)
  call evaluate_sheath_state(input%plasma, 'B', phi, state, status, message)
  if (status /= SHEATH_INVALID_ARGUMENT .or. state%evaluated) error stop 'warm barrier classification'
  input%plasma%ion_drift_mps = 1.0_dp
  call solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_INVALID_ARGUMENT .or. root%valid) error stop 'no speed correction'
  input%plasma%ion_drift_mps = 400e3_dp
  input%plasma%ion_temperature_ev = ieee_value(0.0_dp, ieee_quiet_nan)
  call solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_INVALID_ARGUMENT .or. root%valid) error stop 'nonfinite temperature'
end program test_fixed_entry
