! SPDX-License-Identifier: MIT
program test_search
  use sheath_model
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  implicit none
  type(sheath_solver) :: solver
  type(zhao_equilibrium_input) :: solar
  type(fixed_entry_equilibrium_input) :: fixed
  type(sheath_equilibrium_result) :: baseline, result, seed(1)
  type(prescribed_field_input) :: field
  type(prescribed_field_result), allocatable :: candidates(:)
  type(sheath_search_diagnostics) :: diagnostics
  integer(i32) :: status
  integer :: b, m, i
  real(dp) :: factor
  character(len=7), parameter :: methods(4) = ['auto   ', 'newton ', 'lm     ', 'bracket']
  real(dp), parameter :: angles(3) = [60.0_dp, 20.0_dp, 10.0_dp]
  character(len=1), parameter :: branches(3) = ['A', 'B', 'C']
  character(len=512) :: message

  solar%electron_drift_mode = 'zero'
  solar%ion_temperature_ev = 1.0_dp
  do b = 1, 3
    solar%sun_elevation_deg = angles(b)
    solar%branch = branches(b)
    solver%search%method = 'auto'
    call solver%solve_equilibrium(solar, baseline, status, message)
    if (status /= SHEATH_OK) error stop 'reference root'
    do m = 2, 4
      if (b == 1 .and. m == 4) cycle
      solver%search%method = methods(m)
      call solver%solve_equilibrium(solar, result, status, message, diagnostics)
      if (status /= SHEATH_OK) then
        print *, branches(b), methods(m), trim(message), diagnostics%best_residual
        error stop 'selected equilibrium method'
      end if
      if (abs(result%surface_potential_v - baseline%surface_potential_v) > 2e-7_dp) error stop 'method agreement'
      if (diagnostics%evaluations(b) == 0 .or. diagnostics%best_residual(b) > 1e-10_dp) error stop 'diagnostics'
      if (m == 3 .and. diagnostics%lm_steps(b) == 0) error stop 'LM exercised'
    end do
  end do

  solver = sheath_solver()
  fixed%branch = 'A'
  fixed%plasma%photoelectrons = maxwellian_photoelectrons(55.42562584220407e6_dp, 2.2_dp)
  fixed%plasma%ion_temperature_ev = 12.0_dp
  fixed%plasma%ion_pressure_factor = 3.0_dp
  call solver%solve_equilibrium(fixed, baseline, status, message)
  if (status /= SHEATH_OK) error stop 'fixed baseline'
  seed(1) = baseline
  fixed%plasma%photoelectrons = maxwellian_photoelectrons(1.01_dp*55.42562584220407e6_dp, 2.2_dp)
  solver%search%method = 'newton'
  solver%search%use_default_guesses = .false.
  call solver%solve_equilibrium(fixed, result, status, message, diagnostics, seed)
  if (status /= SHEATH_OK .or. diagnostics%starts(1) /= 1) error stop 'caller continuation'
  solver%search%use_default_guesses = .true.
  call solver%solve_equilibrium(fixed, baseline, status, message)
  if (status /= SHEATH_OK .or. abs(result%surface_potential_v - baseline%surface_potential_v) > 2e-8_dp) &
      error stop 'independent continuation check'
  do i = 1, 2
    factor = 10.0_dp**(18*i - 27)
    fixed%plasma%ion_density_m3 = 8.7e6_dp*factor
    fixed%plasma%photoelectrons = maxwellian_photoelectrons(1.01_dp*55.42562584220407e6_dp*factor, 2.2_dp)
    do m = 1, 3
      solver%search%method = methods(m)
      call solver%solve_equilibrium(fixed, result, status, message, diagnostics)
      if (status /= SHEATH_OK .or. abs(result%surface_potential_v - baseline%surface_potential_v) > 2e-7_dp) &
          error stop 'density-scaled search'
    end do
  end do
  fixed%plasma%ion_density_m3 = 8.7e6_dp
  fixed%plasma%photoelectrons = maxwellian_photoelectrons(55.42562584220407e6_dp, 2.2_dp)
  solver%search%method = 'newton'
  solver%search%max_iterations = 0
  call solver%solve_equilibrium(fixed, result, status, message, diagnostics)
  if (status /= SHEATH_NUMERICAL_FAILURE .or. sum(diagnostics%unconverged) == 0) error stop 'iteration budget'
  solver%search%residual_tolerance = ieee_value(0.0_dp, ieee_quiet_nan)
  call solver%solve_equilibrium(fixed, result, status, message)
  if (status /= SHEATH_INVALID_ARGUMENT) error stop 'invalid options'

  ! Prescribed-field search uses the same selected numerical kernel.
  solver = sheath_solver()
  field%plasma_input = plasma_input(electron_drift_mps=0.0_dp)
  field%branch = 'C'
  field%electric_field_v_m = -0.4_dp
  do m = 2, 3
    solver%search%method = methods(m)
    call solver%solve_prescribed_field_candidates(field, candidates, status, message, diagnostics)
    if (status /= SHEATH_OK .or. .not. allocated(candidates)) error stop 'field method'
    if (diagnostics%evaluations(3) == 0) error stop 'field diagnostics'
  end do
  print *, 'Selectable search and scaling checks passed.'
end program test_search
