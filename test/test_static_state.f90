program test_static_state
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sheath_model
  use sheath_model_constants, only: qe, pi, electron_mass
  implicit none
  type(zhao_plasma_input) :: plasma
  type(zhao_state_result) :: state, trial, left, right
  type(zhao_equilibrium_input) :: equilibrium_input
  type(zhao_equilibrium_result) :: equilibrium
  type(zhao_field_input) :: field_input
  type(zhao_field_result) :: seed(1)
  type(zhao_field_result), allocatable :: candidates(:)
  integer(i32) :: status
  integer :: i
  real(dp) :: density, gamma, scale, phi, edges(257), flux(256)
  character(len=256) :: message

  ! The equilibrium, field closure and potential evaluator must use the same physical state.
  equilibrium_input = zhao_equilibrium_input(electron_drift_mode='zero', branch='A')
  call solve_equilibrium(equilibrium_input, equilibrium, status, message)
  call ok('equilibrium reference')
  plasma%electron_drift_mps = 0.0_dp
  plasma%photoelectrons = maxwellian_photoelectrons(64e6_dp*sin(pi/3.0_dp), 2.2_dp)
  call evaluate_sheath_state(plasma, 'A', equilibrium%surface_potential_v, state, status, message, &
      minimum_potential_v=equilibrium%minimum_potential_v)
  call ok('static A')
  call check(state%admissible, 'complete A is admissible')
  call near(state%ambient_electron_density_m3, equilibrium%ambient_electron_density_m3, 1e-8_dp, 'neutral density')
  call check(abs(state%connection_residual_v2_m2) < 1e-8_dp, 'A connection')
  call check(abs(state%net_current_a_m2) < 1e-12_dp, 'J=0 via common fluxes')
  call evaluate_sheath_state(plasma, 'A', equilibrium%surface_potential_v, trial, status, message, &
      minimum_potential_v=1.1_dp*equilibrium%minimum_potential_v)
  call ok('off-shell A evaluation')
  call check(trial%evaluated .and. .not. trial%admissible, 'finite evaluation is not a completed A root')
  call check(abs(trial%connection_residual_v2_m2) > 1e-8_dp, 'off-shell connection retained')
  call evaluate_sheath_state(plasma, 'A', 1.0_dp, trial, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'A requires its minimum')

  ! B/C state -> prescribed field -> same state, for both analytic and non-Maxwellian sources.
  do i = 1, 4
    plasma = zhao_plasma_input(electron_drift_mps=0.0_dp)
    phi = -1.0_dp
    if (i /= 1) then
      phi = 0.5_dp
      plasma%photoelectrons = binned_photoelectrons([0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], [2e12_dp, 5e11_dp, 1e11_dp])
    end if
    if (i == 3) plasma%photoelectrons = maxwellian_photoelectrons(2e7_dp, 2.2_dp)
    if (i == 4) then
      phi = -1.0_dp
      plasma%photoelectrons = binned_photoelectrons([0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], [2e10_dp, 5e9_dp, 1e9_dp])
    end if
    call evaluate_sheath_state(plasma, merge('C', 'B', phi < 0.0_dp), phi, state, status, message)
    call ok('static B/C')
    call check(state%admissible, 'chosen B/C profile is admissible')
    if (i == 1) call check(state%electric_field_v_m < 0.0_dp, 'C field sign')
    field_input%zhao_plasma_input = plasma
    field_input%branch = state%branch
    field_input%electric_field_v_m = state%electric_field_v_m
    seed(1) = zhao_field_result()
    seed(1)%valid = .true.
    seed(1)%branch = state%branch
    seed(1)%boundary_potential_v = phi
    seed(1)%minimum_potential_v = min(phi, 0.0_dp)
    seed(1)%ambient_electron_density_m3 = state%ambient_electron_density_m3
    call solve_prescribed_field_candidates(field_input, candidates, status, message)
    call ok('B/C field inversion')
    call check(any(abs(candidates%boundary_potential_v - phi) < 1e-6_dp), 'static candidate recovered')
    call near(state%photoelectron_outward_flux_m2_s, &
        state%photoelectron_escape_flux_m2_s + state%photoelectron_return_flux_m2_s, 1e-14_dp, 'state flux conservation')
  end do

  plasma%photoelectrons = maxwellian_photoelectrons(0.0_dp, 2.2_dp)
  call evaluate_sheath_state(plasma, 'B', 0.5_dp, state, status, message)
  call ok('finite nonphysical B')
  call check(.not. state%admissible .and. state%physical_status == SHEATH_NO_PHYSICAL_SOLUTION, 'B asymptotic exclusion')
  call check(state%boundary_field_squared_v2_m2 < 0.0_dp, 'signed E squared retained')
  call check(.not. ieee_is_finite(state%electric_field_v_m), 'imaginary field not clipped to zero')
  plasma%photoelectrons = maxwellian_photoelectrons(64e6_dp*sin(pi/3.0_dp), 2.2_dp)
  call evaluate_sheath_state(plasma, 'B', 10.0_dp, state, status, message)
  call ok('positive boundary E squared does not establish a physical B profile')
  call check(state%boundary_field_squared_v2_m2 > 0.0_dp .and. .not. state%admissible, 'B upstream guard')
  plasma%photoelectrons = maxwellian_photoelectrons(0.0_dp, 2.2_dp)
  call evaluate_sheath_state(plasma, 'B', 0.0_dp, state, status, message)
  call ok('flat zero emission state')
  call check(state%admissible, 'flat state')

  ! Rounding-cell correction at a spectral edge affects ALL electron observables consistently.
  plasma%photoelectrons = binned_photoelectrons([0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], [2e12_dp, 5e11_dp, 1e11_dp])
  phi = 1.0_dp
  call evaluate_sheath_state(plasma, 'B', nearest(phi, -1.0_dp), left, status, message)
  call ok('left neighbor')
  call evaluate_sheath_state(plasma, 'B', nearest(phi, 1.0_dp), right, status, message)
  call ok('right neighbor')
  call evaluate_sheath_state(plasma, 'B', phi, state, status, message)
  call ok('edge state')
  density = 0.5_dp*left%ambient_electron_density_m3 + 0.5_dp*right%ambient_electron_density_m3
  call evaluate_sheath_state(plasma, 'B', phi, trial, status, message, electron_normalization_m3=density)
  call ok('rounding-cell state')
  call near(trial%ambient_electron_density_m3, density, 1e-15_dp, 'corrected normalization')
  call near(trial%electron_inward_flux_m2_s/state%electron_inward_flux_m2_s, &
      density/state%ambient_electron_density_m3, 1e-14_dp, 'consistent corrected flux')
  call evaluate_sheath_state(plasma, 'B', phi, trial, status, message, electron_normalization_m3=2*density)
  call check(status == SHEATH_INVALID_ARGUMENT, 'density cannot escape adjacent-potential cell')

  call evaluate_sheath_state(plasma, 'A', 0.5_dp, trial, status, message, minimum_potential_v=-1e-12_dp)
  call ok('shallow spectral A trial')
  call check(.not. trial%admissible, 'small absolute connection residual alone cannot certify shallow A')

  ! A decimal bin edge must survive internal normalization without changing the one-sided upstream limit.
  do i = 1, 1000
    phi = real(i, dp)/100.0_dp
    if ((phi/12.0_dp)*12.0_dp > phi) exit
  end do
  call check(i <= 1000, 'decimal bin-edge regression setup')
  plasma%photoelectrons = binned_photoelectrons([0.0_dp, phi], [1e13_dp])
  call evaluate_sheath_state(plasma, 'B', phi, state, status, message)
  call ok('exact decimal bin edge')
  call check(state%admissible, 'internal scaling must not move the boundary outside spectral support')

  plasma%photoelectrons = binned_photoelectrons([0.0_dp, 1.0_dp], [0.0_dp])
  call evaluate_sheath_state(plasma, 'C', -1.0_dp, state, status, message)
  call ok('zero spectral emission')
  call check(state%admissible .and. state%photoelectron_outward_flux_m2_s == 0.0_dp, 'zero spectrum C')

  ! Refined Maxwellian spectrum also closes the nonmonotonic upper segment independently of the source temperature.
  scale = 2.2_dp
  gamma = 64e6_dp*sin(pi/3)*sqrt(2*qe*scale/electron_mass)/(2*sqrt(pi))
  do i = 1, 257
    edges(i) = (i - 1)*44.0_dp/256
  end do
  flux = gamma*(exp(-edges(:256)/scale) - exp(-edges(2:)/scale))
  plasma%photoelectrons = binned_photoelectrons(edges, flux)
  field_input%zhao_plasma_input = plasma
  field_input%branch = 'A'
  call evaluate_sheath_state(plasma, 'A', equilibrium%surface_potential_v, state, status, message, &
      minimum_potential_v=equilibrium%minimum_potential_v)
  call ok('spectral A trial')
  field_input%electric_field_v_m = state%electric_field_v_m
  seed(1)%branch = 'A'
  seed(1)%boundary_potential_v = state%boundary_potential_v
  seed(1)%minimum_potential_v = state%minimum_potential_v
  seed(1)%ambient_electron_density_m3 = state%ambient_electron_density_m3
  call solve_prescribed_field_candidates(field_input, candidates, status, message, initial_guesses=seed)
  call ok('spectral A solution')
  call evaluate_sheath_state(plasma, 'A', candidates(1)%boundary_potential_v, state, status, message, &
      minimum_potential_v=candidates(1)%minimum_potential_v)
  call ok('spectral A re-evaluation')
  call check(state%admissible .and. abs(state%connection_residual_v2_m2) < 1e-7_dp, 'spectral A upper connection')
  print *, 'Static sheath states, closure consistency and rounding-cell checks passed.'
contains
  subroutine ok(label)
    character(len=*), intent(in) :: label
    call check(status == SHEATH_OK, label)
  end subroutine
  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, 'FAIL: ', label, ' ', trim(message), ' ', trim(state%physical_message)
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
