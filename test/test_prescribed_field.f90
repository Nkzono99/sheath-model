program test_prescribed_field
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan, ieee_is_finite
  use sheath_model
  implicit none
  real(dp), parameter :: qe = 1.602176634e-19_dp, eps0 = 8.8541878128e-12_dp
  type(zhao_field_input) :: input
  type(zhao_field_result) :: output
  type(zhao_field_result), allocatable :: candidates(:)
  real(dp) :: thermal, expected
  integer(i32) :: status
  character(len=512) :: message

  call solve_prescribed_field(input, output, status, message)
  call ok('zero field without photoelectrons')
  thermal = sqrt(2.0_dp*qe*input%electron_temperature_ev/input%electron_mass_kg)
  expected = 2.0_dp*input%ion_density_m3/(1.0_dp + erf(input%electron_drift_mps/thermal))
  call near(output%ambient_electron_density_m3, expected, expected*1e-12_dp, 'upstream neutrality')
  call near(output%ion_inward_flux_m2_s, input%ion_density_m3*input%ion_drift_mps, 1.0_dp, 'cold ion flux')
  call check(output%branch == 'B', 'degenerate B')
  call near(output%boundary_potential_v, 0.0_dp, 0.0_dp, 'zero gauge')
  call near(output%minimum_potential_v, 0.0_dp, 0.0_dp, 'B minimum')

  input%branch = 'A'
  input%electron_drift_mps = 0.0_dp
  input%electric_field_v_m = 1.4187346568707933e-11_dp/eps0
  input%photoelectrons = maxwellian_photoelectrons(5.5425625842204072e7_dp, 2.2_dp)
  call solve_prescribed_field(input, output, status, message)
  call ok('Type A reference')

  call solve_prescribed_field_candidates(input, candidates, status, message)
  call check(status == SHEATH_OK .and. allocated(candidates), 'candidate enumeration')
  call check(size(candidates) == 1 .and. all(candidates%valid), 'one admissible Type A candidate')
  call near(candidates(1)%boundary_potential_v, output%boundary_potential_v, 1e-10_dp, 'unique candidate potential')

  input%branch = 'auto'
  call solve_prescribed_field(input, output, status, message)
  call check(status == SHEATH_AMBIGUOUS_SOLUTION, 'ambiguous roots must be explicit')
  call check(.not. output%valid .and. output%boundary_potential_v == 0.0_dp, 'failed result reset')
  call solve_prescribed_field_candidates(input, candidates, status, message)
  call check(status == SHEATH_OK .and. allocated(candidates), 'ambiguous candidates remain inspectable')
  call check(size(candidates) == 2 .and. all(candidates%valid), 'two admissible candidates')
  call check(any(candidates%branch == 'A') .and. any(candidates%branch == 'B'), 'A and B candidates')

  input = zhao_field_input(electric_field_v_m=-0.02_dp, electron_drift_mps=0.0_dp)
  call solve_prescribed_field(input, output, status, message)
  call ok('negative field')
  call check(output%branch == 'C' .and. output%boundary_potential_v < 0.0_dp, 'negative field C')
  call near(output%minimum_potential_v, output%boundary_potential_v, 0.0_dp, 'C minimum')
  call near(output%photoelectron_escape_flux_m2_s, 0.0_dp, 0.0_dp, 'no emission')
  call near(output%net_current_a_m2, qe*(output%electron_inward_flux_m2_s - output%ion_inward_flux_m2_s), &
      1e-18_dp, 'C net current')
  input%branch = 'B'
  call solve_prescribed_field(input, output, status, message)
  call check(status == SHEATH_NO_PHYSICAL_SOLUTION, 'explicit branch does not fall back')

  input = zhao_field_input(photoelectrons=maxwellian_photoelectrons(-1.0_dp, 2.2_dp))
  call solve_prescribed_field(input, output, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'negative source density')
  input = zhao_field_input(electric_field_v_m=ieee_value(0.0_dp, ieee_quiet_nan))
  call solve_prescribed_field(input, output, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'NaN field')
  input = zhao_field_input(ion_mass_kg=-1.0_dp)
  call solve_prescribed_field(input, output, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'negative mass')
  input = zhao_field_input(photoelectrons=maxwellian_photoelectrons(0.0_dp, 0.0_dp))
  call solve_prescribed_field(input, output, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'zero temperature')
  input = zhao_field_input(branch='invalid')
  call solve_prescribed_field(input, output, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'unknown branch')
  ! Calls are stateless; invalid input cannot poison a subsequent solution.
  input = zhao_field_input()
  call solve_prescribed_field(input, output, status, message)
  call ok('valid solve after invalid input')
  print *, 'Prescribed-field model checks passed.'
contains
  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, 'FAIL: ', label, '; ', trim(message)
      error stop 1
    end if
  end subroutine check
  subroutine ok(label)
    character(len=*), intent(in) :: label
    call check(status == SHEATH_OK .and. output%valid, label)
  end subroutine ok
  subroutine near(actual, wanted, tolerance, label)
    real(dp), intent(in) :: actual, wanted, tolerance
    character(len=*), intent(in) :: label
    if (.not. ieee_is_finite(actual) .or. abs(actual - wanted) > tolerance) then
      print *, 'actual, expected, tolerance:', actual, wanted, tolerance
      call check(.false., label)
    end if
  end subroutine near
end program test_prescribed_field
