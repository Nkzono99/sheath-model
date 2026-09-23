program test_field_search
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
  use sheath_model
  implicit none
  type(zhao_field_input) :: base, input
  type(zhao_field_search_diagnostics) :: diagnostics, reference_diagnostics
  type(zhao_field_result) :: output, difficult_seed(1)
  type(zhao_field_result), allocatable :: reference(:), scaled(:), previous(:), cold(:), warm(:)
  real(dp), parameter :: density_scales(*) = [1e-8_dp, 1.0_dp, 1e8_dp]
  real(dp), parameter :: temperature_scales(*) = [0.1_dp, 1.0_dp, 10.0_dp]
  integer, parameter :: small_field_ratio_indices(*) = [52, 57]
  real(dp), parameter :: small_fields(*) = [0.03125_dp, 0.078125_dp]
  integer(i32) :: status
  integer :: i, j, step
  real(dp) :: density_scale, temperature_scale
  character(len=512) :: message

  base%electron_drift_mps = 0.0_dp
  base%photoelectrons = maxwellian_photoelectrons(5.5425625842204072e7_dp, 2.2_dp)
  base%electric_field_v_m = 1.6_dp
  call solve_prescribed_field_candidates(base, reference, status, message, reference_diagnostics)
  call check(status == SHEATH_OK, 'reference candidates')
  call check(size(reference) >= 2, 'reference includes multiple branches')
  call check(sum(reference_diagnostics%roots_found) == size(reference), 'deduplicated root counts')

  ! Similarity transformation leaves every dimensionless equation unchanged:
  ! n -> a*n, T -> b*T, v -> sqrt(b)*v, E -> sqrt(a*b)*E.
  ! This detects SI-specific guesses rather than duplicating their construction.
  do i = 1, size(density_scales)
    do j = 1, size(temperature_scales)
      density_scale = density_scales(i)
      temperature_scale = temperature_scales(j)
      input = base
      input%ion_density_m3 = base%ion_density_m3*density_scale
      input%photoelectrons = maxwellian_photoelectrons(5.5425625842204072e7_dp*density_scale, 2.2_dp*temperature_scale)
      input%electron_temperature_ev = base%electron_temperature_ev*temperature_scale
      input%ion_drift_mps = base%ion_drift_mps*sqrt(temperature_scale)
      input%electric_field_v_m = base%electric_field_v_m*sqrt(density_scale*temperature_scale)
      call solve_prescribed_field_candidates(input, scaled, status, message, diagnostics)
      call check(status == SHEATH_OK, 'scaled candidates')
      call compare_roots(reference, scaled, temperature_scale, density_scale)
      call check(all(diagnostics%starts == reference_diagnostics%starts), 'scale-invariant start count')
    end do
  end do

  ! Follow nearby fields while retaining independent starts and every cold root.
  ! Never alias the intent(out) candidate array with the seed input.
  previous = reference
  do step = 1, 4
    input = base
    input%electric_field_v_m = 1.6_dp + 0.005_dp*step
    call solve_prescribed_field_candidates(input, cold, status, message)
    call check(status == SHEATH_OK, 'independent sweep')
    call solve_prescribed_field_candidates(input, warm, status, message, diagnostics, previous)
    call check(status == SHEATH_OK, 'seeded sweep')
    call compare_roots(cold, warm, 1.0_dp, 1.0_dp)
    call check(sum(diagnostics%starts) > sum(reference_diagnostics%starts), 'nearby roots add starts')
    call check(any(diagnostics%unconverged > 0), 'partial search remains visible with accepted roots')
    call check(index(message, 'unresolved') > 0, 'success preserves search warning')
    call move_alloc(warm, previous)
  end do

  ! Shallow A minima found by the earlier search can be missed by independent
  ! starts. A solution at the preceding field recovers these physical branches.
  do i = 1, size(small_fields)
    input = zhao_field_input(electron_drift_mps=0.0_dp)
    input%ion_drift_mps = 468e3_dp*sin(20.0_dp*acos(-1.0_dp)/180.0_dp)
    input%photoelectrons = maxwellian_photoelectrons(input%ion_density_m3*sin(20.0_dp*acos(-1.0_dp)/180.0_dp)* &
        0.5_dp*32.0_dp**(real(small_field_ratio_indices(i), dp)/128.0_dp), 2.2_dp)
    input%electric_field_v_m = small_fields(i) - 0.015625_dp
    call solve_prescribed_field_candidates(input, previous, status, message)
    call check(status == SHEATH_OK, 'nearby small-field reference')
    input%electric_field_v_m = small_fields(i)
    call solve_prescribed_field_candidates(input, warm, status, message, diagnostics, previous)
    call check(status == SHEATH_OK, 'small-field continuation')
    call check(any(warm%branch == 'A' .and. warm%minimum_potential_v < -1e-3_dp), 'shallow A branch recovered')
  end do

  ! First establish a fully rejected search, then add a difficult start.
  ! A rejected root must not hide unresolved attempts in the same search.
  input = zhao_field_input(branch='A', electron_drift_mps=0.0_dp)
  input%ion_drift_mps = 468e3_dp*sin(20.0_dp*acos(-1.0_dp)/180.0_dp)
  input%photoelectrons = maxwellian_photoelectrons( &
      9.82784429_dp*input%ion_density_m3*sin(20.0_dp*acos(-1.0_dp)/180.0_dp), 2.2_dp)
  input%electric_field_v_m = 1.09375_dp
  input%branch = 'B'
  call solve_prescribed_field_candidates(input, scaled, status, message, diagnostics)
  call check(status == SHEATH_NO_PHYSICAL_SOLUTION, 'fully resolved physical rejection stays distinct')
  call check(diagnostics%rejected(2) > 0 .and. diagnostics%unconverged(2) == 0, 'all-rejected search evidence')
  call check(.not. allocated(scaled), 'rejected candidates are not returned')

  difficult_seed(1)%valid = .true.
  difficult_seed(1)%branch = 'B'
  difficult_seed(1)%boundary_potential_v = 1e6_dp
  difficult_seed(1)%ambient_electron_density_m3 = input%ion_density_m3
  call solve_prescribed_field_candidates(input, scaled, status, message, diagnostics, difficult_seed)
  call check(status == SHEATH_NUMERICAL_FAILURE, 'mixed rejection and nonconvergence is unresolved')
  call check(diagnostics%rejected(2) > 0 .and. diagnostics%unconverged(2) > 0, 'mixed search evidence retained')
  call check(.not. allocated(scaled), 'no accepted candidates in mixed search')

  ! Deterministic analytical exclusions need no numerical starts.
  input = base
  input%branch = 'B'
  input%electric_field_v_m = -0.1_dp
  call solve_prescribed_field(input, output, status, message, diagnostics)
  call check(status == SHEATH_NO_PHYSICAL_SOLUTION, 'incompatible field excluded')
  call check(diagnostics%excluded(2) .and. sum(diagnostics%starts) == 0, 'exclusion diagnostic')
  call check(.not. output%valid, 'failed output reset')

  input%electric_field_v_m = ieee_value(0.0_dp, ieee_quiet_nan)
  call solve_prescribed_field_candidates(input, scaled, status, message, diagnostics)
  call check(status == SHEATH_INVALID_ARGUMENT .and. .not. allocated(scaled), 'invalid input reset')
  call check(.not. any(diagnostics%searched) .and. sum(diagnostics%roots_found) == 0, 'diagnostics reset')
  print *, 'Field search diagnostics, similarity and continuation checks passed.'
contains
  subroutine compare_roots(first, second, voltage_scale, density_scale)
    type(zhao_field_result), intent(in) :: first(:), second(:)
    real(dp), intent(in) :: voltage_scale, density_scale
    integer :: k, l
    logical :: found
    call check(size(second) >= size(first), 'no reference root lost')
    do k = 1, size(first)
      found = .false.
      do l = 1, size(second)
        if (second(l)%branch /= first(k)%branch) cycle
        if (abs(second(l)%boundary_potential_v/voltage_scale - first(k)%boundary_potential_v) > 1e-5_dp) cycle
        if (abs(second(l)%minimum_potential_v/voltage_scale - first(k)%minimum_potential_v) > 1e-5_dp) cycle
        if (abs(second(l)%ambient_electron_density_m3/(density_scale*first(k)%ambient_electron_density_m3) - &
            1.0_dp) > 1e-5_dp) cycle
        call check(second(l)%valid .and. ieee_is_finite(second(l)%net_current_a_m2), 'valid finite candidate')
        found = .true.
      end do
      call check(found, 'same physical root recovered')
    end do
  end subroutine compare_roots
  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, 'FAIL: ', label, '; ', trim(message)
      error stop 1
    end if
  end subroutine check
end program test_field_search
