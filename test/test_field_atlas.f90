! SPDX-License-Identifier: MIT
program test_field_atlas
  use sheath_model
  implicit none
  type(zhao_field_input) :: inputs(2), base, query
  type(zhao_field_result) :: root, reference, invalid
  type(zhao_field_result), allocatable :: independent(:), mapped(:), deflated(:)
  type(sheath_field_atlas) :: atlas, loaded, single
  type(sheath_equilibrium_atlas) :: equilibrium_atlas
  type(sheath_atlas_point) :: point
  type(sheath_search_diagnostics) :: diagnostics
  integer(i32), allocatable :: report(:, :)
  integer(i32) :: status
  integer :: i, unit, io
  real(dp), allocatable :: predictions(:, :)
  character(len=512) :: message

  base%electron_drift_mps = 0.0_dp
  base%photoelectrons = maxwellian_photoelectrons(55.42562584220407e6_dp, 2.2_dp)
  base%electric_field_v_m = 1.6_dp
  do i = 1, 2
    inputs(i) = base
    inputs(i)%electric_field_v_m = 1.56_dp + 0.04_dp*i
  end do
  call build_field_atlas(inputs, atlas, status, message, report)
  call check(status == SHEATH_OK .and. atlas%size() >= 4, 'map includes A and B at both fields')
  call check(all(report(:2, :) == SHEATH_OK), 'stored branches reported')
  call check(all(report(3, :) == SHEATH_NO_PHYSICAL_SOLUTION), 'field sign excluded C')
  call solve_prescribed_field_candidates(base, independent, status, message)
  call check(status == SHEATH_OK .and. size(independent) >= 2, 'independent physical roots')
  query = base
  query%electric_field_v_m = 1.62_dp
  call solve_prescribed_field_candidates(query, independent, status, message)
  call check(status == SHEATH_OK, 'independent query')
  query%search%method = 'newton'
  query%search%use_default_guesses = .false.
  query%search%max_starts = 1
  call solve_prescribed_field_candidates(query, mapped, status, message, diagnostics, atlas=atlas)
  call check(status == SHEATH_OK, 'atlas query')
  call compare_roots(independent, mapped, 1.0_dp, 1.0_dp)
  call check(all(diagnostics%atlas_starts(:2) == 1), 'map starts recorded per branch')
  call check(sum(diagnostics%atlas_hits) >= 2, 'map retains competing physical candidates')
  call solve_prescribed_field(query, root, status, message, atlas=atlas)
  call check(status == SHEATH_AMBIGUOUS_SOLUTION .and. .not. root%valid, 'map does not choose an ambiguous root')

  open (newunit=unit, status='scratch', form='formatted')
  call atlas%write(unit, io)
  call check(io == 0, 'write field map')
  rewind (unit)
  call loaded%read(unit, io)
  call check(io == 0 .and. loaded%size() == atlas%size(), 'read field map')
  rewind (unit)
  call equilibrium_atlas%read(unit, io)
  call check(io /= 0 .and. equilibrium_atlas%size() == 0, 'J=0 rejects field map')
  close (unit)
  open (newunit=unit, status='scratch', form='formatted')
  call equilibrium_atlas%write(unit, io)
  rewind (unit)
  call loaded%read(unit, io)
  call check(io /= 0 .and. loaded%size() == atlas%size(), 'field map rejects J=0 transactionally')
  close (unit)

  query = inputs(1)
  query%branch = 'A'
  call solve_prescribed_field(query, reference, status, message)
  call check(status == SHEATH_OK, 'reference stored field')
  query%ion_density_m3 = query%ion_density_m3*1e8_dp
  query%photoelectrons = maxwellian_photoelectrons(55.42562584220407e14_dp, 22.0_dp)
  query%electron_temperature_ev = query%electron_temperature_ev*10.0_dp
  query%ion_drift_mps = query%ion_drift_mps*sqrt(10.0_dp)
  query%electric_field_v_m = query%electric_field_v_m*sqrt(1e9_dp)
  query%search%method = 'newton'
  query%search%use_default_guesses = .false.
  query%search%max_iterations = 0
  query%search%max_starts = 1
  call solve_prescribed_field(query, root, status, message, diagnostics, atlas=loaded, max_roots=1)
  call check(status == SHEATH_OK .and. diagnostics%atlas_hits(1) == 1, 'map uses dimensionless E and plasma')
  call check(abs(root%boundary_potential_v/10.0_dp - reference%boundary_potential_v) < 2e-8_dp, 'voltage similarity')
  call check(abs(root%ambient_electron_density_m3/(1e8_dp*reference%ambient_electron_density_m3) - 1.0_dp) &
      < 1e-9_dp, 'density similarity')

  query = base
  query%branch = 'A'
  call solve_prescribed_field(query, reference, status, message)
  call check(status == SHEATH_OK, 'continuation starting root')
  call add_field_to_atlas(query, reference, single, status, message)
  call check(status == SHEATH_OK .and. single%size() == 1, 'register field root')
  invalid = reference
  invalid%boundary_potential_v = invalid%boundary_potential_v + 1.0_dp
  call add_field_to_atlas(query, invalid, single, status, message)
  call check(status /= SHEATH_OK .and. single%size() == 1, 'original residual checked on registration')
  query%electric_field_v_m = 1.7_dp
  call add_field_to_atlas(query, reference, single, status, message)
  call check(status /= SHEATH_OK .and. single%size() == 1, 'same plasma at another E is not the stored root')
  call solve_prescribed_field(query, reference, status, message)
  call check(status == SHEATH_OK, 'continuation target reference')
  query%search%method = 'newton'
  query%search%max_iterations = 2
  query%search%max_starts = 1
  call solve_prescribed_field(query, root, status, message)
  call check(status == SHEATH_NUMERICAL_FAILURE, 'small corrector budget fails independently')
  do i = 1, 2
    single%continuation%method = 'parameter'
    if (i == 2) single%continuation%method = 'arclength'
    call solve_prescribed_field(query, root, status, message, diagnostics, atlas=single)
    call check(status == SHEATH_OK, 'adaptive field continuation recovers the root')
    call check(abs(root%boundary_potential_v - reference%boundary_potential_v) < 2e-8_dp, 'target root agrees')
    call check(diagnostics%continuation_steps(1) > 0 .and. diagnostics%continuation_retries(1) > 0, &
        'adaptive steps and retries recorded')
  end do

  query = base
  call solve_prescribed_field_candidates(query, independent, status, message)
  call check(status == SHEATH_OK, 'deflation reference')
  call solve_prescribed_field_candidates(query, deflated, status, message, diagnostics, deflation=.true.)
  call check(status == SHEATH_OK .and. sum(diagnostics%deflations) > 0, 'deflation used for field closure')
  call compare_roots(independent, deflated, 1.0_dp, 1.0_dp)

  ! A failed continuation must remain unresolved even when independent starts
  ! converge only to rejected candidates. It is not an exclusion proof.
  query = base
  query%branch = 'B'
  call solve_prescribed_field(query, reference, status, message)
  call check(status == SHEATH_OK, 'B continuation starting root')
  call single%clear()
  call add_field_to_atlas(query, reference, single, status, message)
  call check(status == SHEATH_OK, 'B starting root stored')
  single%options%max_distance = 10.0_dp
  single%continuation%max_steps = 1
  single%continuation%method = 'parameter'
  query%ion_drift_mps = 468e3_dp*sin(20.0_dp*acos(-1.0_dp)/180.0_dp)
  query%photoelectrons = maxwellian_photoelectrons( &
      9.82784429_dp*query%ion_density_m3*sin(20.0_dp*acos(-1.0_dp)/180.0_dp), 2.2_dp)
  query%electric_field_v_m = 1.09375_dp
  call solve_prescribed_field_candidates(query, mapped, status, message, diagnostics)
  call check(status == SHEATH_NO_PHYSICAL_SOLUTION .and. diagnostics%rejected(2) > 0, 'all-rejected reference')
  call solve_prescribed_field_candidates(query, mapped, status, message, diagnostics, atlas=single)
  call check(status == SHEATH_NUMERICAL_FAILURE .and. diagnostics%unconverged(2) > 0, &
      'failed continuation remains unresolved alongside physical rejections')
  single%continuation = sheath_continuation_options()
  single%options%max_distance = 1.0_dp

  ! A changed spectrum shape must not borrow solutions from another family.
  query = zhao_field_input(branch='C', electric_field_v_m=-0.02_dp, electron_drift_mps=0.0_dp)
  query%photoelectrons = binned_photoelectrons([0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], [2e10_dp, 5e9_dp, 1e9_dp])
  call solve_prescribed_field(query, reference, status, message)
  call check(status == SHEATH_OK, 'spectral C reference')
  call single%clear()
  call add_field_to_atlas(query, reference, single, status, message)
  call check(status == SHEATH_OK, 'spectral field root stored')
  query%electric_field_v_m = -0.0201_dp
  query%search%method = 'newton'
  query%search%use_default_guesses = .false.
  call solve_prescribed_field(query, root, status, message, diagnostics, atlas=single)
  call check(status == SHEATH_OK .and. diagnostics%atlas_hits(3) == 1, 'spectral field query')
  point = single%point(1)
  point%spectrum_shape(2) = point%spectrum_shape(2) + 0.1_dp
  call single%predictions(point%key, point%spectrum_shape, 'C', predictions)
  call check(size(predictions, 2) == 0, 'different source shape excluded from map')

  query = base
  query%branch = 'A'
  query%ion_temperature_ev = 12.0_dp
  query%ion_pressure_factor = 3.0_dp
  call solve_prescribed_field(query, reference, status, message)
  call check(status == SHEATH_OK, 'warm ion field root')
  call single%clear()
  call add_field_to_atlas(query, reference, single, status, message)
  call check(status == SHEATH_OK, 'warm ion map')
  query%electric_field_v_m = 1.61_dp
  query%search%method = 'newton'
  query%search%use_default_guesses = .false.
  call solve_prescribed_field(query, root, status, message, diagnostics, atlas=single)
  call check(status == SHEATH_OK .and. diagnostics%atlas_hits(1) == 1, 'warm ion map query')

  single%continuation%max_steps = 0
  call solve_prescribed_field_candidates(query, mapped, status, message, atlas=single)
  call check(status == SHEATH_INVALID_ARGUMENT .and. .not. allocated(mapped), 'invalid continuation controls')
  call solve_prescribed_field_candidates(query, mapped, status, message, max_roots=0)
  call check(status == SHEATH_INVALID_ARGUMENT, 'invalid candidate bound')
  print *, 'Field atlas, continuation, deflation and physical checks passed.'
contains
  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, 'FAIL: ', label, '; ', trim(message)
      error stop 1
    end if
  end subroutine
  subroutine compare_roots(first, second, voltage_scale, density_scale)
    type(zhao_field_result), intent(in) :: first(:), second(:)
    real(dp), intent(in) :: voltage_scale, density_scale
    integer :: k, j
    logical :: found
    do k = 1, size(first)
      found = .false.
      do j = 1, size(second)
        if (first(k)%branch /= second(j)%branch) cycle
        if (abs(first(k)%boundary_potential_v - second(j)%boundary_potential_v/voltage_scale) > 2e-8_dp) cycle
        if (abs(first(k)%minimum_potential_v - second(j)%minimum_potential_v/voltage_scale) > 2e-8_dp) cycle
        if (abs(first(k)%ambient_electron_density_m3/(second(j)%ambient_electron_density_m3/density_scale) - 1.0_dp) &
            > 1e-9_dp) cycle
        found = .true.
      end do
      call check(found, 'independent physical root retained')
    end do
  end subroutine
end program
