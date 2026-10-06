! SPDX-License-Identifier: MIT
program test_atlas
  use sheath_model
  use sheath_model_continuation, only: continue_guarded_system, find_guarded_roots
  implicit none
  type(sheath_solver) :: solver
  type(zhao_equilibrium_input) :: inputs(2), query
  type(fixed_entry_equilibrium_input) :: fixed
  type(fixed_entry_equilibrium_input) :: fixed_sweep(1)
  type(sheath_equilibrium_atlas) :: atlas, loaded
  type(sheath_equilibrium_result) :: result, reference, rejected_seed
  type(sheath_equilibrium_result), allocatable :: candidates(:)
  type(sheath_search_options) :: search
  type(sheath_continuation_options) :: continuation
  type(sheath_search_diagnostics) :: diagnostics
  integer(i32), allocatable :: report(:, :)
  real(dp), allocatable :: roots(:, :)
  real(dp) :: y(1), starts(1, 3)
  real(dp), allocatable :: predictions(:, :)
  type(sheath_atlas_point) :: point
  integer(i32) :: status
  integer :: b, i, unit, io
  logical :: success
  character(len=512) :: message
  real(dp), parameter :: angles(3) = [60.0_dp, 20.0_dp, 10.0_dp]
  character(len=1), parameter :: branches(3) = ['A', 'B', 'C']

  do b = 1, 3
    solver = sheath_solver()
    call atlas%clear()
    do i = 1, 2
      inputs(i)%sun_elevation_deg = angles(b)
      inputs(i)%electron_drift_mode = 'zero'
      inputs(i)%ion_temperature_ev = 1.0_dp
      inputs(i)%branch = branches(b)
      inputs(i)%photoelectron_reference_density_m3 = real(61 + 2*i, dp)*1e6_dp
    end do
    call solver%build_equilibrium_atlas(inputs, atlas, status, message, report)
    if (status /= SHEATH_OK .or. atlas%size() /= 2) then
      print *, trim(message), report
      error stop 'atlas sweep'
    end if
    query = inputs(1)
    query%photoelectron_reference_density_m3 = 64e6_dp
    call solver%solve_equilibrium(query, reference, status, message)
    if (status /= SHEATH_OK) error stop 'independent root'
    solver%search%method = 'newton'
    solver%search%use_default_guesses = .false.
    solver%equilibrium_atlas = atlas
    call solver%solve_equilibrium(query, result, status, message, diagnostics)
    deallocate (solver%equilibrium_atlas)
    if (status /= SHEATH_OK .or. abs(result%surface_potential_v - reference%surface_potential_v) > 2e-8_dp) &
        error stop 'map correction'
    if (diagnostics%atlas_hits(b) /= 1 .or. diagnostics%starts(b) /= 1) error stop 'map used'
    open (newunit=unit, status='scratch', form='formatted')
    call atlas%write(unit, io)
    if (io /= 0) error stop 'atlas write'
    rewind (unit)
    call loaded%read(unit, io)
    close (unit)
    if (io /= 0 .or. loaded%size() /= atlas%size()) error stop 'atlas read'
    solver%equilibrium_atlas = loaded
    call solver%solve_equilibrium(query, result, status, message, diagnostics)
    deallocate (solver%equilibrium_atlas)
    if (status /= SHEATH_OK .or. abs(result%surface_potential_v - reference%surface_potential_v) > 2e-8_dp) &
        error stop 'loaded map correction'
    solver%search%max_starts = 1
    rejected_seed = reference
    rejected_seed%ambient_electron_density_m3 = -1.0_dp
    solver%equilibrium_atlas = loaded
    call solver%solve_equilibrium_candidates(query, candidates, status, message, diagnostics, &
        initial_guesses=[rejected_seed])
    deallocate (solver%equilibrium_atlas)
    if (status /= SHEATH_OK .or. size(candidates) /= 1) error stop 'map candidates'
    if (diagnostics%atlas_starts(b) /= 1 .or. diagnostics%atlas_hits(b) /= 1) error stop 'candidate map diagnostics'
  end do

  solver = sheath_solver()
  fixed%branch = 'A'
  fixed%plasma%photoelectrons = maxwellian_photoelectrons(55.42562584220407e6_dp, 2.2_dp)
  fixed%plasma%ion_temperature_ev = 12.0_dp
  fixed%plasma%ion_pressure_factor = 3.0_dp
  call atlas%clear()
  call solver%solve_equilibrium(fixed, reference, status, message)
  if (status /= SHEATH_OK) error stop 'fixed reference'
  call solver%add_equilibrium_to_atlas(fixed, reference, atlas, status, message)
  if (status /= SHEATH_OK) error stop 'fixed atlas add'
  fixed%plasma%photoelectrons = maxwellian_photoelectrons(120e6_dp, 2.2_dp)
  call solver%solve_equilibrium(fixed, reference, status, message)
  if (status /= SHEATH_OK) error stop 'difficult independent reference'
  solver%search%method = 'newton'
  solver%search%max_starts = 1
  solver%search%max_iterations = 2
  call solver%solve_equilibrium(fixed, result, status, message)
  if (status /= SHEATH_NUMERICAL_FAILURE) error stop 'small budget fails independently'
  do i = 1, 2
    solver%continuation%method = 'parameter'
    if (i == 2) solver%continuation%method = 'arclength'
    solver%equilibrium_atlas = atlas
    call solver%solve_equilibrium(fixed, result, status, message, diagnostics)
    deallocate (solver%equilibrium_atlas)
    if (status /= SHEATH_OK .or. abs(result%surface_potential_v - reference%surface_potential_v) > 2e-8_dp) then
      print *, trim(message), diagnostics%continuation_steps, diagnostics%continuation_retries
      error stop 'adaptive sheath continuation'
    end if
    if (diagnostics%continuation_steps(1) == 0 .or. diagnostics%continuation_retries(1) == 0) &
        error stop 'adaptive recovery exercised'
  end do
  fixed%plasma%photoelectrons = maxwellian_photoelectrons(55.42562584220407e6_dp, 2.2_dp)
  solver%search = sheath_search_options()
  call solver%solve_equilibrium(fixed, reference, status, message)
  if (status /= SHEATH_OK) error stop 'restored reference'
  call solver%solve_equilibrium_candidates(fixed, candidates, status, message, diagnostics)
  if (status /= SHEATH_OK .or. size(candidates) /= 1 .or. diagnostics%deflations(1) == 0) &
      error stop 'physical deflated candidates'
  call loaded%clear()
  fixed_sweep(1) = fixed
  call solver%build_equilibrium_atlas(fixed_sweep, loaded, status, message, deflation=.true.)
  if (status /= SHEATH_OK .or. loaded%size() /= 1) error stop 'deflated map build'
  fixed%plasma%ion_density_m3 = fixed%plasma%ion_density_m3*1e9_dp
  fixed%plasma%photoelectrons = maxwellian_photoelectrons(55.42562584220407e15_dp, 22.0_dp)
  fixed%plasma%ion_temperature_ev = 120.0_dp
  fixed%plasma%electron_temperature_ev = 120.0_dp
  fixed%plasma%ion_entry_speed_mps = fixed%plasma%ion_entry_speed_mps*sqrt(10.0_dp)
  solver%search%method = 'newton'
  solver%search%use_default_guesses = .false.
  solver%search%max_iterations = 0
  solver%equilibrium_atlas = atlas
  call solver%solve_equilibrium(fixed, result, status, message, diagnostics)
  deallocate (solver%equilibrium_atlas)
  if (status /= SHEATH_OK .or. abs(result%surface_potential_v/10.0_dp - reference%surface_potential_v) > 2e-8_dp) &
      error stop 'dimensionless reuse'
  call atlas%clear()
  solver%equilibrium_atlas = atlas
  call solver%solve_equilibrium(fixed, result, status, message, diagnostics)
  deallocate (solver%equilibrium_atlas)
  if (status /= SHEATH_NUMERICAL_FAILURE .or. result%valid) error stop 'empty map is not a root'

  ! Source shape is part of the map family, independent of its amplitude.
  solver = sheath_solver()
  fixed = fixed_entry_equilibrium_input()
  fixed%branch = 'C'
  fixed%plasma%photoelectrons = binned_photoelectrons([0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], &
      [2e10_dp, 5e9_dp, 1e9_dp])
  call solver%solve_equilibrium(fixed, reference, status, message)
  if (status /= SHEATH_OK) error stop 'spectral reference'
  call solver%add_equilibrium_to_atlas(fixed, reference, atlas, status, message)
  if (status /= SHEATH_OK) error stop 'spectral map add'
  open (newunit=unit, status='scratch', form='formatted')
  call atlas%write(unit, io)
  if (io /= 0) error stop 'spectral write'
  rewind (unit)
  call loaded%read(unit, io)
  close (unit)
  if (io /= 0) error stop 'spectral read'
  fixed%plasma%photoelectrons = binned_photoelectrons([0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], &
      [2.02e10_dp, 5.05e9_dp, 1.01e9_dp])
  solver%search%method = 'newton'
  solver%search%use_default_guesses = .false.
  solver%equilibrium_atlas = loaded
  call solver%solve_equilibrium(fixed, result, status, message, diagnostics)
  deallocate (solver%equilibrium_atlas)
  if (status /= SHEATH_OK .or. diagnostics%atlas_hits(3) /= 1) error stop 'spectral map use'
  point = loaded%point(1)
  point%spectrum_shape(2) = point%spectrum_shape(2) + 0.1_dp
  call loaded%predictions(point%key, point%spectrum_shape, 'C', predictions)
  if (size(predictions, 2) /= 0) error stop 'different spectrum not reused'

  search%method = 'newton'
  continuation%initial_step = 0.15_dp
  continuation%max_step = 0.25_dp
  diagnostics = sheath_search_diagnostics()
  call continue_guarded_system(1, fold_residual, [1.0_dp], search, continuation, diagnostics, 1, y, success)
  if (success .or. diagnostics%continuation_retries(1) == 0) error stop 'parameter fold limit'
  continuation%method = 'arclength'
  diagnostics = sheath_search_diagnostics()
  call continue_guarded_system(1, fold_residual, [1.0_dp], search, continuation, diagnostics, 1, y, success)
  if (.not. success .or. abs(y(1) + 1.324717957244746_dp) > 2e-9_dp) error stop 'arclength follows folds'
  starts(1, :) = [0.9_dp, 0.1_dp, -0.9_dp]
  search%method = 'auto'
  diagnostics = sheath_search_diagnostics()
  call find_guarded_roots(1, polynomial, starts, search, diagnostics, 1, roots, 16, .true.)
  if (size(roots, 2) /= 3 .or. diagnostics%deflations(1) == 0) error stop 'deflation finds distinct roots'
  do i = 1, 3
    if (abs(roots(1, i)**3 - roots(1, i)) > 1e-10_dp) error stop 'original residual'
  end do
  print *, 'Atlas, continuation, folds and deflation checks passed.'
contains
  subroutine fold_residual(value, t, f, valid)
    real(dp), intent(in) :: value(:), t
    real(dp), intent(out) :: f(:)
    logical, intent(out) :: valid
    f(1) = value(1)**3 - value(1) + t
    valid = .true.
  end subroutine
  subroutine polynomial(value, f, valid)
    real(dp), intent(in) :: value(:)
    real(dp), intent(out) :: f(:)
    logical, intent(out) :: valid
    f(1) = value(1)**3 - value(1)
    valid = .true.
  end subroutine
end program test_atlas
