! SPDX-License-Identifier: MIT
! The outer-root iteration must follow observed sources, keep its state on failure and survive a restart.
program test_matching_iteration
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sheath_model
  implicit none
  real(dp), parameter :: qe = 1.602176634e-19_dp, electron_mass = 9.1093837015e-31_dp
  real(dp), parameter :: pi = acos(-1.0_dp)
  type(sheath_solver) :: solver
  type(plasma_input) :: plasma, low_sun
  type(fixed_entry_equilibrium_input) :: direct
  type(sheath_equilibrium_result) :: root
  type(sheath_matching_iteration) :: iteration, restored, other
  type(sheath_matching_conditions) :: before, after
  type(emission_source) :: binned, empty, kept, saved
  integer(i32) :: status
  integer :: unit, i
  real(dp), allocatable :: edges(:), flux(:)
  real(dp) :: gamma, temperature
  character(len=512) :: message
  character(len=10), parameter :: accesses(2) = ['stream    ', 'sequential']

  temperature = 2.2_dp
  plasma%electron_drift_mps = 0.0_dp
  plasma%emission = maxwellian_emission(64e6_dp*sin(pi/3.0_dp), temperature)

  call iteration%update(plasma%emission, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'update needs initialization')
  call iteration%initialize(plasma, solver, status, message)
  call ok('initialize')
  before = iteration%conditions()
  direct%plasma = plasma
  direct%branch = 'A'
  call solver%solve_equilibrium(direct, root, status, message)
  call ok('direct A root')
  call check(before%valid .and. before%branch == 'A', 'auto starts with A')
  call near(before%plane_potential_v, root%surface_potential_v, 1e-12_dp, 'same root as a direct solve')
  call check(iteration%accepted_solves() == 1 .and. iteration%relative_change() == 0.0_dp, 'initial counters')

  ! A binned half-Maxwellian on the outflow grid reproduces the analytic root up to bin discretization.
  gamma = 64e6_dp*sin(pi/3.0_dp)*sqrt(2.0_dp*qe*temperature/electron_mass)/(2.0_dp*sqrt(pi))
  edges = log_energy_edges()
  flux = gamma*(exp(-edges(:size(edges) - 1)/temperature) - exp(-edges(2:)/temperature))
  call observed_outflow_source(edges, flux, binned, status, message)
  call ok('binned source')
  call near(binned%mean_normal_energy(), temperature, 1e-2_dp, 'binned mean normal energy')
  call iteration%update(binned, status, message)
  call ok('update with binned source')
  after = iteration%conditions()
  call check(after%branch == 'A', 'branch kept')
  call check(abs(after%plane_potential_v - before%plane_potential_v) < 0.02_dp, 'binned wall potential')
  call near(after%emitted_escape_flux_m2_s, before%emitted_escape_flux_m2_s, 1e-2_dp, 'binned escape flux')
  call check(iteration%accepted_solves() == 2, 'accepted update counted')
  call check(iteration%relative_change() > 0.0_dp .and. iteration%relative_change() < 1e-2_dp, 'small source change')
  kept = iteration%source()
  call check(kept%is_binned(), 'accepted source kept')

  ! Failed and invalid updates leave the accepted state untouched.
  call other%initialize(plasma, solver, status, message, branch='a')
  call ok('explicit A')
  before = other%conditions()
  empty = binned_emission([0.0_dp, 1.0_dp], [0.0_dp])
  call other%update(empty, status, message)
  call check(status /= SHEATH_OK, 'A has no root without emission')
  after = other%conditions()
  call check(after%plane_potential_v == before%plane_potential_v .and. other%accepted_solves() == 1, 'failure keeps root')
  kept = other%source()
  call check(.not. kept%is_binned(), 'failure keeps source')
  call other%update(binned_emission([1.0_dp, 0.0_dp], [1e10_dp]), status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'invalid source rejected')
  after = other%conditions()
  call check(after%plane_potential_v == before%plane_potential_v, 'invalid source keeps root')

  ! Branch policy validation and a low-sun C start.
  call other%initialize(plasma, solver, status, message, branch='D')
  call check(status == SHEATH_INVALID_ARGUMENT, 'unknown branch')
  call other%initialize(plasma, solver, status, message, initial_order='AAB')
  call check(status == SHEATH_INVALID_ARGUMENT, 'order must be a permutation')
  low_sun%electron_drift_mps = 0.0_dp
  low_sun%ion_entry_speed_mps = 468e3_dp*sin(10.0_dp*pi/180.0_dp)
  low_sun%emission = maxwellian_emission(64e6_dp*sin(10.0_dp*pi/180.0_dp), temperature)
  call other%initialize(low_sun, solver, status, message, initial_order='cab')
  call ok('low-sun start')
  after = other%conditions()
  call check(after%branch == 'C', 'low-sun order starts with C')
  call other%update(maxwellian_emission(1.01_dp*64e6_dp*sin(10.0_dp*pi/180.0_dp), temperature), status, message)
  call ok('low-sun update')
  after = other%conditions()
  call check(after%branch == 'C', 'update prefers the current branch')
  call near(other%relative_change(), 0.01_dp, 1e-10_dp, 'relative flux change')

  ! Restart through both unformatted access modes.
  do i = 1, 2
    open (newunit=unit, status='scratch', form='unformatted', access=trim(accesses(i)))
    call iteration%write_state(unit, status, message)
    call ok('write '//trim(accesses(i)))
    rewind (unit)
    call restored%read_state(unit, plasma, solver, status, message)
    call ok('read '//trim(accesses(i)))
    close (unit)
    before = iteration%conditions()
    after = restored%conditions()
    call check(after%branch == before%branch, 'restored branch')
    call near(after%plane_potential_v, before%plane_potential_v, 1e-9_dp, 'restored wall potential')
    call near(after%emitted_escape_flux_m2_s, before%emitted_escape_flux_m2_s, 1e-9_dp, 'restored source')
    call check(restored%accepted_solves() == iteration%accepted_solves(), 'restored count')
    call check(restored%relative_change() == iteration%relative_change(), 'restored change')
    kept = iteration%source()
    saved = restored%source()
    call check(all(saved%search_breakpoints() == kept%search_breakpoints()), 'restored spectrum')
  end do

  open (newunit=unit, status='scratch', form='unformatted', access='stream')
  write (unit) 99
  rewind (unit)
  call restored%read_state(unit, plasma, solver, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'unknown state version rejected')
  close (unit)
  call restored%write_state(unit, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'failed read leaves no state to write')

  print *, 'Matching iteration updates, failure retention and restart passed.'
contains
  subroutine ok(label)
    character(len=*), intent(in) :: label
    call check(status == SHEATH_OK, label)
  end subroutine
  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, 'FAIL: ', label, ' ', trim(message)
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
