! SPDX-License-Identifier: MIT
program test_solver_api
  use sheath_model
  implicit none
  type(sheath_solver) :: solver
  type(fixed_entry_equilibrium_input) :: first, second, invalid
  type(sheath_equilibrium_result) :: root, other, repeated
  type(sheath_profile_result) :: profile
  type(sheath_equilibrium_atlas) :: atlas
  type(sheath_search_diagnostics) :: diagnostics
  integer(i32) :: status
  character(len=512) :: message

  first%branch = 'A'
  first%plasma%photoelectrons = maxwellian_photoelectrons(55.42562584220407e6_dp, 2.2_dp)
  second = first
  second%plasma%photoelectrons = maxwellian_photoelectrons(60e6_dp, 2.2_dp)
  call solver%solve_equilibrium(first, root, status, message)
  call check(status == SHEATH_OK, 'first input')
  call solver%solve_equilibrium(second, other, status, message)
  call check(status == SHEATH_OK, 'different conditions on the same solver')
  call solver%solve_equilibrium(first, repeated, status, message)
  call check(status == SHEATH_OK, 'repeated input')
  call check(root%surface_potential_v == repeated%surface_potential_v, 'call order independence')
  call check(abs(root%surface_potential_v - other%surface_potential_v) > 1e-3_dp, 'physical inputs affect the root')
  call solver%add_equilibrium_to_atlas(first, root, atlas, status, message)
  call check(status == SHEATH_OK .and. atlas%size() == 1, 'explicit registration')
  solver%equilibrium_atlas = atlas
  solver%search%method = 'newton'
  solver%search%use_default_guesses = .false.
  solver%search%max_iterations = 0
  call solver%solve_equilibrium(first, repeated, status, message, diagnostics)
  call check(status == SHEATH_OK .and. diagnostics%atlas_hits(1) == 1, 'attached map')
  call check(solver%equilibrium_atlas%size() == 1 .and. atlas%size() == 1, 'queries do not modify maps')
  deallocate (solver%equilibrium_atlas)
  call solver%solve_equilibrium(first, repeated, status, message)
  call check(status == SHEATH_NUMERICAL_FAILURE, 'search disabled')
  call solver%build_profile(first, root, profile, status, message)
  call check(status == SHEATH_OK .and. allocated(profile%z_m), 'profile reconstruction does not search')
  call solver%build_profile(second, root, profile, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT .and. .not. allocated(profile%z_m), 'mismatched root rejected')
  invalid = first
  invalid%plasma%ion_density_m3 = -1.0_dp
  call solver%solve_equilibrium(invalid, repeated, status, message, diagnostics)
  call check(status == SHEATH_INVALID_ARGUMENT .and. .not. repeated%valid, 'invalid physical conditions')
  call check(.not. any(diagnostics%searched), 'per-call diagnostics')
  print *, 'Solver responsibilities and reuse checks passed.'
contains
  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, trim(label), trim(message)
      error stop 1
    end if
  end subroutine
end program
