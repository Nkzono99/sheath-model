! SPDX-License-Identifier: MIT
program solution_atlas
  use sheath_model
  implicit none
  type(fixed_entry_equilibrium_input) :: inputs(2), query
  type(sheath_equilibrium_atlas) :: atlas
  type(zhao_equilibrium_result) :: root
  type(zhao_equilibrium_result), allocatable :: candidates(:)
  type(sheath_search_diagnostics) :: diagnostics
  integer(i32) :: status
  integer :: i, unit, io
  logical :: exists
  character(len=512) :: message, path
  do i = 1, 2
    inputs(i)%branch = 'A'
    inputs(i)%plasma%ion_temperature_ev = 12.0_dp
    inputs(i)%plasma%ion_pressure_factor = 3.0_dp
    inputs(i)%plasma%photoelectrons = maxwellian_photoelectrons(real(51 + 3*i, dp)*1e6_dp, 2.2_dp)
  end do
  ! Select 'arclength' to follow simple folds along the dimensionless input path.
  atlas%continuation%method = 'parameter'
  call get_command_argument(1, path)
  exists = .false.
  if (len_trim(path) > 0) inquire (file=trim(path), exist=exists)
  if (.not. exists) then
    call build_equilibrium_atlas(inputs, atlas, status, message)
    if (status /= SHEATH_OK) then
      print *, trim(message)
      stop 1
    end if
  end if
  if (len_trim(path) > 0 .and. .not. exists) then
    open (newunit=unit, file=trim(path), status='replace', form='formatted', iostat=io)
    if (io /= 0) stop 1
    call atlas%write(unit, io)
    close (unit)
    if (io /= 0) stop 1
  end if
  if (len_trim(path) > 0) then
    call atlas%clear()
    open (newunit=unit, file=trim(path), status='old', form='formatted', iostat=io)
    if (io /= 0) stop 1
    call atlas%read(unit, io)
    close (unit)
    if (io /= 0) stop 1
  end if
  query = inputs(1)
  query%plasma%photoelectrons = maxwellian_photoelectrons(55.42562584220407e6_dp, 2.2_dp)
  query%search%method = 'newton'
  query%search%max_starts = 1
  call solve_equilibrium(query, root, status, message, diagnostics, atlas=atlas)
  if (status /= SHEATH_OK) then
    print *, trim(message)
    stop 1
  end if
  print '(a,i0)', 'Stored roots: ', atlas%size()
  print '(a,2f14.8)', 'Surface / minimum potential [V]: ', root%surface_potential_v, root%minimum_potential_v
  print '(a,i0)', 'Starts using the atlas: ', diagnostics%atlas_starts(1)
  call solve_equilibrium_candidates(query, candidates, status, message, atlas=atlas, deflation=.true.)
  if (status /= SHEATH_OK) stop 1
  print '(a,i0)', 'Located admissible candidates: ', size(candidates)
end program solution_atlas
