! SPDX-License-Identifier: MIT
!> Build/read a field map, correct the query, and inspect all located candidates.
program field_atlas
  use sheath_model
  implicit none
  type(sheath_solver) :: solver
  type(prescribed_field_input) :: inputs(2), query
  type(sheath_field_atlas) :: atlas
  type(sheath_search_diagnostics) :: diagnostics
  type(prescribed_field_result), allocatable :: candidates(:)
  integer(i32) :: status
  integer :: i, unit, io
  logical :: exists
  character(len=512) :: path, message
  query%electron_drift_mps = 0.0_dp
  query%photoelectrons = maxwellian_photoelectrons(55.42562584220407e6_dp, 2.2_dp)
  query%electric_field_v_m = 1.62_dp
  path = ''
  call get_command_argument(1, path)
  exists = .false.
  if (len_trim(path) > 0) inquire (file=trim(path), exist=exists)
  if (exists) then
    open (newunit=unit, file=trim(path), status='old', form='formatted')
    call atlas%read(unit, io)
    close (unit)
    if (io /= 0) error stop 'Cannot read the prescribed-field atlas.'
  else
    do i = 1, 2
      inputs(i) = query
      inputs(i)%electric_field_v_m = 1.56_dp + 0.04_dp*i
    end do
    call solver%build_field_atlas(inputs, atlas, status, message)
    if (status /= SHEATH_OK) then
      print *, trim(message)
      error stop 'Cannot build the prescribed-field atlas.'
    end if
    if (len_trim(path) > 0) then
      open (newunit=unit, file=trim(path), status='replace', form='formatted')
      call atlas%write(unit, io)
      close (unit)
      if (io /= 0) error stop 'Cannot write the prescribed-field atlas.'
    end if
  end if
  ! Choose parameter or arclength; steps use dimensionless plasma/field coordinates.
  solver%continuation%method = 'arclength'
  solver%search%method = 'newton'
  solver%search%max_starts = 8
  solver%field_atlas = atlas
  call solver%solve_prescribed_field_candidates(query, candidates, status, message, diagnostics, &
      deflation=.true.)
  deallocate (solver%field_atlas)
  if (status /= SHEATH_OK) then
    print *, trim(message)
    error stop 'No admissible candidate located.'
  end if
  print *, 'Stored roots:', atlas%size(), '; specified normal field [V/m]:', query%electric_field_v_m
  print *, 'Located candidates:', size(candidates)
  do i = 1, size(candidates)
    print '(a,1x,3(es16.8,1x))', candidates(i)%branch, candidates(i)%boundary_potential_v, &
        candidates(i)%minimum_potential_v, candidates(i)%net_current_a_m2
  end do
  print *, 'Atlas starts A/B/C:', diagnostics%atlas_starts
  print *, 'Continuation steps A/B/C:', diagnostics%continuation_steps
end program
