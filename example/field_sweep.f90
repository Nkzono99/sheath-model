! Follow a short E_H sweep, retaining all candidates and search diagnostics.
program field_sweep
  use sheath_model
  implicit none
  type(zhao_field_input) :: input
  type(zhao_field_search_diagnostics) :: diagnostics
  type(zhao_field_result), allocatable :: previous(:), candidates(:)
  integer(i32) :: status
  integer :: step, i
  character(len=256) :: message

  input%electron_drift_mps = 0.0_dp
  input%photoelectrons = maxwellian_photoelectrons(5.5425625842204072e7_dp, 2.2_dp)
  allocate (previous(0))
  print '(a)', 'E_H [V/m], branch, phi_H [V], J_z [A/m^2]'
  do step = 0, 4
    input%electric_field_v_m = 1.58_dp + 0.01_dp*step
    call solve_prescribed_field_candidates(input, candidates, status, message, diagnostics, previous)
    if (status /= SHEATH_OK) then
      print *, 'Search status: ', status, trim(message)
      cycle
    end if
    do i = 1, size(candidates)
      print '(f8.3,1x,a,2(1x,es16.8))', input%electric_field_v_m, candidates(i)%branch, &
          candidates(i)%boundary_potential_v, candidates(i)%net_current_a_m2
    end do
    print *, 'Unconverged starts (A/B/C): ', diagnostics%unconverged
    call move_alloc(candidates, previous)
  end do
end program field_sweep
