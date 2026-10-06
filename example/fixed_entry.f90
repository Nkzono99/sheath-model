program fixed_entry
  use sheath_model
  implicit none
  type(sheath_solver) :: solver
  type(fixed_entry_equilibrium_input) :: input
  type(sheath_equilibrium_result) :: result
  type(sheath_state_result) :: state
  integer(i32) :: status
  character(len=256) :: message

  input%branch = 'A'
  ! auto / newton / lm; bracket is available for J=0 B/C.
  solver%search%method = 'newton'
  input%plasma%ion_density_m3 = 8.7e6_dp
  input%plasma%ion_entry_speed_mps = 405299.88897111727_dp
  input%plasma%ion_temperature_ev = 12.0_dp
  input%plasma%ion_pressure_factor = 3.0_dp
  input%plasma%electron_temperature_ev = 12.0_dp
  input%plasma%electron_drift_mps = 0.0_dp
  input%plasma%photoelectrons = maxwellian_photoelectrons(55.42562584220407e6_dp, 2.2_dp)
  call solver%solve_equilibrium(input, result, status, message)
  if (status /= SHEATH_OK) then
    print *, trim(message)
    stop 1
  end if
  call evaluate_sheath_state(input%plasma, 'A', result%surface_potential_v, state, status, message, &
      minimum_potential_v=result%minimum_potential_v)
  if (status /= SHEATH_OK .or. .not. state%admissible) stop 1
  print '(a,f14.8)', 'Surface potential [V]: ', result%surface_potential_v
  print '(a,f14.8)', 'Minimum potential [V]: ', result%minimum_potential_v
  print '(a,es14.6)', 'Surface field [V/m]: ', state%electric_field_v_m
  print '(a,es14.6)', 'Inward ion flux [m^-2 s^-1]: ', result%ion_inward_flux_m2_s
end program fixed_entry
