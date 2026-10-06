! Compare the two models with the same 60-degree solar-wind/photoelectron parameters.
program compare_closures
  use sheath_model
  implicit none
  type(sheath_solver) :: solver
  type(zhao_equilibrium_input) :: equilibrium_input
  type(sheath_equilibrium_result) :: equilibrium
  type(prescribed_field_input) :: field_input
  type(prescribed_field_result) :: response
  integer(i32) :: status
  character(len=256) :: message

  equilibrium_input%branch = 'A'
  equilibrium_input%electron_drift_mode = 'zero'
  call solver%solve_equilibrium(equilibrium_input, equilibrium, status, message)
  call require_success()
  print *, 'Zhao J=0: phi0 [V], J_z [A/m^2]'
  print *, equilibrium%surface_potential_v, equilibrium%net_current_a_m2

  field_input%branch = 'A'
  field_input%electron_drift_mps = 0.0_dp
  field_input%electric_field_v_m = 1.62_dp
  field_input%photoelectrons = maxwellian_photoelectrons(5.5425625842204072e7_dp, 2.2_dp)
  call solver%solve_prescribed_field(field_input, response, status, message)
  call require_success()
  print *, 'Prescribed E_H: E_H [V/m], phi_H [V], J_z [A/m^2]'
  print *, field_input%electric_field_v_m, response%boundary_potential_v, response%net_current_a_m2
contains
  subroutine require_success()
    if (status /= SHEATH_OK) then
      print *, trim(message)
      stop 1
    end if
  end subroutine require_success
end program compare_closures
