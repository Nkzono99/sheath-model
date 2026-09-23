! Evaluate a static response curve without specifying E_H or a time-stepping equation.
program potential_scan
  use sheath_model
  implicit none
  type(zhao_plasma_input) :: plasma
  type(zhao_state_result) :: state
  integer(i32) :: status
  integer :: i
  real(dp) :: phi
  character(len=256) :: message
  plasma%electron_drift_mps = 0.0_dp
  plasma%photoelectrons = binned_photoelectrons([0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], &
      [2e12_dp, 5e11_dp, 1e11_dp])
  print '(a)', 'phi_H_V,E_squared_V2_m2,E_V_m,N_e_m3,J_A_m2,admissible'
  do i = 1, 100
    phi = real(i, dp)*0.02_dp
    call evaluate_sheath_state(plasma, 'B', phi, state, status, message)
    if (status /= SHEATH_OK) then
      print *, trim(message)
      error stop 1
    end if
    print '(5(es22.14,","),l1)', phi, state%boundary_field_squared_v2_m2, state%electric_field_v_m, &
        state%ambient_electron_density_m3, state%net_current_a_m2, state%admissible
  end do
end program
