program equilibrium_profile
  use sheath_model
  implicit none
  type(zhao_equilibrium_input) :: input
  type(zhao_profile_options) :: options
  type(zhao_profile_result) :: profile
  integer(i32) :: status
  integer :: i
  character(len=256) :: message
  input%branch = 'A'
  input%electron_drift_mode = 'zero'
  input%sun_elevation_deg = 60.0_dp
  call solve_profile(input, options, profile, status, message)
  if (status /= sheath_ok) then
    print *, trim(message)
    stop 1
  end if
  print '(a)', 'z_m,potential_v,electric_field_v_m,charge_c_m3'
  do i = 1, size(profile%z_m)
    print '(es24.16,3(",",es24.16))', profile%z_m(i), profile%potential_v(i), &
      profile%electric_field_v_m(i), profile%density(i)%charge_c_m3
  end do
end program equilibrium_profile
