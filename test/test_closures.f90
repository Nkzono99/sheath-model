! J=0 and E_H share populations, but close different equations.
program test_closures
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sheath_model
  implicit none
  real(dp), parameter :: qe = 1.602176634e-19_dp, eps0 = 8.8541878128e-12_dp, pi = acos(-1.0_dp)
  real(dp), parameter :: elevations(3) = [60.0_dp, 20.0_dp, 10.0_dp]
  character(len=1), parameter :: branches(3) = ['A', 'B', 'C']
  type(zhao_equilibrium_input) :: input
  type(zhao_equilibrium_result) :: root
  type(zhao_profile_result) :: profile
  type(zhao_field_input) :: field_input
  type(zhao_field_result) :: response
  integer(i32) :: status
  integer :: i
  real(dp) :: drift, n_source, vth_e, vth_pe, gamma_e, gamma_i, gamma_pe, current_scale, field
  character(len=512) :: message
  do i = 1, 3
    input = zhao_equilibrium_input(branch=branches(i), sun_elevation_deg=elevations(i))
    call solve_equilibrium(input, root, status, message)
    call ok('J=0 root '//branches(i))
    drift = input%solar_wind_speed_mps*sin(elevations(i)*pi/180.0_dp)
    n_source = input%photoelectron_reference_density_m3*sin(elevations(i)*pi/180.0_dp)
    vth_e = sqrt(2.0_dp*qe*input%electron_temperature_ev/input%electron_mass_kg)
    vth_pe = sqrt(2.0_dp*qe*input%photoelectron_temperature_ev/input%electron_mass_kg)
    ! Direct velocity quadrature is independent of the solver's erfc expression.
    gamma_e = integrate_electron_flux(root%ambient_electron_density_m3, vth_e, drift, &
                                      sqrt(-root%minimum_potential_v/input%electron_temperature_ev) - drift/vth_e)
    gamma_i = input%ion_density_m3*drift
    gamma_pe = n_source*vth_pe/(2.0_dp*sqrt(pi))* &
               exp((root%minimum_potential_v - root%surface_potential_v)/input%photoelectron_temperature_ev)
    current_scale = qe*max(gamma_e, gamma_i, gamma_pe)
    call near(qe*(gamma_e - gamma_i - gamma_pe), 0.0_dp, 1e-8_dp*current_scale, 'independent J=0 current integral')
    call near(root%net_current_a_m2, 0.0_dp, 1e-8_dp*current_scale, 'reported equilibrium current')
    call near(root%electron_inward_flux_m2_s, gamma_e, 1e-8_dp*gamma_e, 'electron flux diagnostic')
    call near(root%ion_inward_flux_m2_s, gamma_i, 1e-12_dp*gamma_i, 'ion flux diagnostic')
    call near(root%photoelectron_escape_flux_m2_s, gamma_pe, 1e-12_dp*gamma_pe, 'escaping PE flux diagnostic')
  end do
  input = zhao_equilibrium_input(branch='A')
  call solve_profile(input, zhao_profile_options(), profile, status, message)
  call ok('J=0 field reconstruction')
  root = profile%equilibrium
  field = profile%electric_field_v_m(1)
  field_input%branch = 'A'
  field_input%root_selection = 'minimum_energy'
  n_source = input%photoelectron_reference_density_m3*sin(input%sun_elevation_deg*pi/180.0_dp)
  vth_pe = sqrt(2.0_dp*qe*input%photoelectron_temperature_ev/input%electron_mass_kg)
  field_input%electric_field_v_m = field
  field_input%photoelectron_source_density_m3 = n_source
  call solve_prescribed_field(field_input, response, status, message)
  call ok('E_H set to the J=0 solution field')
  call near(response%boundary_potential_v, root%surface_potential_v, 3e-5_dp, 'common solution potential')
  call near(response%ambient_electron_density_m3, root%ambient_electron_density_m3, 30.0_dp, 'common solution density')
  call near(response%net_current_a_m2, 0.0_dp, 1e-10_dp, 'common solution current')
  ! A different field must be preserved, without restoring J=0.
  field_input%electric_field_v_m = 1.01_dp*field
  call solve_prescribed_field(field_input, response, status, message)
  call ok('nonzero-current E_H model')
  call near(integrate_type_a_field(response%boundary_potential_v, response%minimum_potential_v, &
                        response%ambient_electron_density_m3), field_input%electric_field_v_m, 2e-6_dp, 'independent Sagdeev field')
  gamma_pe = n_source*vth_pe/(2.0_dp*sqrt(pi))* &
             exp((response%minimum_potential_v - response%boundary_potential_v)/input%photoelectron_temperature_ev)
  current_scale = qe*response%electron_inward_flux_m2_s
  call near(response%photoelectron_escape_flux_m2_s, gamma_pe, 1e-12_dp*gamma_pe, 'E_H escaping flux')
  call near(response%net_current_a_m2, qe*(response%electron_inward_flux_m2_s - &
                                        response%ion_inward_flux_m2_s - gamma_pe), 1e-12_dp*current_scale, 'E_H current diagnostic')
  call check(abs(response%net_current_a_m2) > 1e-3_dp*current_scale, 'E_H model does not impose J=0')
  print *, 'J=0 and E_H closure checks passed.'
contains
  real(dp) function integrate_electron_flux(density, vth, velocity, lower) result(flux)
    real(dp), intent(in) :: density, vth, velocity, lower
    integer, parameter :: panels = 4000
    real(dp) :: x, h, weight
    integer :: j
    h = (max(12.0_dp, lower + 12.0_dp) - lower)/panels
    flux = 0.0_dp
    do j = 0, panels
      x = lower + j*h
      weight = 2.0_dp
      if (mod(j, 2) == 1) weight = 4.0_dp
      if (j == 0 .or. j == panels) weight = 1.0_dp
      flux = flux + weight*(vth*x + velocity)*exp(-x*x)
    end do
    flux = flux*h*density/(3.0_dp*sqrt(pi))
  end function integrate_electron_flux
  real(dp) function integrate_type_a_field(phi0, phim, density) result(e_h)
    real(dp), intent(in) :: phi0, phim, density
    integer, parameter :: panels = 2048
    real(dp) :: t, phi, s_pe, s_e, rho, n_i, n_e, n_pe, weight, integral, vth, drift_ratio
    integer :: j
    vth = sqrt(2.0_dp*qe*field_input%electron_temperature_ev/field_input%electron_mass_kg)
    drift_ratio = field_input%electron_drift_mps/vth
    integral = 0.0_dp
    do j = 0, panels
      t = real(j, dp)/panels
      phi = phim + (phi0 - phim)*t*t
      s_pe = sqrt(max(0.0_dp, (phi - phim)/input%photoelectron_temperature_ev))
      s_e = sqrt(max(0.0_dp, (phi - phim)/field_input%electron_temperature_ev))
      n_i = field_input%ion_density_m3/sqrt(1.0_dp - 2.0_dp*qe*phi/(field_input%ion_mass_kg*field_input%ion_drift_mps**2))
      n_e = 0.5_dp*density*exp(phi/field_input%electron_temperature_ev)*erfc(s_e - drift_ratio)
      n_pe = 0.5_dp*n_source*exp((phi - phi0)/input%photoelectron_temperature_ev)*(1.0_dp + erf(s_pe))
      rho = qe*(n_i - n_e - n_pe)
      weight = 2.0_dp
      if (mod(j, 2) == 1) weight = 4.0_dp
      if (j == 0 .or. j == panels) weight = 1.0_dp
      integral = integral + weight*rho*2.0_dp*(phi0 - phim)*t
    end do
    integral = integral/(3.0_dp*panels)
    call check(integral < 0.0_dp, 'real E_H integral')
    e_h = sqrt(-2.0_dp*integral/eps0)
  end function integrate_type_a_field
  subroutine ok(label)
    character(len=*), intent(in) :: label
    call check(status == sheath_ok, label)
  end subroutine ok
  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, 'FAIL: ', label, '; ', trim(message)
      error stop 1
    end if
  end subroutine check
  subroutine near(actual, expected, tolerance, label)
    real(dp), intent(in) :: actual, expected, tolerance
    character(len=*), intent(in) :: label
    if (.not. ieee_is_finite(actual) .or. abs(actual - expected) > tolerance) then
      print *, actual, expected, tolerance
      call check(.false., label)
    end if
  end subroutine near
end program test_closures
