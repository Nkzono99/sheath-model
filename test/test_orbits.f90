program test_orbits
  use sheath_model
  use sheath_model_orbits, only: electron_density
  use sheath_model_core, only: zhao_params_type, type_a_e2_sum_at_infinity
  implicit none
  type(zhao_equilibrium_input) :: input
  type(zhao_equilibrium_result) :: root
  type(zhao_profile_result) :: profile
  type(zhao_field_input) :: field_input
  type(zhao_field_result), allocatable :: candidates(:)
  type(zhao_params_type) :: p
  real(dp) :: free, reflected, psi, previous_phi, direct, a, width, weight, cutoff
  real(dp), parameter :: states(3) = [0.25_dp, -0.03_dp, -0.3_dp]
  real(dp), parameter :: barriers(3) = [0.0_dp, -0.08_dp, -0.5_dp]
  real(dp), parameter :: pi = acos(-1.0_dp)
  real(dp) :: integral_zero, integral_small
  integer(i32) :: status
  integer :: i, j
  character(len=512) :: message
  do i = 0, 20
    psi = real(i, dp)/4.0_dp
    call electron_density(psi, 0.0_dp, 0.0_dp, free, reflected)
    if (abs(free - 0.5_dp*exp(psi)*erfc(sqrt(psi))) > 2e-9_dp) error stop 'zero drift cutoff'
    if (reflected /= 0.0_dp) error stop 'spurious B reflection'
  end do
  p%n_swi_inf_m3 = 8.7e6_dp
  p%n_phe_ref_m3 = 64e6_dp
  p%n_phe0_m3 = 32e6_dp
  p%t_phe_ev = 2.2_dp
  p%t_swe_ev = 12.0_dp
  p%tau = p%t_swe_ev/p%t_phe_ev
  p%mach = 5.0_dp
  p%u = 0.0_dp
  integral_zero = type_a_e2_sum_at_infinity(p, 3.0_dp, -1.0_dp, 8.0e6_dp)
  do i = -1, 1, 2
    p%u = i*1e-10_dp
    integral_small = type_a_e2_sum_at_infinity(p, 3.0_dp, -1.0_dp, 8.0e6_dp)
    if (abs(integral_small - integral_zero) > 1e-9_dp) error stop 'u=0 integral continuity'
  end do
  ! Independent composite Simpson integral in upstream velocity (different measure and quadrature).
  do i = 1, 3
    call electron_density(states(i), barriers(i), 0.2_dp, free, reflected)
    cutoff = sqrt(-barriers(i))
    width = 12.0_dp/32768.0_dp
    direct = 0.0_dp
    do j = 0, 32768
      a = cutoff + j*width
      weight = 2.0_dp
      if (mod(j, 2) == 1) weight = 4.0_dp
      if (j == 0 .or. j == 32768) weight = 1.0_dp
      direct = direct + weight*a*exp(-(a - 0.2_dp)**2)/sqrt(a*a + states(i))
    end do
    direct = direct*width/(3.0_dp*sqrt(pi))
    if (abs(free - direct) > 1e-10_dp) error stop 'orbit density vs independent upstream integral'
  end do
  input = zhao_equilibrium_input(branch='A')
  call solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_NO_PHYSICAL_SOLUTION .or. root%valid) error stop 'drifting A asymptotic obstruction'
  input%branch = 'auto'
  call solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%branch /= 'B') error stop 'auto must skip inadmissible roots'
  input = zhao_equilibrium_input(branch='A', electron_drift_mode='zero')
  call solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK) error stop 'A60 acceptance'
  input%sun_elevation_deg = 19.0_dp
  call solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_OK .or. root%surface_potential_v >= 0.0_dp) error stop 'negative surface A'
  call solve_profile(input, zhao_profile_options(), profile, status, message)
  if (status /= SHEATH_OK) then
    print *, trim(message)
    error stop 'accepted A profile'
  end if
  input = zhao_equilibrium_input(branch='C', sun_elevation_deg=1.0_dp, electron_drift_mode='zero')
  call solve_equilibrium(input, root, status, message)
  if (status /= SHEATH_NO_PHYSICAL_SOLUTION .or. root%valid) error stop 'unphysical C accepted'
  field_input = zhao_field_input()
  field_input%photoelectron_source_density_m3 = 64e6_dp*sin(20.0_dp*acos(-1.0_dp)/180.0_dp)
  field_input%electron_drift_mps = 0.0_dp
  field_input%ion_drift_mps = 468e3_dp*sin(20.0_dp*acos(-1.0_dp)/180.0_dp)
  do i = -1, 1
    field_input%electric_field_v_m = real(i, dp)*0.01_dp
    call solve_prescribed_field_candidates(field_input, candidates, status, message)
    if (status /= SHEATH_OK) error stop 'field transition candidate search'
    if (size(candidates) /= 1) error stop 'transition duplicates must coalesce'
    if (candidates(1)%boundary_potential_v >= 0.0_dp) error stop 'transition is nonflat'
    if (i <= 0 .and. candidates(1)%branch /= 'C') error stop 'negative field C transition'
    if (i > 0 .and. candidates(1)%branch /= 'A') error stop 'positive field A transition'
    if (i > -1) then
      if (abs(candidates(1)%boundary_potential_v - previous_phi) > 0.01_dp) error stop 'discontinuous transition'
    end if
    previous_phi = candidates(1)%boundary_potential_v
  end do
  print *, 'Orbit and acceptance checks passed.'
end program test_orbits
