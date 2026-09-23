program test_sheath_model
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sheath_model
  implicit none
  real(dp), parameter :: eps0 = 8.8541878128e-12_dp, qe = 1.602176634e-19_dp
  type(zhao_equilibrium_input) :: equilibrium_input
  type(zhao_equilibrium_result) :: root
  type(zhao_density_result) :: density
  type(zhao_profile_options) :: options
  type(zhao_profile_result) :: profile
  real(dp) :: derivative, midpoint
  integer(i32) :: status
  integer :: i, n, branch_index
  character(len=512) :: message
  character(len=1), parameter :: branches(3) = ['A', 'B', 'C']
  ! Independently checked with sheath_model.solver (SciPy root) at alpha=60,20,10 deg.
  real(dp), parameter :: phi_reference(3) = [3.840189093737787_dp, 1.6159581059198893_dp, -4.237755720386508_dp]
  real(dp), parameter :: density_reference(3) = [9.916982425098725e6_dp, 6.898994373424846e6_dp, 8.510500075540943e6_dp]

  do branch_index = 1, 3
    equilibrium_input = zhao_equilibrium_input(branch=branches(branch_index), electron_drift_mode='zero')
    if (branch_index == 2) equilibrium_input%sun_elevation_deg = 20.0_dp
    if (branch_index == 3) equilibrium_input%sun_elevation_deg = 10.0_dp
    call solve_equilibrium(equilibrium_input, root, status, message)
    call ok('equilibrium '//branches(branch_index))
    call check(root%valid .and. root%branch == branches(branch_index), 'equilibrium branch')
    call near(root%surface_potential_v, phi_reference(branch_index), 1e-7_dp, 'Python reference potential')
    call near(root%ambient_electron_density_m3, density_reference(branch_index), 0.1_dp, 'Python reference density')
    call evaluate_density(equilibrium_input, root, 0.0_dp, density, status, message, side='upper')
    call ok('far-field density')
    call near(density%charge_c_m3, 0.0_dp, qe*equilibrium_input%ion_density_m3*1e-8_dp, 'charge neutrality')
    write (*, '(a,1x,a,3es25.16)') 'equilibrium', root%branch, root%surface_potential_v, &
      root%minimum_potential_v, root%ambient_electron_density_m3
    call solve_profile(equilibrium_input, options, profile, status, message)
    call ok('profile '//branches(branch_index))
    n = size(profile%z_m)
    call check(n > 30, 'profile size')
    call near(profile%potential_v(1), root%surface_potential_v, 1e-12_dp, 'surface potential')
    call check(all(profile%z_m(2:) > profile%z_m(:n - 1)), 'strictly increasing heights')
    call check(all(ieee_is_finite(profile%electric_field_v_m)), 'finite profile field')
    ! Independent differential checks: E=-dphi/dz and dE/dz=rho/eps0.
    do i = 40, n - 40, max(1, n/12)
      if (root%branch == 'A' .and. abs(i - options%points_per_segment) < 10) cycle
      derivative = -(profile%potential_v(i + 1) - profile%potential_v(i - 1))/(profile%z_m(i + 1) - profile%z_m(i - 1))
      call near(profile%electric_field_v_m(i), derivative, 0.003_dp*max(1e-3_dp, abs(derivative)), 'E=-grad(phi)')
      derivative = (profile%electric_field_v_m(i + 1) - profile%electric_field_v_m(i - 1))/ &
                   (profile%z_m(i + 1) - profile%z_m(i - 1))
      midpoint = profile%density(i)%charge_c_m3/eps0
      call near(derivative, midpoint, 0.005_dp*max(1e-4_dp, abs(midpoint)), 'Poisson equation')
    end do
    if (root%branch == 'A') then
      call check(any(profile%electric_field_v_m < 0.0_dp), 'A field sign reversal')
      call near(minval(profile%potential_v), root%minimum_potential_v, 1e-12_dp, 'A turning point')
    end if
  end do
  ! A shared density factor cannot change the voltage; Debye length scales as n^(-1/2).
  do i = -2, 2, 2
    equilibrium_input = zhao_equilibrium_input(branch='A', electron_drift_mode='zero')
    equilibrium_input%ion_density_m3 = equilibrium_input%ion_density_m3*10.0_dp**i
    equilibrium_input%photoelectron_reference_density_m3 = equilibrium_input%photoelectron_reference_density_m3*10.0_dp**i
    call solve_equilibrium(equilibrium_input, root, status, message)
    call ok('density scale invariance')
    call near(root%surface_potential_v, phi_reference(1), 1e-6_dp, 'density scaling voltage')
    call near(root%ambient_electron_density_m3/10.0_dp**i, density_reference(1), 0.1_dp, 'density scaling electron density')
  end do
  equilibrium_input%sun_elevation_deg = 0.0_dp
  call solve_equilibrium(equilibrium_input, root, status, message)
  call check(status == sheath_invalid_argument .and. .not. root%valid, 'zero-elevation degeneracy')
  call solve_profile(equilibrium_input, options, profile, status, message)
  call check(.not. allocated(profile%z_m), 'failed profile clears old allocation')
  print *, 'All sheath model contract and physics checks passed.'
contains
  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, 'FAIL: ', label, '; ', trim(message)
      error stop 1
    end if
  end subroutine check
  subroutine ok(label)
    character(len=*), intent(in) :: label
    call check(status == sheath_ok, label)
  end subroutine ok
  subroutine near(actual, wanted, tolerance, label)
    real(dp), intent(in) :: actual, wanted, tolerance
    character(len=*), intent(in) :: label
    if (.not. ieee_is_finite(actual) .or. abs(actual - wanted) > tolerance) then
      print *, 'actual, expected, tolerance:', actual, wanted, tolerance
      call check(.false., label)
    end if
  end subroutine near
end program test_sheath_model
