! SPDX-License-Identifier: MIT
! Public API consumer regressions, without any application dependency.
program test_delegated_physics
  use sheath_model
  use sheath_model_constants, only: qe, eps0, pi
  implicit none
  type(sheath_solver) :: solver
  type(prescribed_field_input) :: input
  type(sheath_state_result) :: state
  type(prescribed_field_result) :: previous(1)
  type(prescribed_field_result), allocatable :: roots(:)
  type(sheath_search_diagnostics) :: diagnostics
  real(dp), allocatable :: edges(:), flux(:)
  real(dp) :: scale, closure, density, leading, gamma
  integer(i32) :: status
  integer :: i, pass
  character(len=512) :: message
  ! Exact masses and temperatures belong to the physical input, not the caller.
  input = prescribed_field_input(branch='A', ion_density_m3=5e6_dp, ion_entry_speed_mps=4e5_dp, &
      electron_drift_mps=0.0_dp, electron_mass_kg=9.1093837139e-31_dp)
  input%electron_temperature_ev = 10.0_dp*1.160451812e4_dp*1.380649e-23_dp/qe
  call load_source('test/fixtures/nonmonotonic_connection_spectrum.csv')
  call evaluate_sheath_state(input, 'A', 6.597581495239707_dp, state, status, message, &
      minimum_potential_v=-0.25413329160991566_dp)
  call check(status == SHEATH_OK .and. state%admissible, 'spectral A connected state')
  call check(abs(state%electric_field_v_m - 2.1163398618942133_dp) < 2e-12_dp, 'A independent field reference')
  closure = eps0*state%electric_field_v_m - 1.9250395951168886e-11_dp + 2*state%net_current_a_m2
  call check(abs(closure) < 2e-18_dp, 'caller-formed charge balance from generic state')
  call check(abs(state%connection_residual_normalized) < 1e-10_dp, 'normalized upper connection')
  scale = input%photoelectrons%potential_scale(input%electron_temperature_ev)
  call check(abs(state%connection_residual_normalized - state%connection_residual_v2_m2*eps0*sqrt(scale)/ &
      (qe*input%ion_density_m3*(-state%minimum_potential_v)**1.5_dp)) < 1e-12_dp, 'SI and normalized residual agree')
  input%electric_field_v_m = state%electric_field_v_m
  call solver%solve_prescribed_field_candidates(input, roots, status, message)
  call check(status == SHEATH_OK, 'spectral A field inversion')
  call check(any(abs(roots%boundary_potential_v - 6.597581495239707_dp) < 1e-7_dp), 'spectral A recovered')

  ! A collapsed Maxwellian segment has tiny E^2 but a finite nonzero connection.
  input%electron_temperature_ev = 10.0_dp
  input%ion_entry_speed_mps = 309496.90072670573_dp
  input%electron_mass_kg = 9.1093837015e-31_dp
  gamma = 1989307674603.747_dp
  input%photoelectrons = maxwellian_photoelectrons(0.0_dp, 2.0_dp)
  input%photoelectrons = input%photoelectrons%with_outward_flux(gamma, input%electron_mass_kg)
  call evaluate_sheath_state(input, 'A', 0.22886728419952757_dp, state, status, message, minimum_potential_v=-8e-15_dp)
  call check(status == SHEATH_OK .and. .not. state%admissible, 'collapsed Maxwellian A rejected')
  call check(state%physical_status == SHEATH_NO_PHYSICAL_SOLUTION, 'finite evaluation differs from physical acceptance')
  density = gamma*2*sqrt(pi)/sqrt(2*qe*2.0_dp/input%electron_mass_kg)
  leading = 2/(3*sqrt(pi))*(density*exp(-0.22886728419952757_dp/2.0_dp) - &
      state%ambient_electron_density_m3/sqrt(5.0_dp))/input%ion_density_m3
  call check(abs(state%connection_residual_normalized - leading) < 2e-7_dp, 'independent shallow-depth asymptotic')
  call check(abs(state%connection_residual_normalized - 0.47991033405427047_dp) < 2e-7_dp, 'nondegenerate residual')
  call evaluate_sheath_state(input, 'A', 0.22886728419952757_dp, state, status, message, minimum_potential_v=-1e-100_dp)
  call check(status == SHEATH_OK .and. .not. state%admissible, 'sub-ulp Maxwellian depth remains nonconnecting')
  call check(abs(state%connection_residual_normalized - leading) < 2e-7_dp, 'stable depth-normalized limit')
  ! The same guard also applies to a bin source; no source-dependent acceptance.
  input%photoelectrons = binned_photoelectrons([0.0_dp, 1.0_dp, 3.0_dp, 8.0_dp], [2e12_dp, 5e11_dp, 1e11_dp])
  call evaluate_sheath_state(input, 'A', 0.5_dp, state, status, message, minimum_potential_v=-1e-12_dp)
  call check(status == SHEATH_OK .and. .not. state%admissible, 'collapsed bin A rejected')
  call check(abs(state%connection_residual_normalized) > 1e-7_dp, 'bin upper residual exposed')
  call evaluate_sheath_state(input, 'A', 0.5_dp, state, status, message, minimum_potential_v=-1e-100_dp)
  call check(status == SHEATH_OK .and. .not. state%admissible, 'sub-ulp bin depth remains nonconnecting')

  ! A steep unsmoothed source with a physical B root missed by coupled multistart.
  input = prescribed_field_input(branch='B', ion_density_m3=5e6_dp, ion_entry_speed_mps=4e5_dp, &
      electron_drift_mps=4e5_dp, electron_mass_kg=9.1093837139e-31_dp)
  input%electron_temperature_ev = 10.0_dp*1.160451812e4_dp*1.380649e-23_dp/qe
  call load_source('test/fixtures/prescribed_field_b_spectrum.csv')
  input%electric_field_v_m = 2.7514597094363297e-12_dp/eps0
  call evaluate_sheath_state(input, 'B', 3.0663883946402564_dp, state, status, message)
  call check(status == SHEATH_OK .and. state%admissible, 'independent B reference')
  call check(abs(state%electric_field_v_m - input%electric_field_v_m) < 1e-11_dp, 'reference matches field')
  previous(1) = prescribed_field_result(valid=.true., branch='B', boundary_potential_v=1.43306_dp, &
      ambient_electron_density_m3=4.488e6_dp)
  do pass = 0, 2
    if (pass == 0) then
      call solver%solve_prescribed_field_candidates(input, roots, status, message, diagnostics)
    else if (pass == 1) then
      call solver%solve_prescribed_field_candidates(input, roots, status, message, diagnostics, previous)
    else
      solver%search%method = 'bracket'
      solver%search%use_default_guesses = .false.
      call solver%solve_prescribed_field_candidates(input, roots, status, message, diagnostics)
    end if
    call check(status == SHEATH_OK, 'B field inversion without and with an unrelated seed')
    call check(any(abs(roots%boundary_potential_v - 3.0663883946402564_dp) < 1e-7_dp), 'steep B root recovered')
    call check(diagnostics%brackets(2) > 0, 'scalar search used')
    if (pass == 2) call check(any(roots%nonlinear_iterations > 0), 'scalar refinement count retained')
    do i = 1, size(roots)
      call evaluate_sheath_state(input, 'B', roots(i)%boundary_potential_v, state, status, message)
      call check(status == SHEATH_OK .and. state%admissible, 'every returned candidate rechecked')
      call check(abs(state%electric_field_v_m - input%electric_field_v_m) < 1e-8_dp, 'original field reproduced')
    end do
  end do
  ! C uses the same scalar algorithm and the negative electric-field convention.
  input = prescribed_field_input(branch='C', electron_drift_mps=0.0_dp)
  call evaluate_sheath_state(input, 'C', -1.0_dp, state, status, message)
  call check(status == SHEATH_OK .and. state%admissible, 'C trial')
  input%electric_field_v_m = state%electric_field_v_m
  call solver%solve_prescribed_field_candidates(input, roots, status, message, diagnostics)
  call check(status == SHEATH_OK, 'C bracket inversion')
  call check(any(abs(roots%boundary_potential_v + 1.0_dp) < 1e-7_dp), 'C root recovered')
  input%branch = 'A'
  call solver%solve_prescribed_field_candidates(input, roots, status, message)
  call check(status == SHEATH_INVALID_ARGUMENT, 'bracket requires explicit B/C')
  print *, 'Generic consumer states and difficult prescribed-field roots passed.'
contains
  subroutine load_source(path)
    character(len=*), intent(in) :: path
    integer :: unit, ios, n, j
    real(dp) :: low, high, value
    character(len=256) :: header
    if (allocated(edges)) deallocate (edges, flux)
    open (newunit=unit, file=path, status='old')
    read (unit, '(a)') header
    n = 0
    do
      read (unit, *, iostat=ios) low, high, value
      if (ios < 0) exit
      call check(ios == 0, 'fixture format')
      n = n + 1
    end do
    allocate (edges(n + 1), flux(n))
    rewind (unit)
    read (unit, '(a)') header
    do j = 1, n
      read (unit, *) edges(j), high, flux(j)
      edges(j + 1) = high
    end do
    close (unit)
    input%photoelectrons = binned_photoelectrons(edges, flux)
  end subroutine
  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, 'FAIL: ', label, ' ', trim(message), ' ', trim(state%physical_message)
      error stop 1
    end if
  end subroutine
end program
