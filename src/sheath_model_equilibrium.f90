! SPDX-License-Identifier: MIT
module sheath_model_equilibrium
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sheath_model_constants, only: dp, i32, pi, eps0, qe, electron_mass, proton_mass, lower_ascii
  use sheath_model_photoelectrons, only: maxwellian_photoelectrons
  use sheath_model_ions, only: ion_critical_potential, ion_density_ratio
  use sheath_model_state, only: zhao_plasma_input, prepare_plasma_params
  use sheath_model_core, only: zhao_params_type, &
      evaluate_zhao_density_hat, &
      evaluate_zhao_fluxes
  use sheath_model_search, only: sheath_search_options, sheath_search_diagnostics, valid_search_options
  use sheath_model_equilibrium_search, only: search_equilibrium_branch, equilibrium_residual
  use sheath_model_status, only: SHEATH_OK, SHEATH_INVALID_ARGUMENT, SHEATH_NUMERICAL_FAILURE, SHEATH_NO_PHYSICAL_SOLUTION

  implicit none

  private

  public :: zhao_equilibrium_input, fixed_entry_equilibrium_input, zhao_equilibrium_result, zhao_density_result
  public :: solve_equilibrium, evaluate_density, solve_profile
  public :: zhao_profile_options, zhao_profile_result

  type, abstract :: equilibrium_input
    type(sheath_search_options) :: search
    character(len=9) :: branch = 'auto'
  end type equilibrium_input

  !> Fixed normal plasma state and emitted source for the J=0 closure.
  !! No wind projection or Bohm-speed adjustment is applied. Plasma units are SI except temperatures [eV].
  type, extends(equilibrium_input) :: fixed_entry_equilibrium_input
    type(zhao_plasma_input) :: plasma = zhao_plasma_input(electron_drift_mps=0.0_dp)
  end type fixed_entry_equilibrium_input

  !> Plasma and illumination inputs for the J=0 closure; units are given by the component suffixes.
  !! branch selects A/B/C/auto; normal drift projects the wind speed along the surface normal.
  type, extends(equilibrium_input) :: zhao_equilibrium_input
    real(dp) :: sun_elevation_deg = 60.0_dp
    real(dp) :: ion_density_m3 = 8.7e6_dp
    real(dp) :: photoelectron_reference_density_m3 = 64.0e6_dp
    real(dp) :: electron_temperature_ev = 12.0_dp
    real(dp) :: ion_temperature_ev = 0.0_dp
    real(dp) :: ion_pressure_factor = 1.0_dp
    real(dp) :: photoelectron_temperature_ev = 2.2_dp
    real(dp) :: solar_wind_speed_mps = 468.0e3_dp
    real(dp) :: ion_mass_kg = proton_mass
    real(dp) :: electron_mass_kg = electron_mass
    character(len=6) :: electron_drift_mode = 'normal'
    character(len=6) :: ion_drift_mode = 'normal'
  end type zhao_equilibrium_input

  !> Accepted J=0 root, fluxes and current in SI units, with dimensionless residual_norm.
  !! valid marks success; ambient_electron_density_m3 is the Maxwellian normalization, not total upstream density.
  type :: zhao_equilibrium_result
    logical :: valid = .false.
    character(len=1) :: branch = ' '
    real(dp) :: surface_potential_v = 0.0_dp
    real(dp) :: minimum_potential_v = 0.0_dp
    real(dp) :: ambient_electron_density_m3 = 0.0_dp
    real(dp) :: debye_length_m = 0.0_dp
    real(dp) :: residual_norm = huge(1.0_dp) ! scaled neutrality/current/E^2 residual
    real(dp) :: electron_inward_flux_m2_s = 0.0_dp
    real(dp) :: ion_inward_flux_m2_s = 0.0_dp
    real(dp) :: photoelectron_escape_flux_m2_s = 0.0_dp
    real(dp) :: net_current_a_m2 = 0.0_dp ! Conventional electric current along +z
  end type zhao_equilibrium_result

  !> Local species number densities [m^-3] and net charge density [C/m^3].
  type :: zhao_density_result
    real(dp) :: ion_m3 = 0.0_dp
    real(dp) :: electron_free_m3 = 0.0_dp
    real(dp) :: electron_reflected_m3 = 0.0_dp
    real(dp) :: photoelectron_free_m3 = 0.0_dp
    real(dp) :: photoelectron_captured_m3 = 0.0_dp
    real(dp) :: charge_c_m3 = 0.0_dp
  end type zhao_density_result

  !> Profile sampling controls: points per monotonic segment, maximum height [m], and potential cutoff [V].
  type :: zhao_profile_options
    integer :: points_per_segment = 4000
    real(dp) :: max_distance_m = 100.0_dp
    real(dp) :: potential_cutoff_v = 2.2e-3_dp
  end type zhao_profile_options

  !> J=0 equilibrium and sampled height [m], potential [V], field [V/m], and local densities.
  !! All arrays share the same node order; turning_height_m is -1 for branches without an internal minimum.
  type :: zhao_profile_result
    type(zhao_equilibrium_result) :: equilibrium
    real(dp) :: turning_height_m = -1.0_dp ! A only; -1 means no internal minimum
    real(dp), allocatable :: z_m(:), potential_v(:), electric_field_v_m(:)
    type(zhao_density_result), allocatable :: density(:)
  end type zhao_profile_result

contains

  subroutine prepare_params(input, p, status, message)
    class(equilibrium_input), intent(in) :: input
    type(zhao_params_type), intent(out) :: p
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    p = zhao_params_type()
    status = SHEATH_INVALID_ARGUMENT
    message = 'branch must be auto, A, B, or C.'
    select case (trim(lower_ascii(input%branch)))
    case ('auto', 'a', 'b', 'c')
    case default
      return
    end select

    message = 'Invalid search options; bracket is available only for explicit B/C equilibrium branches.'
    if (.not. valid_search_options(input%search)) return
    if (trim(lower_ascii(input%search%method)) == 'bracket' .and. &
        trim(lower_ascii(input%branch)) /= 'b' .and. trim(lower_ascii(input%branch)) /= 'c') return
    select type (input)
    type is (fixed_entry_equilibrium_input)
      call prepare_plasma_params(input%plasma, p, status, message)
      p%search = input%search
      p%search%method = lower_ascii(p%search%method)
      return
    type is (zhao_equilibrium_input)
      message = 'Equilibrium inputs must be finite.'
      if (.not. all(ieee_is_finite([input%sun_elevation_deg, input%ion_density_m3, &
          input%photoelectron_reference_density_m3, input%electron_temperature_ev, &
          input%photoelectron_temperature_ev, input%solar_wind_speed_mps, input%ion_mass_kg, input%electron_mass_kg, &
          input%ion_temperature_ev, input%ion_pressure_factor]))) return

      message = 'Densities, temperatures, wind speed, and masses must be positive; elevation must be in [0,90].'
      if (min(input%ion_density_m3, input%photoelectron_reference_density_m3, input%electron_temperature_ev, &
          input%photoelectron_temperature_ev, input%solar_wind_speed_mps, &
          input%ion_mass_kg, input%electron_mass_kg) <= 0.0_dp) return
      if (input%sun_elevation_deg < 0.0_dp .or. input%sun_elevation_deg > 90.0_dp) return
      message = 'Ion temperature must be nonnegative and pressure factor positive.'
      if (input%ion_temperature_ev < 0.0_dp .or. input%ion_pressure_factor <= 0.0_dp) return

      message = 'Electron drift mode must be normal, full, or zero; ion mode normal or full.'
      if (input%electron_drift_mode /= 'normal' .and. input%electron_drift_mode /= 'full' .and. &
          input%electron_drift_mode /= 'zero') return
      if (input%ion_drift_mode /= 'normal' .and. input%ion_drift_mode /= 'full') return

      p%search = input%search
      p%search%method = lower_ascii(p%search%method)
      p%alpha_rad = input%sun_elevation_deg*pi/180.0_dp
      p%n_swi_inf_m3 = input%ion_density_m3
      p%density_scale_m3 = input%photoelectron_reference_density_m3
      p%emission_density_scale_m3 = p%density_scale_m3*sin(p%alpha_rad)
      p%t_swe_ev = input%electron_temperature_ev
      p%t_swi_ev = input%ion_temperature_ev
      p%ion_pressure_factor = input%ion_pressure_factor
      p%potential_scale_v = input%photoelectron_temperature_ev
      p%photoelectrons = maxwellian_photoelectrons(p%emission_density_scale_m3, input%photoelectron_temperature_ev)
      p%m_i_kg = input%ion_mass_kg
      p%m_e_kg = input%electron_mass_kg

      p%v_d_electron_mps = input%solar_wind_speed_mps
      p%v_d_ion_mps = input%solar_wind_speed_mps
      if (input%electron_drift_mode == 'normal') then
        p%v_d_electron_mps = p%v_d_electron_mps*sin(p%alpha_rad)
      end if
      if (input%electron_drift_mode == 'zero') then
        p%v_d_electron_mps = 0.0_dp
      end if
      if (input%ion_drift_mode == 'normal') then
        p%v_d_ion_mps = p%v_d_ion_mps*sin(p%alpha_rad)
      end if

      message = 'Zero normal ion drift is degenerate; use a positive elevation or explicit full ion drift.'
      if (p%v_d_ion_mps <= 0.0_dp) return

      p%v_swe_th_mps = sqrt(2.0_dp*qe*p%t_swe_ev/p%m_e_kg)
      p%velocity_scale_mps = sqrt(2.0_dp*qe*p%potential_scale_v/p%m_e_kg)
      p%cs_mps = sqrt(qe*p%t_swe_ev/p%m_i_kg)
      p%mach = p%v_d_ion_mps/p%cs_mps
      message = 'Ion entry speed must exceed the ion thermal sound speed; no presheath acceleration is applied.'
      if (.not. ieee_is_finite(ion_critical_potential(0.5_dp*p%t_swe_ev*p%mach**2, &
          p%ion_pressure_factor*p%t_swi_ev))) return
      p%u = p%v_d_electron_mps/p%v_swe_th_mps
      p%tau = p%t_swe_ev/p%potential_scale_v
      p%length_scale_m = sqrt(eps0*p%potential_scale_v/(p%density_scale_m3*qe))
      status = SHEATH_NUMERICAL_FAILURE
      message = 'Parameter normalization is non-finite or underflowed.'
      if (.not. all(ieee_is_finite([p%v_swe_th_mps, p%velocity_scale_mps, p%cs_mps, p%mach, p%u, p%tau, &
          p%length_scale_m]))) return
      if (min(p%v_swe_th_mps, p%velocity_scale_mps, p%cs_mps, p%mach, p%tau, p%length_scale_m) <= 0.0_dp) return

      status = SHEATH_OK
      message = ''
    class default
      message = 'Unsupported equilibrium input type.'
    end select
  end subroutine prepare_params

  !> Solve the J=0 closure for the plasma inputs and return a physically admissible equilibrium.
  !! branch='auto' returns the first admissible branch in the model's search order.
  !! status/message report the outcome; output%valid is false on failure. Physical outputs use SI units.
  subroutine solve_equilibrium(input, output, status, message, diagnostics, initial_guesses)
    class(equilibrium_input), intent(in) :: input
    type(zhao_equilibrium_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    type(zhao_params_type) :: p
    type(sheath_search_diagnostics), intent(out), optional :: diagnostics
    type(zhao_equilibrium_result), intent(in), optional :: initial_guesses(:)
    type(sheath_search_diagnostics) :: search
    real(dp), allocatable :: initial(:, :)
    real(dp) :: found(3)
    integer :: i, seed_count
    type(zhao_equilibrium_result) :: trial
    real(dp) :: phi0, phim, density, residual(3)
    real(dp) :: outward, returning
    character(len=1) :: order(3)
    integer :: attempt, count
    logical :: nonphysical, unresolved_search
    character(len=1) :: branch
    logical :: success

    output = zhao_equilibrium_result()
    search = sheath_search_diagnostics()
    if (present(diagnostics)) diagnostics = search
    call prepare_params(input, p, status, message)
    if (status /= SHEATH_OK) return

    order = ['A', 'B', 'C']
    select type (input)
    type is (zhao_equilibrium_input)
      if (input%sun_elevation_deg < 20.0_dp) order = ['C', 'A', 'B']
    end select
    count = 3
    if (trim(lower_ascii(input%branch)) /= 'auto') then
      order(1) = input%branch(1:1)
      count = 1
    end if
    nonphysical = .false.
    unresolved_search = .false.
    do attempt = 1, count
      branch = order(attempt)
      if (branch >= 'a' .and. branch <= 'c') branch = achar(iachar(branch) - 32)
      seed_count = 0
      if (allocated(initial)) deallocate (initial)
      allocate (initial(3, 0))
      if (present(initial_guesses)) then
        deallocate (initial)
        allocate (initial(3, size(initial_guesses)))
        do i = 1, size(initial_guesses)
          if (.not. initial_guesses(i)%valid .or. initial_guesses(i)%branch /= branch) cycle
          seed_count = seed_count + 1
          initial(:, seed_count) = [initial_guesses(i)%surface_potential_v, initial_guesses(i)%minimum_potential_v, &
              initial_guesses(i)%ambient_electron_density_m3]
        end do
      end if
      call search_equilibrium_branch(p, branch, initial(:, :seed_count), found, success, search)
      if (present(diagnostics)) diagnostics = search
      phi0 = found(1)
      phim = found(2)
      density = found(3)
      if (.not. success) then
        nonphysical = nonphysical .or. any(search%excluded) .or. any(search%rejected > 0)
        unresolved_search = unresolved_search .or. search%unconverged(index('ABC', branch)) > 0 .or. &
            search%profile_failures(index('ABC', branch)) > 0 .or. (.not. search%excluded(index('ABC', branch)) .and. &
            search%roots_found(index('ABC', branch)) == 0 .and. search%rejected(index('ABC', branch)) == 0)
        cycle
      end if
      exit
    end do
    if (.not. success) then
      status = SHEATH_NUMERICAL_FAILURE
      message = 'Equilibrium root search did not converge for the requested branch.'
      if (nonphysical .and. .not. unresolved_search) then
        status = SHEATH_NO_PHYSICAL_SOLUTION
        message = 'Requested branches or located roots are excluded by physical profile conditions.'
      end if
      return
    end if

    call equilibrium_residual(p, branch, [phi0, phim, density], residual)
    trial = zhao_equilibrium_result(.false., branch, phi0, phim, density, p%length_scale_m, &
        maxval(abs(residual)))
    call evaluate_zhao_fluxes(p, phi0, phim, density, trial%electron_inward_flux_m2_s, &
        trial%ion_inward_flux_m2_s, outward, trial%photoelectron_escape_flux_m2_s, returning)
    trial%net_current_a_m2 = qe*(trial%electron_inward_flux_m2_s - trial%ion_inward_flux_m2_s - &
        trial%photoelectron_escape_flux_m2_s)
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Equilibrium flux or current evaluation is non-finite.'
    if (.not. all(ieee_is_finite([trial%electron_inward_flux_m2_s, trial%ion_inward_flux_m2_s, &
        trial%photoelectron_escape_flux_m2_s, trial%net_current_a_m2]))) return

    trial%valid = .true.
    output = trial
    status = SHEATH_OK
    message = ''
  end subroutine solve_equilibrium

  !> Return species densities [m^-3] and charge density [C/m^3] at potential_v [V].
  !! Pass a valid equilibrium solution and the same input used to obtain it.
  !! Type A requires side='lower' (surface to minimum) or 'upper' (minimum to upstream); B/C may omit it.
  !! status/message report invalid potentials or failed density evaluations.
  subroutine evaluate_density( &
      input, solution, potential_v, &
      output, status, message, &
      side &
      )
    class(equilibrium_input), intent(in) :: input
    type(zhao_equilibrium_result), intent(in) :: solution
    real(dp), intent(in) :: potential_v
    type(zhao_density_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    character(len=*), intent(in), optional :: side

    type(zhao_params_type) :: p
    real(dp) :: d(5), upper
    character(len=9) :: region

    output = zhao_density_result()
    call prepare_params(input, p, status, message)
    if (status /= SHEATH_OK) return

    status = SHEATH_INVALID_ARGUMENT
    message = 'A valid equilibrium solution and finite potential are required.'
    if (.not. solution%valid) return
    if (.not. all(ieee_is_finite([potential_v, solution%surface_potential_v, solution%minimum_potential_v, &
        solution%ambient_electron_density_m3]))) return
    if (solution%ambient_electron_density_m3 <= 0.0_dp) return

    region = 'monotonic'
    upper = max(solution%surface_potential_v, 0.0_dp)

    select case (solution%branch)
    case ('A')
      message = 'Type A density requires side=lower or side=upper.'
      if (.not. present(side)) return

      region = lower_ascii(side)
      if (region /= 'lower' .and. region /= 'upper') return

      upper = solution%surface_potential_v
      if (region == 'upper') then
        upper = 0.0_dp
      end if
    case ('B', 'C')
    case default
      return
    end select

    message = 'Potential is outside the selected branch interval or blocks the ion fluid branch.'
    if (potential_v < solution%minimum_potential_v .or. potential_v > upper) return
    if (.not. ieee_is_finite(ion_density_ratio(potential_v, 0.5_dp*p%t_swe_ev*p%mach**2, &
        p%ion_pressure_factor*p%t_swi_ev))) return

    call evaluate_zhao_density_hat(p, solution%branch, region, potential_v/p%potential_scale_v, &
        solution%surface_potential_v/p%potential_scale_v, solution%minimum_potential_v/p%potential_scale_v, &
        solution%ambient_electron_density_m3/p%density_scale_m3, d(1), d(2), d(3), d(4), d(5))
    d = d*p%density_scale_m3
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Density evaluation is non-finite or negative.'
    if (.not. all(ieee_is_finite(d)) .or. any(d < 0.0_dp)) return

    output = zhao_density_result(d(1), d(2), d(3), d(4), d(5), qe*(d(1) - sum(d(2:5))))
    status = SHEATH_OK
    message = ''
  end subroutine evaluate_density

  !> Solve the J=0 equilibrium and reconstruct its 1D profile using the Poisson first integral.
  !! options controls sampling and truncation of the semi-infinite domain; no forced zero tail is attached.
  !! On SHEATH_OK, output contains allocated arrays in SI units; otherwise inspect status and message.
  subroutine solve_profile(input, options, output, status, message)
    class(equilibrium_input), intent(in) :: input
    type(zhao_profile_options), intent(in) :: options
    type(zhao_profile_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    type(zhao_params_type) :: p
    type(zhao_equilibrium_result) :: root
    type(zhao_profile_result) :: trial
    real(dp), allocatable :: phi(:), distance(:), rho(:), e2(:), z(:), v(:), e(:)
    real(dp) :: phi0, phim, cutoff, t, delta, turn, part, d(5)
    integer :: n, i, segment, total, kept
    character(len=9) :: side

    call prepare_params(input, p, status, message)
    if (status /= SHEATH_OK) return

    status = SHEATH_INVALID_ARGUMENT
    message = 'Profile requires at least 32 points per segment and finite positive distance/cutoff.'
    if (options%points_per_segment < 32) return
    if (.not. all(ieee_is_finite([options%max_distance_m, options%potential_cutoff_v]))) return
    if (min(options%max_distance_m, options%potential_cutoff_v) <= 0.0_dp) return

    call solve_equilibrium(input, root, status, message)
    if (status /= SHEATH_OK) return

    n = options%points_per_segment
    allocate (phi(n), distance(n), rho(n), e2(n))
    allocate (z(2*n), v(2*n), e(2*n))
    phi0 = root%surface_potential_v/p%potential_scale_v
    phim = root%minimum_potential_v/p%potential_scale_v
    cutoff = min(options%potential_cutoff_v/p%potential_scale_v, 0.5_dp*abs(phim))
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Profile first integral is non-finite or does not support a real electric field.'
    total = 0
    turn = 0.0_dp
    if (root%branch == 'A') then
      do segment = 1, 2
        side = 'lower'
        delta = phi0 - phim
        if (segment == 2) then
          side = 'upper'
          delta = -cutoff - phim
        end if
        do i = 1, n
          t = real(i - 1, dp)/real(n - 1, dp)
          phi(i) = phim + delta*t*t
          call density_at(phi(i), side, d)
          rho(i) = d(1) - sum(d(2:5))
        end do
        e2(1) = 0.0_dp
        distance(1) = 0.0_dp
        do i = 2, n
          part = phi(i) - phi(i - 1)
          e2(i) = e2(i - 1) - (rho(i) + rho(i - 1))*part
          if (.not. ieee_is_finite(e2(i)) .or. e2(i) <= 0.0_dp) return

          if (i == 2) then
            ! Integrate the 1/sqrt(phi-phi_min) endpoint analytically.
            distance(i) = 2.0_dp*part/sqrt(e2(i))
          else
            distance(i) = distance(i - 1) + 0.5_dp*part*(1.0_dp/sqrt(e2(i - 1)) + 1.0_dp/sqrt(e2(i)))
          end if
        end do
        if (segment == 1) then
          turn = distance(n)
          do i = 1, n
            z(i) = turn - distance(n + 1 - i)
            v(i) = phi(n + 1 - i)
            e(i) = sqrt(e2(n + 1 - i))
          end do
          total = n
        else
          z(n + 1:2*n - 1) = turn + distance(2:n)
          v(n + 1:2*n - 1) = phi(2:n)
          e(n + 1:2*n - 1) = -sqrt(e2(2:n))
          total = 2*n - 1
        end if
      end do
    else
      ! Integrate from the exact upstream potential, then omit that infinite endpoint.
      cutoff = min(options%potential_cutoff_v/p%potential_scale_v, 0.5_dp*abs(phi0))
      do i = 1, n
        t = real(i - 1, dp)/real(n - 1, dp)
        phi(i) = phi0*(1.0_dp - t)**2
        call density_at(phi(i), 'monotonic', d)
        rho(i) = d(1) - sum(d(2:5))
      end do
      e2(n) = 0.0_dp
      do i = n - 1, 1, -1
        e2(i) = e2(i + 1) + (rho(i + 1) + rho(i))*(phi(i + 1) - phi(i))
      end do
      do i = 1, n - 1
        if (abs(phi(i)) < cutoff) exit
        if (.not. ieee_is_finite(e2(i)) .or. e2(i) <= 0.0_dp) return

        z(i) = 0.0_dp
        if (i > 1) then
          z(i) = z(i - 1) + 0.5_dp*abs(phi(i) - phi(i - 1))* &
              (1.0_dp/sqrt(e2(i)) + 1.0_dp/sqrt(e2(i - 1)))
        end if
        v(i) = phi(i)
        e(i) = sign(sqrt(e2(i)), phi0)
        total = i
      end do
    end if
    z(1:total) = z(1:total)*p%length_scale_m
    v(1:total) = v(1:total)*p%potential_scale_v
    ! Preserve exact physical endpoints through the normalized-coordinate round trip.
    v(1) = root%surface_potential_v
    if (root%branch == 'A') then
      v(n) = root%minimum_potential_v
    end if
    e(1:total) = e(1:total)*p%potential_scale_v/p%length_scale_m
    if (.not. all(ieee_is_finite(z(1:total))) .or. .not. all(ieee_is_finite(e(1:total)))) return

    kept = count(z(1:total) <= options%max_distance_m)
    if (kept < 2) then
      status = SHEATH_INVALID_ARGUMENT
      message = 'max_distance_m contains fewer than two nodes; increase points_per_segment or distance.'
      return
    end if

    trial%equilibrium = root
    if (root%branch == 'A') then
      trial%turning_height_m = turn*p%length_scale_m
    end if
    trial%z_m = z(1:kept)
    trial%potential_v = v(1:kept)
    trial%electric_field_v_m = e(1:kept)
    allocate (trial%density(kept))
    do i = 1, kept
      side = 'monotonic'
      if (root%branch == 'A') then
        side = 'lower'
        if (i > n) then
          side = 'upper'
        end if
      end if
      call evaluate_density(input, root, v(i), trial%density(i), status, message, side)
      if (status /= SHEATH_OK) return
    end do
    output = trial
    status = SHEATH_OK
    message = ''

  contains

    subroutine density_at(phi_hat, region, values)
      real(dp), intent(in) :: phi_hat
      character(len=*), intent(in) :: region
      real(dp), intent(out) :: values(5)

      call evaluate_zhao_density_hat(p, root%branch, region, phi_hat, phi0, phim, &
          root%ambient_electron_density_m3/p%density_scale_m3, &
          values(1), values(2), values(3), values(4), values(5))
    end subroutine density_at

  end subroutine solve_profile

end module sheath_model_equilibrium
