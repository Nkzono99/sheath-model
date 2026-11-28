! SPDX-License-Identifier: MIT
module sheath_model_equilibrium
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sheath_model_constants, only: dp, i32, pi, electron_mass, proton_mass, lower_ascii
  use sheath_model_emission, only: maxwellian_emission
  use sheath_model_state, only: plasma_input, prepare_plasma_params
  use sheath_model_core, only: zhao_params_type, evaluate_zhao_fluxes
  use sheath_model_search, only: sheath_search_options, sheath_search_diagnostics, valid_search_options, &
      sheath_continuation_options, resolve_search_options
  use sheath_model_atlas, only: sheath_equilibrium_atlas
  use sheath_model_atlas_physics, only: equilibrium_key, atlas_equilibrium_seeds, continue_equilibrium_from_atlas
  use sheath_model_coordinates, only: encode_unknowns
  use sheath_model_admissibility, only: validate_zhao_profile
  use sheath_model_equilibrium_search, only: search_equilibrium_branch, equilibrium_residual
  use sheath_model_status, only: SHEATH_OK, SHEATH_INVALID_ARGUMENT, SHEATH_NUMERICAL_FAILURE, SHEATH_NO_PHYSICAL_SOLUTION

  implicit none

  private

  public :: sheath_equilibrium_input, zhao_equilibrium_input, fixed_entry_equilibrium_input, &
      sheath_equilibrium_result, sheath_density_result
  public :: solve_equilibrium, evaluate_density, build_profile
  public :: sheath_profile_options, sheath_profile_result
  public :: build_equilibrium_atlas, add_equilibrium_to_atlas, solve_equilibrium_candidates
  ! Package-internal: other modules derive the same parameters from an equilibrium input.
  public :: prepare_params

  type, abstract :: sheath_equilibrium_input
    character(len=9) :: branch = 'auto'
  end type sheath_equilibrium_input

  !> Fixed normal plasma state and emitted source for the J=0 closure.
  !! No wind projection or Bohm-speed adjustment is applied. Plasma units are SI except temperatures [eV].
  type, extends(sheath_equilibrium_input) :: fixed_entry_equilibrium_input
    type(plasma_input) :: plasma
  end type fixed_entry_equilibrium_input

  !> Plasma and illumination inputs for the J=0 closure; units are given by the component suffixes.
  !! branch selects A/B/C/auto; normal drift projects the wind speed along the surface normal.
  type, extends(sheath_equilibrium_input) :: zhao_equilibrium_input
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
  type :: sheath_equilibrium_result
    logical :: valid = .false.
    character(len=1) :: branch = ' '
    real(dp) :: surface_potential_v = 0.0_dp
    real(dp) :: minimum_potential_v = 0.0_dp
    real(dp) :: ambient_electron_density_m3 = 0.0_dp
    real(dp) :: debye_length_m = 0.0_dp
    real(dp) :: residual_norm = huge(1.0_dp) ! scaled neutrality/current/E^2 residual
    real(dp) :: electron_inward_flux_m2_s = 0.0_dp
    real(dp) :: ion_inward_flux_m2_s = 0.0_dp
    real(dp) :: emitted_escape_flux_m2_s = 0.0_dp
    real(dp) :: net_current_a_m2 = 0.0_dp ! Conventional electric current along +z
    !> Potential width next to upstream where E^2<0 [V]; nonzero only for drifting A/C roots accepted by
    !! upstream_band_tolerance or upstream_field_floor_v_m.
    real(dp) :: upstream_negative_band_v = 0.0_dp
    !> Field deficit sqrt(-min E^2) inside that band [V/m]; the external field the upstream plasma must supply.
    real(dp) :: upstream_field_deficit_v_m = 0.0_dp
  end type sheath_equilibrium_result

  !> Local species number densities [m^-3] and net charge density [C/m^3].
  type :: sheath_density_result
    real(dp) :: ion_m3 = 0.0_dp
    real(dp) :: electron_free_m3 = 0.0_dp
    real(dp) :: electron_reflected_m3 = 0.0_dp
    real(dp) :: emitted_free_m3 = 0.0_dp
    real(dp) :: emitted_captured_m3 = 0.0_dp
    real(dp) :: charge_c_m3 = 0.0_dp
  end type sheath_density_result

  !> Profile sampling controls: points per monotonic segment, maximum height [m], and potential cutoff [V].
  type :: sheath_profile_options
    integer :: points_per_segment = 4000
    real(dp) :: max_distance_m = 100.0_dp
    real(dp) :: potential_cutoff_v = 2.2e-3_dp
    !> With an accepted upstream E^2<0 band, stop this fraction of the remaining depth before the band.
    real(dp) :: upstream_band_margin = 0.05_dp
  end type sheath_profile_options

  !> J=0 equilibrium and sampled height [m], potential [V], field [V/m], and local densities.
  !! All arrays share the same node order; turning_height_m is -1 for branches without an internal minimum.
  type :: sheath_profile_result
    type(sheath_equilibrium_result) :: equilibrium
    real(dp) :: turning_height_m = -1.0_dp ! A only; -1 means no internal minimum
    real(dp), allocatable :: z_m(:), potential_v(:), electric_field_v_m(:)
    type(sheath_density_result), allocatable :: density(:)
  end type sheath_profile_result

  interface
    !> Return species densities [m^-3] and charge density [C/m^3] at potential_v [V].
    !! Pass a valid equilibrium solution and the same input used to obtain it.
    !! Type A requires side='lower' (surface to minimum) or 'upper' (minimum to upstream); B/C may omit it.
    !! status/message report invalid potentials or failed density evaluations.
    module subroutine evaluate_density( &
        input, solution, potential_v, &
        output, status, message, &
        side &
        )
      class(sheath_equilibrium_input), intent(in) :: input
      type(sheath_equilibrium_result), intent(in) :: solution
      real(dp), intent(in) :: potential_v
      type(sheath_density_result), intent(out) :: output
      integer(i32), intent(out) :: status
      character(len=*), intent(out) :: message
      character(len=*), intent(in), optional :: side
    end subroutine evaluate_density

    !> Reconstruct an existing J=0 root using the Poisson first integral, without root search.
    !! options controls sampling and truncation of the semi-infinite domain; no forced zero tail is attached.
    !! On SHEATH_OK, output contains allocated arrays in SI units; otherwise inspect status and message.
    module subroutine build_profile(input, root, options, output, status, message, residual_tolerance, upstream_band_tolerance, &
        upstream_field_floor_v_m)
      class(sheath_equilibrium_input), intent(in) :: input
      type(sheath_profile_options), intent(in) :: options
      type(sheath_profile_result), intent(out) :: output
      integer(i32), intent(out) :: status
      character(len=*), intent(out) :: message
      real(dp), intent(in) :: residual_tolerance
      real(dp), intent(in), optional :: upstream_band_tolerance ! see sheath_search_options; default 0
      real(dp), intent(in), optional :: upstream_field_floor_v_m ! see sheath_search_options; default 0
      type(sheath_equilibrium_result), intent(in) :: root
    end subroutine build_profile
  end interface

contains

  subroutine prepare_params(input, p, status, message, search_options, continuation_options)
    class(sheath_equilibrium_input), intent(in) :: input
    type(zhao_params_type), intent(out) :: p
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(sheath_search_options), intent(in), optional :: search_options
    type(sheath_continuation_options), intent(in), optional :: continuation_options
    type(sheath_search_options) :: search
    type(sheath_continuation_options) :: continuation
    character(len=len(input%branch)) :: branch

    branch = lower_ascii(input%branch)
    status = SHEATH_INVALID_ARGUMENT
    message = 'branch must be auto, A, B, or C.'
    select case (branch)
    case ('auto', 'a', 'b', 'c')
    case default
      return
    end select

    message = 'Invalid search options; bracket is available only for explicit B/C equilibrium branches.'
    if (.not. resolve_search_options(branch, search, continuation, search_options, continuation_options)) return

    select type (input)
    type is (fixed_entry_equilibrium_input)
      call prepare_plasma_params(input%plasma, p, status, message)
    type is (zhao_equilibrium_input)
      call prepare_solar_params(input, p, status, message)
    class default
      message = 'Unsupported equilibrium input type.'
      return
    end select
    p%search = search
    p%continuation = continuation
  end subroutine prepare_params

  !> Project illumination and wind onto the physical plasma input used by the other closures.
  subroutine prepare_solar_params(input, p, status, message)
    type(zhao_equilibrium_input), intent(in) :: input
    type(zhao_params_type), intent(out) :: p
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(plasma_input) :: plasma
    real(dp) :: sine_elevation

    status = SHEATH_INVALID_ARGUMENT
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

    sine_elevation = sin(input%sun_elevation_deg*pi/180.0_dp)
    plasma = plasma_input(ion_density_m3=input%ion_density_m3, electron_temperature_ev=input%electron_temperature_ev, &
        ion_temperature_ev=input%ion_temperature_ev, ion_pressure_factor=input%ion_pressure_factor, &
        electron_drift_mps=input%solar_wind_speed_mps, ion_entry_speed_mps=input%solar_wind_speed_mps, &
        ion_mass_kg=input%ion_mass_kg, electron_mass_kg=input%electron_mass_kg, &
        emission=maxwellian_emission(input%photoelectron_reference_density_m3*sine_elevation, &
        input%photoelectron_temperature_ev))
    if (input%electron_drift_mode == 'normal') plasma%electron_drift_mps = plasma%electron_drift_mps*sine_elevation
    if (input%electron_drift_mode == 'zero') plasma%electron_drift_mps = 0.0_dp
    if (input%ion_drift_mode == 'normal') plasma%ion_entry_speed_mps = plasma%ion_entry_speed_mps*sine_elevation
    message = 'Zero normal ion drift is degenerate; use a positive elevation or explicit full ion drift.'
    if (plasma%ion_entry_speed_mps <= 0.0_dp) return
    call prepare_plasma_params(plasma, p, status, message, density_scale_m3=input%photoelectron_reference_density_m3)
  end subroutine prepare_solar_params

  !> Solve the J=0 closure for the plasma inputs and return a physically admissible equilibrium.
  !! branch='auto' returns the first admissible branch in the model's search order.
  !! status/message report the outcome; output%valid is false on failure. Physical outputs use SI units.
  subroutine solve_equilibrium(input, output, status, message, diagnostics, initial_guesses, atlas, &
      search_options, continuation_options)
    type(sheath_search_options), intent(in), optional :: search_options
    type(sheath_continuation_options), intent(in), optional :: continuation_options
    class(sheath_equilibrium_input), intent(in) :: input
    type(sheath_equilibrium_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    type(zhao_params_type) :: p
    type(sheath_search_diagnostics), intent(out), optional :: diagnostics
    type(sheath_equilibrium_result), intent(in), optional :: initial_guesses(:)
    type(sheath_equilibrium_atlas), intent(in), optional :: atlas
    type(sheath_search_diagnostics) :: search
    real(dp), allocatable :: initial(:, :)
    real(dp) :: found(3)
    integer :: k, atlas_count
    character(len=1) :: order(3)
    integer :: attempt, count
    logical :: nonphysical, unresolved_search
    character(len=1) :: branch
    logical :: success

    output = sheath_equilibrium_result()
    search = sheath_search_diagnostics()
    if (present(diagnostics)) diagnostics = search
    call prepare_params(input, p, status, message, &
        search_options=search_options, continuation_options=continuation_options)
    if (status /= SHEATH_OK) return
    if (present(atlas)) then
      if (.not. atlas%valid()) then
        status = SHEATH_INVALID_ARGUMENT
        message = 'Invalid atlas or continuation options.'
        return
      end if
    end if

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
      k = index('ABC', branch)
      call equilibrium_seeds(p, branch, initial, atlas_count, initial_guesses, atlas)
      call search_equilibrium_branch(p, branch, initial, found, success, search, atlas_count)
      if (.not. success .and. present(atlas) .and. .not. search%excluded(k) .and. p%search%method /= 'bracket') then
        call continue_equilibrium_from_atlas(p, branch, atlas, search, found, success)
      end if
      if (present(diagnostics)) diagnostics = search
      if (.not. success) then
        nonphysical = nonphysical .or. any(search%excluded) .or. any(search%rejected > 0)
        unresolved_search = unresolved_search .or. search%unconverged(k) > 0 .or. search%profile_failures(k) > 0 .or. &
            (.not. search%excluded(k) .and. search%roots_found(k) == 0 .and. search%rejected(k) == 0)
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

    call make_equilibrium_result(p, branch, found, output, status, message)
  end subroutine solve_equilibrium

  !> Explicit guesses keep their input order and precede all atlas predictions.
  subroutine equilibrium_seeds(p, branch, seeds, atlas_count, guesses, atlas)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), allocatable, intent(out) :: seeds(:, :)
    integer, intent(out) :: atlas_count
    type(sheath_equilibrium_result), intent(in), optional :: guesses(:)
    type(sheath_equilibrium_atlas), intent(in), optional :: atlas
    real(dp), allocatable :: predictions(:, :)
    integer :: explicit_count, i, next

    explicit_count = 0
    if (present(guesses)) explicit_count = count(guesses%valid .and. guesses%branch == branch)
    atlas_count = 0
    if (present(atlas)) then
      call atlas_equilibrium_seeds(p, branch, atlas, predictions)
      atlas_count = size(predictions, 2)
    end if
    allocate (seeds(3, explicit_count + atlas_count))
    if (present(guesses)) then
      next = 0
      do i = 1, size(guesses)
        if (.not. guesses(i)%valid .or. guesses(i)%branch /= branch) cycle
        next = next + 1
        seeds(:, next) = [guesses(i)%surface_potential_v, guesses(i)%minimum_potential_v, &
            guesses(i)%ambient_electron_density_m3]
      end do
    end if
    if (atlas_count > 0) seeds(:, explicit_count + 1:) = predictions
  end subroutine equilibrium_seeds

  subroutine make_equilibrium_result(p, branch, physical, output, status, message)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: physical(3)
    type(sheath_equilibrium_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(sheath_equilibrium_result) :: trial
    real(dp) :: residual(3), outward, returning, minimum_e2, boundary_e2, band, deficit
    output = sheath_equilibrium_result()
    call equilibrium_residual(p, branch, physical, residual)
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Original equilibrium residual exceeds tolerance.'
    if (.not. all(ieee_is_finite(residual))) return
    if (maxval(abs(residual)) > p%search%residual_tolerance) return
    trial = sheath_equilibrium_result(.false., branch, physical(1), physical(2), physical(3), p%length_scale_m, &
        maxval(abs(residual)))
    call evaluate_zhao_fluxes(p, physical(1), physical(2), physical(3), trial%electron_inward_flux_m2_s, &
        trial%ion_inward_flux_m2_s, outward, trial%emitted_escape_flux_m2_s, returning, trial%net_current_a_m2)
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Equilibrium flux or current evaluation is non-finite.'
    if (.not. all(ieee_is_finite([trial%electron_inward_flux_m2_s, trial%ion_inward_flux_m2_s, &
        trial%emitted_escape_flux_m2_s, trial%net_current_a_m2]))) return

    call validate_zhao_profile(p, branch, physical(1)/p%potential_scale_v, physical(2)/p%potential_scale_v, &
        physical(3)/p%density_scale_m3, minimum_e2, boundary_e2, status, message, band, deficit)
    if (status /= SHEATH_OK) return
    trial%upstream_negative_band_v = band*p%potential_scale_v
    trial%upstream_field_deficit_v_m = deficit

    trial%valid = .true.
    output = trial
    status = SHEATH_OK
    message = ''
  end subroutine make_equilibrium_result

  !> Enumerate located physical roots with optional deflation; no completeness claim.
  !! max_roots is per A/B/C type, and initial_guesses/atlas supplement default seeds.
  subroutine solve_equilibrium_candidates(input, candidates, status, message, diagnostics, &
      initial_guesses, atlas, deflation, max_roots, search_options, continuation_options)
    type(sheath_search_options), intent(in), optional :: search_options
    type(sheath_continuation_options), intent(in), optional :: continuation_options
    class(sheath_equilibrium_input), intent(in) :: input
    type(sheath_equilibrium_result), allocatable, intent(out) :: candidates(:)
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(sheath_search_diagnostics), intent(out), optional :: diagnostics
    type(sheath_equilibrium_result), intent(in), optional :: initial_guesses(:)
    type(sheath_equilibrium_atlas), intent(in), optional :: atlas
    logical, intent(in), optional :: deflation
    integer, intent(in), optional :: max_roots
    type(sheath_search_diagnostics) :: search
    type(zhao_params_type) :: p
    type(sheath_equilibrium_result) :: root
    real(dp), allocatable :: initial(:, :), physical_roots(:, :)
    real(dp) :: first(3)
    integer :: k, i, atlas_count, limit
    logical :: success, use_deflation
    character(len=1) :: branch
    character(len=1), parameter :: branches(3) = ['A', 'B', 'C']
    allocate (candidates(0))
    search = sheath_search_diagnostics()
    if (present(diagnostics)) diagnostics = search
    call prepare_params(input, p, status, message, &
        search_options=search_options, continuation_options=continuation_options)
    if (status /= SHEATH_OK) return
    limit = 16
    if (present(max_roots)) limit = max_roots
    status = SHEATH_INVALID_ARGUMENT
    message = 'max_roots must be positive and atlas controls valid.'
    if (limit < 1) return
    if (present(atlas)) then
      if (.not. atlas%valid()) return
    end if
    use_deflation = .true.
    if (present(deflation)) use_deflation = deflation
    do k = 1, 3
      branch = branches(k)
      if (trim(lower_ascii(input%branch)) /= 'auto' .and. &
          trim(lower_ascii(input%branch)) /= lower_ascii(branch)) cycle
      call equilibrium_seeds(p, branch, initial, atlas_count, initial_guesses, atlas)
      call search_equilibrium_branch(p, branch, initial, first, success, search, atlas_count, &
          physical_roots, use_deflation, limit)
      if (.not. success .and. present(atlas) .and. .not. search%excluded(k) .and. p%search%method /= 'bracket') then
        call continue_equilibrium_from_atlas(p, branch, atlas, search, first, success)
        if (success) physical_roots = reshape(first, [3, 1])
      end if
      do i = 1, size(physical_roots, 2)
        call make_equilibrium_result(p, branch, physical_roots(:, i), root, status, message)
        if (status /= SHEATH_OK) then
          if (present(diagnostics)) diagnostics = search
          return
        end if
        candidates = [candidates, root]
      end do
    end do
    if (present(diagnostics)) diagnostics = search
    status = SHEATH_OK
    message = 'Located roots do not establish completeness or dynamical stability.'
    if (size(candidates) == 0) then
      status = SHEATH_NUMERICAL_FAILURE
      message = 'Finite search found no admissible roots.'
      if (all(search%excluded .or. .not. search%searched)) then
        status = SHEATH_NO_PHYSICAL_SOLUTION
        message = 'All requested branches are analytically excluded.'
      end if
    end if
  end subroutine solve_equilibrium_candidates

  !> Store only an independently checked root, in dimensionless coordinates.
  subroutine add_equilibrium_to_atlas(input, root, atlas, status, message, component, search_options, continuation_options)
    type(sheath_search_options), intent(in), optional :: search_options
    type(sheath_continuation_options), intent(in), optional :: continuation_options
    class(sheath_equilibrium_input), intent(in) :: input
    type(sheath_equilibrium_result), intent(in) :: root
    type(sheath_equilibrium_atlas), intent(inout) :: atlas
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    integer, intent(in), optional :: component
    type(zhao_params_type) :: p
    real(dp) :: key(6), y(3), raw(3), physical(3), minimum_e2, boundary_e2
    real(dp), allocatable :: shape(:)
    logical :: valid
    call prepare_params(input, p, status, message, &
        search_options=search_options, continuation_options=continuation_options)
    if (status /= SHEATH_OK) return
    status = SHEATH_INVALID_ARGUMENT
    message = 'Atlas requires valid controls and a converged, physical A/B/C root.'
    if (.not. atlas%valid() .or. .not. root%valid .or. index('ABC', root%branch) == 0) return
    if (present(component)) then
      if (component < 1) return
    end if
    physical = [root%surface_potential_v, root%minimum_potential_v, root%ambient_electron_density_m3]
    if (root%branch == 'B' .and. physical(2) /= 0.0_dp) return
    if (root%branch == 'C' .and. physical(2) /= physical(1)) return
    call encode_unknowns(p, root%branch, physical(1), physical(2), physical(3), y, valid)
    if (.not. valid) return
    call equilibrium_residual(p, root%branch, physical, raw)
    if (.not. all(ieee_is_finite(raw))) return
    if (maxval(abs(raw)) > p%search%residual_tolerance) return
    call validate_zhao_profile(p, root%branch, physical(1)/p%potential_scale_v, physical(2)/p%potential_scale_v, &
        physical(3)/p%density_scale_m3, minimum_e2, boundary_e2, status, message)
    if (status /= SHEATH_OK) return
    call equilibrium_key(p, key, shape)
    call atlas%insert(key, shape, root%branch, y, component)
    status = SHEATH_OK
    message = ''
  end subroutine add_equilibrium_to_atlas

  !> Build a table from a sweep, then retry unresolved points using later neighbors.
  !! report is ordered (A/B/C,input); unrequested branches have INVALID_ARGUMENT.
  subroutine build_equilibrium_atlas(inputs, atlas, status, message, report, deflation, search_options, continuation_options)
    type(sheath_search_options), intent(in), optional :: search_options
    type(sheath_continuation_options), intent(in), optional :: continuation_options
    class(sheath_equilibrium_input), intent(in) :: inputs(:)
    type(sheath_equilibrium_atlas), intent(inout) :: atlas
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    integer(i32), allocatable, intent(out), optional :: report(:, :)
    logical, intent(in), optional :: deflation
    class(sheath_equilibrium_input), allocatable :: current
    type(sheath_equilibrium_result) :: root
    type(sheath_equilibrium_result), allocatable :: candidates(:)
    type(zhao_params_type) :: p
    integer(i32) :: outcomes(3, size(inputs)), local_status
    integer :: pass, i, k, j, size_before
    logical :: discover
    logical :: excluded(3, size(inputs))
    type(sheath_search_diagnostics) :: diagnostics
    character(len=1), parameter :: branches(3) = ['A', 'B', 'C']
    outcomes = SHEATH_INVALID_ARGUMENT
    excluded = .false.
    status = SHEATH_INVALID_ARGUMENT
    message = 'Invalid atlas controls.'
    if (.not. atlas%valid()) return
    discover = .false.
    if (present(deflation)) discover = deflation
    do pass = 1, 3*size(inputs) + 1
      size_before = atlas%size()
      do i = 1, size(inputs)
        if (allocated(current)) deallocate (current)
        allocate (current, source=inputs(i))
        call prepare_params(current, p, local_status, message, &
            search_options=search_options, continuation_options=continuation_options)
        if (local_status /= SHEATH_OK) then
          status = local_status
          if (present(report)) report = outcomes
          return
        end if
        do k = 1, 3
          if (trim(lower_ascii(inputs(i)%branch)) /= 'auto' .and. &
              trim(lower_ascii(inputs(i)%branch)) /= lower_ascii(branches(k))) cycle
          if (pass > 1) then
            if (outcomes(k, i) == SHEATH_OK .or. excluded(k, i)) cycle
          end if
          current%branch = branches(k)
          if (discover) then
            call solve_equilibrium_candidates(current, candidates, local_status, message, diagnostics, &
                atlas=atlas, deflation=.true., &
                search_options=search_options, continuation_options=continuation_options)
          else
            call solve_equilibrium(current, root, local_status, message, diagnostics, atlas=atlas, &
                search_options=search_options, continuation_options=continuation_options)
            if (allocated(candidates)) deallocate (candidates)
            allocate (candidates(0))
            if (local_status == SHEATH_OK) candidates = [root]
          end if
          outcomes(k, i) = local_status
          excluded(k, i) = diagnostics%excluded(k)
          if (local_status == SHEATH_INVALID_ARGUMENT) then
            status = local_status
            if (present(report)) report = outcomes
            return
          end if
          if (local_status /= SHEATH_OK) cycle
          do j = 1, size(candidates)
            call add_equilibrium_to_atlas(current, candidates(j), atlas, local_status, message, &
                search_options=search_options, continuation_options=continuation_options)
            if (local_status /= SHEATH_OK) then
              status = local_status
              if (present(report)) report = outcomes
              return
            end if
          end do
        end do
      end do
      if (.not. any((outcomes == SHEATH_NUMERICAL_FAILURE .or. outcomes == SHEATH_NO_PHYSICAL_SOLUTION) &
          .and. .not. excluded) .or. atlas%size() == size_before) exit
    end do
    if (present(report)) report = outcomes
    status = SHEATH_OK
    message = 'Atlas built; unresolved points do not establish absence of solutions.'
  end subroutine build_equilibrium_atlas

end module sheath_model_equilibrium
