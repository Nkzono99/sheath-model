! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone modules; status-returning public facade in sheath_model.
!> Enumerate Zhao roots and select a physical solution.
!! 数値解法と物理式は numerics / physics に委譲し、選択順と縮退判定をここに集める。
submodule(sheath_model_field) sheath_model_field_roots
  use sheath_model_scalar_search, only: find_scalar_roots
  use sheath_model_core, only: integrate_zhao_rho
  use sheath_model_ions, only: ion_critical_potential
  implicit none

  real(dp), parameter :: root_cluster_tolerance = 1.0e-6_dp

contains

  module subroutine solve_field_root( &
      model, params, interface_field_v_m, &
      root, status, message, &
      diagnostics, initial_guesses, atlas, deflation, max_roots &
      )
    character(len=*), intent(in) :: model
    type(zhao_params_type), intent(in) :: params
    real(dp), intent(in) :: interface_field_v_m
    type(zhao_field_root), intent(out) :: root
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(sheath_search_diagnostics), intent(out) :: diagnostics
    type(prescribed_field_result), intent(in), optional :: initial_guesses(:)
    type(sheath_field_atlas), intent(in), optional :: atlas
    logical, intent(in), optional :: deflation
    integer, intent(in), optional :: max_roots

    type(zhao_field_root), allocatable :: roots(:)

    root = zhao_field_root()
    call find_field_roots(model, params, interface_field_v_m, roots, status, message, diagnostics, &
        initial_guesses, atlas, deflation, max_roots)
    if (status /= SHEATH_OK) return

    if (size(roots) == 1) then
      root = roots(1)
    else
      status = SHEATH_AMBIGUOUS_SOLUTION
      message = 'Multiple admissible roots found; use solve_prescribed_field_candidates to inspect them.'
    end if
  end subroutine solve_field_root

  module subroutine find_field_roots( &
      model, params, interface_field_v_m, &
      roots, status, message, &
      diagnostics, initial_guesses, atlas, deflation, max_roots &
      )
    character(len=*), intent(in) :: model
    type(zhao_params_type), intent(in) :: params
    real(dp), intent(in) :: interface_field_v_m
    type(zhao_field_root), allocatable, intent(out) :: roots(:)
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(sheath_search_diagnostics), intent(out) :: diagnostics
    type(prescribed_field_result), intent(in), optional :: initial_guesses(:)
    type(sheath_field_atlas), intent(in), optional :: atlas
    logical, intent(in), optional :: deflation
    integer, intent(in), optional :: max_roots

    character(len=1) :: order(3)
    type(zhao_field_root), allocatable :: found(:), candidates(:)
    type(zhao_field_root) :: flat
    real(dp) :: field_scale, target, density
    integer :: branch_count, i, j, k, count, n, capacity, branch_index
    logical :: duplicate, unresolved

    diagnostics = sheath_search_diagnostics()
    field_scale = params%potential_scale_v/params%length_scale_m
    target = interface_field_v_m/field_scale
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Invalid field normalization.'
    if (.not. ieee_is_finite(target) .or. field_scale <= 0.0_dp) return

    call field_branch_order(model, target, order, branch_count, status, message)
    if (status /= SHEATH_OK) return

    capacity = 16
    if (present(max_roots)) capacity = max_roots
    allocate (found(3*capacity + 1))
    n = 0

    ! The flat state is one candidate, never a shortcut around the non-flat search.
    if (interface_field_v_m == 0.0_dp .and. (model == 'auto' .or. model == 'b')) then
      density = neutral_electron_density(params, 'B', 0.0_dp, 0.0_dp)
      if (density > 0.0_dp .and. ieee_is_finite(density)) then
        flat = zhao_field_root()
        flat%branch = 'B'
        flat%ambient_electron_density_m3 = density
        flat%residual_norm = 0.0_dp
        flat%minimum_field_squared_hat = 0.0_dp
        n = 1
        found(n) = flat
      end if
    end if

    do i = 1, branch_count
      call collect_field_branch_roots(params, order(i), target, candidates, count, diagnostics, &
          initial_guesses, atlas, deflation, capacity)
      do j = 1, count
        duplicate = .false.
        do k = 1, n
          duplicate = field_roots_equivalent(params, candidates(j), found(k))
          if (interface_field_v_m == 0.0_dp .and. found(k)%phi0_v == 0.0_dp) then
            duplicate = duplicate .or. max(abs(candidates(j)%phi0_v), abs(candidates(j)%phi_m_v)) &
                < 1e-4_dp*params%potential_scale_v
          end if
          if (duplicate) exit
        end do
        if (duplicate) cycle
        n = n + 1
        found(n) = candidates(j)
      end do
    end do

    do i = 1, n
      branch_index = index('ABC', found(i)%branch)
      diagnostics%roots_found(branch_index) = diagnostics%roots_found(branch_index) + 1
    end do
    unresolved = any(diagnostics%unconverged > 0 .or. diagnostics%profile_failures > 0 .or. &
        (diagnostics%searched .and. .not. diagnostics%excluded .and. diagnostics%starts == 0))
    if (n == 0) then
      if (unresolved) then
        status = SHEATH_NUMERICAL_FAILURE
        message = 'No admissible root found; some starts or profile evaluations remain unresolved.'
      else
        status = SHEATH_NO_PHYSICAL_SOLUTION
        message = 'All searched branches or converged candidates were excluded by physical conditions.'
      end if
      return
    end if

    roots = found(:n)
    status = SHEATH_OK
    message = ''
    if (unresolved) then
      message = 'Admissible candidates found; some starts or profile evaluations remain unresolved.'
    end if
  end subroutine find_field_roots

  subroutine field_branch_order(model, target_field_hat, order, count, status, message)
    character(len=*), intent(in) :: model
    real(dp), intent(in) :: target_field_hat
    character(len=1), intent(out) :: order(3)
    integer, intent(out) :: count
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    order = ' '
    count = 0
    status = SHEATH_OK
    message = ''

    select case (trim(model))
    case ('a')
      order(1) = 'A'
      count = 1
    case ('b')
      order(1) = 'B'
      count = 1
    case ('c')
      order(1) = 'C'
      count = 1
    case ('auto')
      if (target_field_hat >= 0.0_dp) then
        order = ['A', 'B', 'C']
      else
        order = ['C', 'A', 'B']
      end if
      count = 3
    case default
      status = SHEATH_INVALID_ARGUMENT
      message = 'unknown prescribed-field Zhao branch.'
    end select
  end subroutine field_branch_order

  subroutine collect_field_branch_roots( &
      params, branch, target_field_hat, &
      unique_roots, unique_count, &
      diagnostics, initial_guesses, atlas, deflation, max_roots &
      )
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: target_field_hat
    type(zhao_field_root), allocatable, intent(out) :: unique_roots(:)
    integer, intent(out) :: unique_count
    type(sheath_search_diagnostics), intent(inout) :: diagnostics
    type(prescribed_field_result), intent(in), optional :: initial_guesses(:)
    type(sheath_field_atlas), intent(in), optional :: atlas
    logical, intent(in), optional :: deflation
    integer, intent(in) :: max_roots

    real(dp) :: defaults(3, default_field_starts), y(3), norm, encoded(3), key(7), raw(3)
    real(dp), allocatable :: guesses(:, :), predictions(:, :), shape(:), algebraic(:, :)
    logical, allocatable :: atlas_flags(:)
    integer, allocatable :: origins(:), root_iterations(:)
    type(zhao_field_root), allocatable :: continued(:)
    type(zhao_field_root) :: candidate_root
    integer :: guess_count, default_count, guess_index, iterations, evaluations, lm_steps, root_index, k, capacity
    integer(i32) :: profile_status
    logical :: success, compatible, duplicate_root, use_deflation
    character(len=512) :: profile_message

    k = index('ABC', branch)
    unique_count = 0
    diagnostics%searched(k) = .true.
    compatible = (branch == 'C' .and. target_field_hat <= 0.0_dp) .or. &
        ((branch == 'A' .or. branch == 'B') .and. target_field_hat >= 0.0_dp)
    if (.not. compatible .or. ((branch == 'A' .or. branch == 'C') .and. params%u > 0.0_dp)) then
      diagnostics%excluded(k) = .true.
      return
    end if

    capacity = default_field_starts
    if (present(initial_guesses)) then
      capacity = capacity + size(initial_guesses)
    end if
    allocate (predictions(3, 0))
    if (present(atlas)) then
      call field_atlas_key(params, target_field_hat, key, shape)
      call atlas%predictions(key, shape, branch, predictions)
      capacity = capacity + size(predictions, 2)
    end if
    allocate (guesses(3, capacity), unique_roots(max_roots), atlas_flags(capacity))
    atlas_flags = .false.
    guess_count = 0

    ! Nearby physical solutions supplement the independent starts; they do not
    ! select a preferred root or change the specified plasma/boundary conditions.
    if (present(initial_guesses)) then
      do guess_index = 1, size(initial_guesses)
        if (.not. initial_guesses(guess_index)%valid .or. initial_guesses(guess_index)%branch /= branch) cycle
        call encode_unknowns(params, branch, initial_guesses(guess_index)%boundary_potential_v, &
            initial_guesses(guess_index)%minimum_potential_v, &
            initial_guesses(guess_index)%ambient_electron_density_m3, encoded, success)
        if (.not. success) cycle
        guess_count = guess_count + 1
        guesses(:, guess_count) = encoded
      end do
    end if

    guesses(:, guess_count + 1:guess_count + size(predictions, 2)) = predictions
    atlas_flags(guess_count + 1:guess_count + size(predictions, 2)) = .true.
    guess_count = guess_count + size(predictions, 2)

    default_count = 0
    if (params%search%use_default_guesses) call make_branch_guesses(params, branch, target_field_hat, defaults, default_count)
    guesses(:, guess_count + 1:guess_count + default_count) = defaults(:, :default_count)
    guess_count = guess_count + default_count

    use_deflation = .false.
    if (present(deflation)) use_deflation = deflation
    if (branch /= 'A' .and. (params%search%method == 'auto' .or. params%search%method == 'bracket')) &
        call collect_scalar_roots()
    if (params%search%method == 'bracket') return
    if (use_deflation .and. unique_count < max_roots) then
      call find_guarded_roots(merge(3, 2, branch == 'A'), field_residual, guesses(:merge(3, 2, branch == 'A'), :guess_count), &
          params%search, diagnostics, k, algebraic, max_roots, .true., &
          atlas_flags=atlas_flags(:guess_count), origins=origins, root_iterations=root_iterations)
      do guess_index = 1, size(algebraic, 2)
        y = 0.0_dp
        y(:size(algebraic, 1)) = algebraic(:, guess_index)
        call evaluate_charge_residual(params, branch, target_field_hat, y, raw, success)
        if (.not. success) cycle
        iterations = root_iterations(guess_index)
        call accept_candidate(y, iterations, atlas_flags(origins(guess_index)))
      end do
    else if (unique_count < max_roots) then

      do guess_index = 1, min(guess_count, params%search%max_starts)
        if (unique_count >= max_roots) exit
        diagnostics%starts(k) = diagnostics%starts(k) + 1
        if (atlas_flags(guess_index)) diagnostics%atlas_starts(k) = diagnostics%atlas_starts(k) + 1
        call solve_field_branch( &
            params, branch, target_field_hat, &
            guesses(:, guess_index), y, &
            norm, iterations, &
            success, evaluations, lm_steps &
            )
        diagnostics%evaluations(k) = diagnostics%evaluations(k) + evaluations
        diagnostics%iterations(k) = diagnostics%iterations(k) + iterations
        diagnostics%lm_steps(k) = diagnostics%lm_steps(k) + lm_steps
        diagnostics%best_residual(k) = min(diagnostics%best_residual(k), norm)
        if (.not. success) then
          diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
          cycle
        end if

        call accept_candidate(y, iterations, atlas_flags(guess_index))
      end do
    end if

    if (present(atlas) .and. unique_count < max_roots) then
      call continue_field_roots(params, branch, target_field_hat, atlas, continued, diagnostics)
      do guess_index = 1, size(continued)
        if (unique_count >= max_roots) exit
        candidate_root = continued(guess_index)
        call encode_unknowns(params, branch, candidate_root%phi0_v, candidate_root%phi_m_v, &
            candidate_root%ambient_electron_density_m3, y, success)
        if (.not. success) cycle
        call accept_candidate(y, int(continued(guess_index)%nonlinear_iterations), .true.)
      end do
    end if
  contains
    subroutine collect_scalar_roots()
      real(dp), allocatable :: knots(:), edges(:), potentials(:)
      real(dp) :: limit, temporary
      integer :: ngrid, point, j, remaining
      limit = params%search%potential_extent
      if (branch == 'B') limit = min(limit, nearest(ion_critical_potential(0.5_dp*params%t_swe_ev*params%mach**2, &
          params%ion_pressure_factor*params%t_swi_ev)/params%potential_scale_v, -1.0_dp))
      if (limit <= 0.0_dp) return
      edges = params%photoelectrons%search_breakpoints()/params%potential_scale_v
      allocate (knots(2*params%search%bracket_points + 3*size(edges)))
      ngrid = 0
      do point = 1, params%search%bracket_points
        ngrid = ngrid + 1
        knots(ngrid) = limit*exp(-28.0_dp + 28.0_dp*real(point - 1, dp)/real(params%search%bracket_points - 1, dp))
        ngrid = ngrid + 1
        knots(ngrid) = limit*real(point, dp)/real(params%search%bracket_points, dp)
      end do
      if (branch == 'B') then
        do point = 1, size(edges)
          if (edges(point) <= 0.0_dp .or. edges(point) >= limit) cycle
          knots(ngrid + 1:ngrid + 3) = [nearest(edges(point), -1.0_dp), edges(point), nearest(edges(point), 1.0_dp)]
          ngrid = ngrid + 3
        end do
      else
        knots(:ngrid) = -knots(:ngrid)
      end if
      do point = 2, ngrid
        temporary = knots(point)
        j = point - 1
        do while (j >= 1)
          if (knots(j) <= temporary) exit
          knots(j + 1) = knots(j)
          j = j - 1
        end do
        knots(j + 1) = temporary
      end do
      remaining = max_roots - unique_count
      call find_scalar_roots(scalar_residual, knots(:ngrid), params%search, potentials, diagnostics, k, remaining, scalar_candidate)
      ! Rejected algebraic candidates provide physical evidence; no located
      ! candidate at all remains an unresolved finite search.
      if (size(potentials) == 0 .and. diagnostics%rejected(k) == 0) &
          diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
    end subroutine

    subroutine scalar_candidate(phi_hat, iterations, accepted)
      real(dp), intent(in) :: phi_hat
      integer, intent(in) :: iterations
      logical, intent(out) :: accepted
      real(dp) :: physical(3)
      integer :: before
      before = unique_count
      physical(1) = phi_hat*params%potential_scale_v
      physical(2) = min(physical(1), 0.0_dp)
      physical(3) = neutral_electron_density(params, branch, physical(1), physical(2))
      ! No logarithmic encode/decode round trip at spectral boundaries.
      call accept_candidate([0.0_dp, 0.0_dp, 0.0_dp], iterations, .false., physical)
      accepted = unique_count > before
    end subroutine

    subroutine scalar_residual(phi_hat, f, valid)
      real(dp), intent(in) :: phi_hat
      real(dp), intent(out) :: f
      logical, intent(out) :: valid
      real(dp) :: phi, phim, density, integral
      f = huge(1.0_dp)
      valid = .false.
      phi = phi_hat*params%potential_scale_v
      phim = min(phi, 0.0_dp)
      density = neutral_electron_density(params, branch, phi, phim)
      if (.not. ieee_is_finite(density) .or. density <= 0.0_dp) return
      integral = integrate_zhao_rho(params, branch, 'monotonic', phi_hat, 0.0_dp, phi_hat, &
          phim/params%potential_scale_v, density/params%density_scale_m3)
      f = (2.0_dp*integral - target_field_hat**2)/max(1.0_dp, target_field_hat**2)
      valid = ieee_is_finite(f)
    end subroutine

    subroutine field_residual(value, f, valid)
      real(dp), intent(in) :: value(:)
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: valid
      real(dp) :: coordinates(3), residual(3)
      coordinates = 0.0_dp
      coordinates(:size(value)) = value
      call evaluate_charge_residual(params, branch, target_field_hat, coordinates, residual, valid)
      f = residual(:size(value))
    end subroutine

    subroutine accept_candidate(coordinates, iterations, from_atlas, physical)
      real(dp), intent(in) :: coordinates(3)
      integer, intent(in) :: iterations
      logical, intent(in) :: from_atlas
      real(dp), intent(in), optional :: physical(3)
      real(dp) :: original(3)
      candidate_root = zhao_field_root()
      candidate_root%branch = branch
      if (present(physical)) then
        candidate_root%phi0_v = physical(1)
        candidate_root%phi_m_v = physical(2)
        candidate_root%ambient_electron_density_m3 = physical(3)
      else
        call decode_unknowns(params, branch, coordinates, candidate_root%phi0_v, candidate_root%phi_m_v, &
            candidate_root%ambient_electron_density_m3, success)
        if (.not. success) then
          diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
          return
        end if
      end if
      call evaluate_physical_field_residual(params, branch, target_field_hat, candidate_root%phi0_v, &
          candidate_root%phi_m_v, candidate_root%ambient_electron_density_m3, original, success)
      diagnostics%evaluations(k) = diagnostics%evaluations(k) + 1
      if (.not. success) then
        diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
        return
      end if
      if (maxval(abs(original)) > params%search%residual_tolerance) then
        diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
        return
      end if

      if (target_field_hat == 0.0_dp .and. branch == 'A' .and. candidate_root%phi0_v < 0.0_dp .and. &
          candidate_root%phi0_v - candidate_root%phi_m_v < root_cluster_tolerance*params%potential_scale_v) then
        candidate_root%branch = 'C'
        candidate_root%phi0_v = candidate_root%phi_m_v
      end if
      candidate_root%residual_norm = maxval(abs(original))
      candidate_root%nonlinear_iterations = int(iterations, i32)
      call validate_field_root_profile(params, candidate_root, target_field_hat, profile_status, profile_message)
      if (profile_status == SHEATH_NO_PHYSICAL_SOLUTION) then
        diagnostics%rejected(k) = diagnostics%rejected(k) + 1
        return
      else if (profile_status /= SHEATH_OK) then
        diagnostics%profile_failures(k) = diagnostics%profile_failures(k) + 1
        return
      end if

      duplicate_root = .false.
      do root_index = 1, unique_count
        if (.not. field_roots_equivalent(params, candidate_root, unique_roots(root_index))) cycle
        duplicate_root = .true.
        if (candidate_root%residual_norm < unique_roots(root_index)%residual_norm) then
          unique_roots(root_index) = candidate_root
        end if
        exit
      end do
      if (.not. duplicate_root) then
        if (unique_count >= max_roots) return
        unique_count = unique_count + 1
        unique_roots(unique_count) = candidate_root
        if (from_atlas) diagnostics%atlas_hits(k) = diagnostics%atlas_hits(k) + 1
      end if
    end subroutine
  end subroutine collect_field_branch_roots

  pure logical function field_roots_equivalent(params, first, second) result(equivalent)
    type(zhao_params_type), intent(in) :: params
    type(zhao_field_root), intent(in) :: first, second

    real(dp) :: first_phi0_hat, second_phi0_hat, first_phi_m_hat, second_phi_m_hat
    real(dp) :: log_density_ratio

    equivalent = .false.
    if (first%branch /= second%branch) return
    if (min(first%ambient_electron_density_m3, second%ambient_electron_density_m3) <= 0.0_dp) return

    first_phi0_hat = first%phi0_v/params%potential_scale_v
    second_phi0_hat = second%phi0_v/params%potential_scale_v
    first_phi_m_hat = first%phi_m_v/params%potential_scale_v
    second_phi_m_hat = second%phi_m_v/params%potential_scale_v
    log_density_ratio = log(first%ambient_electron_density_m3/second%ambient_electron_density_m3)
    if (.not. all(ieee_is_finite([ &
        first_phi0_hat, second_phi0_hat, first_phi_m_hat, second_phi_m_hat, &
        log_density_ratio &
        ]))) return

    equivalent = abs(first_phi0_hat - second_phi0_hat) <= &
        root_cluster_tolerance*max(1.0_dp, abs(first_phi0_hat), abs(second_phi0_hat)) .and. &
        abs(first_phi_m_hat - second_phi_m_hat) <= &
        root_cluster_tolerance*max(1.0_dp, abs(first_phi_m_hat), abs(second_phi_m_hat)) .and. &
        abs(log_density_ratio) <= root_cluster_tolerance
  end function field_roots_equivalent

end submodule sheath_model_field_roots
