! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone modules; status-returning public facade in sheath_model.
!> Enumerate Zhao roots and select a physical solution.
!! 数値解法と物理式は numerics / physics に委譲し、選択順と縮退判定をここに集める。
submodule(sheath_model_field) sheath_model_field_roots
  implicit none

  real(dp), parameter :: root_cluster_tolerance = 1.0e-6_dp

contains

  module subroutine solve_field_root(model, params, interface_field_v_m, root, status, message, diagnostics, initial_guesses)
    character(len=*), intent(in) :: model
    type(zhao_params_type), intent(in) :: params
    real(dp), intent(in) :: interface_field_v_m
    type(zhao_field_root), intent(out) :: root
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(zhao_field_search_diagnostics), intent(out) :: diagnostics
    type(zhao_field_result), intent(in), optional :: initial_guesses(:)

    type(zhao_field_root), allocatable :: roots(:)
    root = zhao_field_root()
    call find_field_roots(model, params, interface_field_v_m, roots, status, message, diagnostics, initial_guesses)
    if (status /= SHEATH_OK) return
    if (size(roots) == 1) then
      root = roots(1)
    else
      status = SHEATH_AMBIGUOUS_SOLUTION
      message = 'Multiple admissible roots found; use solve_prescribed_field_candidates to inspect them.'
    end if
  end subroutine solve_field_root

  module subroutine find_field_roots(model, params, interface_field_v_m, roots, status, message, diagnostics, initial_guesses)
    character(len=*), intent(in) :: model
    type(zhao_params_type), intent(in) :: params
    real(dp), intent(in) :: interface_field_v_m
    type(zhao_field_root), allocatable, intent(out) :: roots(:)
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(zhao_field_search_diagnostics), intent(out) :: diagnostics
    type(zhao_field_result), intent(in), optional :: initial_guesses(:)

    character(len=1) :: order(3)
    type(zhao_field_root), allocatable :: found(:), candidates(:)
    type(zhao_field_root) :: flat
    real(dp) :: field_scale, target, density
    integer :: branch_count, i, j, k, count, n, capacity, branch_index
    logical :: duplicate, unresolved

    diagnostics = zhao_field_search_diagnostics()
    field_scale = params%t_phe_ev/params%lambda_d_phe_ref_m
    target = interface_field_v_m/field_scale
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Invalid field normalization.'
    if (.not. ieee_is_finite(target) .or. field_scale <= 0.0_dp) return
    call field_branch_order(model, target, order, branch_count, status, message)
    if (status /= SHEATH_OK) return
    capacity = default_field_starts
    if (present(initial_guesses)) capacity = capacity + size(initial_guesses)
    allocate (found(3*capacity + 1))
    n = 0
    ! The flat state is one candidate, never a shortcut around the non-flat search.
    if (interface_field_v_m == 0.0_dp .and. (model == 'auto' .or. model == 'b')) then
      density = (2.0_dp*params%n_swi_inf_m3 - params%n_phe0_m3)/(1.0_dp + erf(params%u))
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
      call collect_field_branch_roots(params, order(i), target, candidates, count, diagnostics, initial_guesses)
      do j = 1, count
        duplicate = .false.
        do k = 1, n
          duplicate = field_roots_equivalent(params, candidates(j), found(k))
          if (interface_field_v_m == 0.0_dp .and. found(k)%phi0_v == 0.0_dp) then
            duplicate = duplicate .or. max(abs(candidates(j)%phi0_v), abs(candidates(j)%phi_m_v)) &
                < 1e-4_dp*params%t_phe_ev
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
    if (unresolved) message = 'Admissible candidates found; some starts or profile evaluations remain unresolved.'
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

  subroutine collect_field_branch_roots(params, branch, target_field_hat, unique_roots, unique_count, &
      diagnostics, initial_guesses)
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: target_field_hat
    type(zhao_field_root), allocatable, intent(out) :: unique_roots(:)
    integer, intent(out) :: unique_count
    type(zhao_field_search_diagnostics), intent(inout) :: diagnostics
    type(zhao_field_result), intent(in), optional :: initial_guesses(:)

    real(dp) :: defaults(3, default_field_starts), y(3), norm, encoded(3)
    real(dp), allocatable :: guesses(:, :)
    type(zhao_field_root) :: candidate_root
    integer :: guess_count, default_count, guess_index, iterations, root_index, k, capacity
    integer(i32) :: profile_status
    logical :: success, compatible, duplicate_root
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
    if (present(initial_guesses)) capacity = capacity + size(initial_guesses)
    allocate (guesses(3, capacity), unique_roots(capacity))
    guess_count = 0
    ! Nearby physical solutions supplement the independent starts; they do not
    ! select a preferred root or change the specified plasma/boundary conditions.
    if (present(initial_guesses)) then
      do guess_index = 1, size(initial_guesses)
        if (.not. initial_guesses(guess_index)%valid .or. initial_guesses(guess_index)%branch /= branch) cycle
        call encode_field_unknowns(params, branch, initial_guesses(guess_index)%boundary_potential_v, &
            initial_guesses(guess_index)%minimum_potential_v, &
            initial_guesses(guess_index)%ambient_electron_density_m3, encoded, success)
        if (.not. success) cycle
        guess_count = guess_count + 1
        guesses(:, guess_count) = encoded
      end do
    end if
    call make_field_branch_guesses(params, branch, target_field_hat, defaults, default_count)
    guesses(:, guess_count + 1:guess_count + default_count) = defaults(:, :default_count)
    guess_count = guess_count + default_count
    do guess_index = 1, guess_count
      diagnostics%starts(k) = diagnostics%starts(k) + 1
      call newton_field_branch(params, branch, target_field_hat, guesses(:, guess_index), &
          y, norm, iterations, success)
      if (.not. success) then
        diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
        cycle
      end if
      candidate_root = zhao_field_root()
      candidate_root%branch = branch
      call decode_field_unknowns(params, branch, y, candidate_root%phi0_v, candidate_root%phi_m_v, &
          candidate_root%ambient_electron_density_m3, success)
      if (.not. success) then
        diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
        cycle
      end if
      if (target_field_hat == 0.0_dp .and. branch == 'A' .and. candidate_root%phi0_v < 0.0_dp .and. &
          candidate_root%phi0_v - candidate_root%phi_m_v < root_cluster_tolerance*params%t_phe_ev) then
        candidate_root%branch = 'C'
        candidate_root%phi0_v = candidate_root%phi_m_v
      end if
      candidate_root%residual_norm = norm
      candidate_root%nonlinear_iterations = int(iterations, i32)
      call validate_field_root_profile(params, candidate_root, target_field_hat, profile_status, profile_message)
      if (profile_status == SHEATH_NO_PHYSICAL_SOLUTION) then
        diagnostics%rejected(k) = diagnostics%rejected(k) + 1
        cycle
      else if (profile_status /= SHEATH_OK) then
        diagnostics%profile_failures(k) = diagnostics%profile_failures(k) + 1
        cycle
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
        unique_count = unique_count + 1
        unique_roots(unique_count) = candidate_root
      end if
    end do
  end subroutine collect_field_branch_roots

  pure logical function field_roots_equivalent(params, first, second) result(equivalent)
    type(zhao_params_type), intent(in) :: params
    type(zhao_field_root), intent(in) :: first, second

    real(dp) :: first_phi0_hat, second_phi0_hat, first_phi_m_hat, second_phi_m_hat
    real(dp) :: log_density_ratio

    equivalent = .false.
    if (first%branch /= second%branch) return
    if (min(first%ambient_electron_density_m3, second%ambient_electron_density_m3) <= 0.0_dp) return
    first_phi0_hat = first%phi0_v/params%t_phe_ev
    second_phi0_hat = second%phi0_v/params%t_phe_ev
    first_phi_m_hat = first%phi_m_v/params%t_phe_ev
    second_phi_m_hat = second%phi_m_v/params%t_phe_ev
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
