! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone modules; status-returning public facade in sheath_model.
!> Enumerate Zhao roots and select a physical solution.
!! 数値解法と物理式は numerics / physics に委譲し、選択順と縮退判定をここに集める。
submodule(sheath_model_field) sheath_model_field_roots
  implicit none

  real(dp), parameter :: root_cluster_tolerance = 1.0e-6_dp

contains

  module procedure solve_field_root
  type(zhao_field_root), allocatable :: roots(:)
  root = zhao_field_root()
  call find_field_roots(model, params, interface_field_v_m, roots, status, message)
  if (status /= sheath_ok) return
  if (size(roots) == 1) then
    root = roots(1)
  else
    status = sheath_ambiguous_solution
    message = 'Multiple admissible roots found; use solve_prescribed_field_candidates to inspect them.'
  end if
  end procedure solve_field_root

  module procedure find_field_roots
  character(len=1) :: order(3)
  type(zhao_field_root) :: found(25), candidates(8), flat
  real(dp) :: field_scale, target, density
  integer :: branch_count, i, j, k, count, n
  logical :: nonphysical, failed_profile, duplicate, saw_nonphysical, saw_failure, unresolved_search
  field_scale = params%t_phe_ev/params%lambda_d_phe_ref_m
  target = interface_field_v_m/field_scale
  status = sheath_numerical_failure
  message = 'Invalid field normalization.'
  if (.not. ieee_is_finite(target) .or. field_scale <= 0.0_dp) return
  call field_branch_order(model, target, order, branch_count, status, message)
  if (status /= sheath_ok) return
  n = 0
  saw_nonphysical = .false.
  saw_failure = .false.
  unresolved_search = .false.
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
    call collect_field_branch_roots(params, order(i), target, candidates, count, &
                                    nonphysical, failed_profile, status, message)
    saw_nonphysical = saw_nonphysical .or. nonphysical .or. status == sheath_no_physical_solution
    saw_failure = saw_failure .or. failed_profile
    unresolved_search = unresolved_search .or. status == sheath_numerical_failure
    if (status /= sheath_ok) cycle
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
  if (saw_failure) then
    status = sheath_numerical_failure
    message = 'A candidate profile integral could not be evaluated.'
    return
  end if
  if (n == 0) then
    status = sheath_numerical_failure
    message = 'No root converged in the finite multistart search.'
    if (saw_nonphysical .and. .not. unresolved_search) then
      status = sheath_no_physical_solution
      message = 'No admissible profile found among the converged roots or compatible branches.'
    end if
    return
  end if
  roots = found(:n)
  status = sheath_ok
  message = ''
  end procedure find_field_roots

  subroutine field_branch_order(model, target_field_hat, order, count, status, message)
    character(len=*), intent(in) :: model
    real(dp), intent(in) :: target_field_hat
    character(len=1), intent(out) :: order(3)
    integer, intent(out) :: count
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    order = ' '
    count = 0
    status = sheath_ok
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
      status = sheath_invalid_argument
      message = 'unknown prescribed-field Zhao branch.'
    end select
  end subroutine field_branch_order

  subroutine collect_field_branch_roots( &
    params, branch, target_field_hat, unique_roots, unique_count, &
    saw_nonphysical_profile, saw_numerical_profile_failure, status, message &
    )
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: target_field_hat
    type(zhao_field_root), intent(out) :: unique_roots(8)
    integer, intent(out) :: unique_count
    logical, intent(out) :: saw_nonphysical_profile, saw_numerical_profile_failure
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    real(dp) :: guesses(3, 8), y(3), norm
    type(zhao_field_root) :: candidate_root
    integer :: guess_count, guess_index, iterations, root_index
    integer(i32) :: profile_status
    logical :: success, compatible, duplicate_root
    character(len=512) :: profile_message

    unique_roots = zhao_field_root()
    unique_count = 0
    saw_nonphysical_profile = .false.
    saw_numerical_profile_failure = .false.
    status = sheath_no_physical_solution
    message = ''
    compatible = (branch == 'C' .and. target_field_hat <= 0.0_dp) .or. &
                 ((branch == 'A' .or. branch == 'B') .and. target_field_hat >= 0.0_dp)
    if (.not. compatible) then
      message = 'Zhao branch and boundary field signs are incompatible.'
      return
    end if

    if ((branch == 'A' .or. branch == 'C') .and. params%u > 0.0_dp) then
      saw_nonphysical_profile = .true.
      message = 'Reflected drifting electrons cannot approach neutral zero-field infinity.'
      return
    end if

    call make_field_branch_guesses(params, branch, guesses, guess_count)
    do guess_index = 1, guess_count
      call newton_field_branch( &
        params, branch, target_field_hat, guesses(:, guess_index), y, norm, iterations, success &
        )
      if (.not. success) cycle
      candidate_root = zhao_field_root()
      candidate_root%branch = branch
      call decode_field_unknowns( &
        params, branch, y, candidate_root%phi0_v, candidate_root%phi_m_v, &
        candidate_root%ambient_electron_density_m3, success &
        )
      if (.not. success) cycle
      if (target_field_hat == 0.0_dp .and. branch == 'A' .and. candidate_root%phi0_v < 0.0_dp .and. &
          candidate_root%phi0_v - candidate_root%phi_m_v < root_cluster_tolerance*params%t_phe_ev) then
        candidate_root%branch = 'C'
        candidate_root%phi0_v = candidate_root%phi_m_v
      end if
      candidate_root%residual_norm = norm
      candidate_root%nonlinear_iterations = int(iterations, i32)
      call validate_field_root_profile( &
        params, candidate_root, target_field_hat, profile_status, profile_message &
        )
      if (profile_status == sheath_no_physical_solution) then
        saw_nonphysical_profile = .true.
        cycle
      else if (profile_status /= sheath_ok) then
        saw_numerical_profile_failure = .true.
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
    status = sheath_ok
    if (unique_count == 0) then
      status = sheath_numerical_failure
      if (saw_nonphysical_profile) status = sheath_no_physical_solution
    end if
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
