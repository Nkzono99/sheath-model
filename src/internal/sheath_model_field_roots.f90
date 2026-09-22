! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone modules; status-returning public facade in sheath_model.
!> Enumerate Zhao roots and select a physical solution.
!! 数値解法と物理式は numerics / physics に委譲し、選択順と縮退判定をここに集める。
submodule(sheath_model_field) sheath_model_field_roots
  implicit none

  real(dp), parameter :: zero_field_tolerance_hat = 1.0e-12_dp
  real(dp), parameter :: root_cluster_tolerance = 1.0e-6_dp
  real(dp), parameter :: energy_tie_tolerance = 1.0e-6_dp

contains

  module procedure solve_field_root

  character(len=1) :: order(3), candidate
  type(zhao_field_root) :: trial_root, successful_root, successful_roots(3)
  real(dp) :: field_scale, target_field_hat, degenerate_density_m3
  integer :: candidate_count, candidate_index, successful_count
  logical :: saw_numerical_failure, saw_ambiguous_solution

  root = zhao_field_root()
  status = sheath_no_physical_solution
  message = ''
  field_scale = params%t_phe_ev/params%lambda_d_phe_ref_m
  target_field_hat = interface_field_v_m/field_scale
  if (.not. all(ieee_is_finite([field_scale, target_field_hat])) .or. field_scale <= 0.0_dp) then
    status = sheath_numerical_failure
    message = 'prescribed-field Zhao field normalization is invalid.'
    return
  end if

  call field_branch_order(model, target_field_hat, order, candidate_count, status, message)
  if (status /= sheath_ok) return
  if (abs(target_field_hat) <= zero_field_tolerance_hat) then
    if (trim(model) /= 'auto' .and. trim(model) /= 'b') then
      status = sheath_no_physical_solution
      message = 'requested Zhao branch does not contain the zero-field state.'
      return
    end if
    degenerate_density_m3 = ( &
                            2.0_dp*params%n_swi_inf_m3 - params%n_phe0_m3 &
                            )/(1.0_dp + erf(params%u))
    if (.not. ieee_is_finite(degenerate_density_m3) .or. degenerate_density_m3 <= 0.0_dp) then
      status = sheath_no_physical_solution
      message = 'zero-field Zhao-B state has no positive ambient electron density.'
      return
    end if
    root%branch = 'B'
    root%ambient_electron_density_m3 = degenerate_density_m3
    root%residual_norm = 0.0_dp
    root%minimum_field_squared_hat = 0.0_dp
    root%potential_energy_j_m2 = 0.0_dp
    root%nonlinear_iterations = 0_i32
    status = sheath_ok
    message = 'zero-field degenerate Zhao-B state'
    return
  end if

  saw_numerical_failure = .false.
  saw_ambiguous_solution = .false.
  successful_count = 0
  successful_root = zhao_field_root()
  do candidate_index = 1, candidate_count
    candidate = order(candidate_index)
    call solve_one_field_branch( &
      params, candidate, target_field_hat, root_selection, trial_root, status, message &
      )
    if (status == sheath_ok) then
      successful_count = successful_count + 1
      successful_roots(successful_count) = trial_root
      if (successful_count == 1) successful_root = trial_root
    end if
    if (status == sheath_numerical_failure) saw_numerical_failure = .true.
    if (status == sheath_ambiguous_solution) saw_ambiguous_solution = .true.
  end do
  if (saw_ambiguous_solution) then
    status = sheath_ambiguous_solution
    message = 'prescribed-field Zhao branch search found multiple roots within one branch.'
    return
  end if
  if (successful_count == 1 .and. saw_numerical_failure .and. trim(model) == 'auto') then
    status = sheath_numerical_failure
    message = 'prescribed-field Zhao auto selection could not certify a unique branch.'
    return
  else if (successful_count == 1) then
    root = successful_root
    status = sheath_ok
    message = ''
    return
  else if (successful_count > 1) then
    if (saw_numerical_failure .and. trim(model) == 'auto' .and. &
        trim(root_selection) == 'minimum_energy') then
      status = sheath_numerical_failure
      message = 'prescribed-field Zhao minimum-energy selection could not certify every candidate branch.'
    else if (trim(root_selection) == 'minimum_energy') then
      call select_minimum_energy_root( &
        params, successful_roots, successful_count, root, status, message &
        )
    else
      status = sheath_ambiguous_solution
      message = 'prescribed-field Zhao auto selection is ambiguous across multiple physical branches.'
    end if
    return
  end if
  if (saw_numerical_failure) then
    status = sheath_numerical_failure
    message = 'prescribed-field Zhao branch search did not converge.'
  else
    status = sheath_no_physical_solution
    message = 'no Zhao branch satisfies the prescribed boundary field.'
  end if
  end procedure solve_field_root

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
      if (target_field_hat > 0.0_dp) then
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

  subroutine solve_one_field_branch(params, branch, target_field_hat, root_selection, root, status, message)
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    character(len=*), intent(in) :: root_selection
    real(dp), intent(in) :: target_field_hat
    type(zhao_field_root), intent(out) :: root
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    type(zhao_field_root) :: unique_roots(8)
    integer :: unique_count
    logical :: saw_nonphysical_profile, saw_numerical_profile_failure

    root = zhao_field_root()
    root%branch = branch
    call collect_field_branch_roots( &
      params, branch, target_field_hat, unique_roots, unique_count, &
      saw_nonphysical_profile, saw_numerical_profile_failure, status, message &
      )
    if (status /= sheath_ok) return
    if (unique_count > 1) then
      if (trim(root_selection) == 'minimum_energy') then
        call select_minimum_energy_root(params, unique_roots, unique_count, root, status, message)
      else
        status = sheath_ambiguous_solution
        message = 'prescribed-field Zhao solve found multiple roots in the requested branch.'
      end if
    else if (saw_numerical_profile_failure) then
      status = sheath_numerical_failure
      message = 'prescribed-field Zhao root profile could not be certified numerically.'
    else if (unique_count == 1) then
      root = unique_roots(1)
      if (trim(root_selection) == 'minimum_energy') then
        call evaluate_root_potential_energy(params, root, status, message)
      else
        status = sheath_ok
        message = ''
      end if
    else if (saw_nonphysical_profile) then
      status = sheath_no_physical_solution
      message = 'prescribed-field Zhao endpoint root has no real connecting field profile.'
    else
      status = sheath_numerical_failure
      message = 'prescribed-field Zhao Newton solve did not converge.'
    end if
  end subroutine solve_one_field_branch

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
    compatible = (branch == 'C' .and. target_field_hat < 0.0_dp) .or. &
                 ((branch == 'A' .or. branch == 'B') .and. target_field_hat > 0.0_dp)
    if (.not. compatible) then
      message = 'Zhao branch and boundary field signs are incompatible.'
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
  end subroutine collect_field_branch_roots

  subroutine select_minimum_energy_root(params, roots, root_count, root, status, message)
    type(zhao_params_type), intent(in) :: params
    type(zhao_field_root), intent(in) :: roots(:)
    integer, intent(in) :: root_count
    type(zhao_field_root), intent(out) :: root
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    type(zhao_field_root) :: candidates(size(roots))
    real(dp) :: energy_scale
    integer :: candidate_index, best_index

    root = zhao_field_root()
    candidates = roots
    status = sheath_numerical_failure
    message = ''
    if (root_count < 1 .or. root_count > size(roots)) then
      message = 'prescribed-field Zhao minimum-energy selection received an invalid candidate count.'
      return
    end if
    do candidate_index = 1, root_count
      call evaluate_root_potential_energy(params, candidates(candidate_index), status, message)
      if (status /= sheath_ok) return
    end do
    best_index = 1
    do candidate_index = 2, root_count
      if (candidates(candidate_index)%potential_energy_j_m2 < &
          candidates(best_index)%potential_energy_j_m2) best_index = candidate_index
    end do
    do candidate_index = 1, root_count
      if (candidate_index == best_index) cycle
      energy_scale = max( &
                     abs(candidates(best_index)%potential_energy_j_m2), &
                     abs(candidates(candidate_index)%potential_energy_j_m2), tiny(1.0_dp) &
                     )
      if (abs( &
          candidates(candidate_index)%potential_energy_j_m2 - &
          candidates(best_index)%potential_energy_j_m2 &
          ) <= energy_tie_tolerance*energy_scale) then
        status = sheath_ambiguous_solution
        message = 'prescribed-field Zhao minimum-energy candidates are numerically tied.'
        return
      end if
    end do
    root = candidates(best_index)
    status = sheath_ok
    message = ''
  end subroutine select_minimum_energy_root

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
