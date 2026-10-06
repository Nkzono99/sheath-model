! SPDX-License-Identifier: MIT
!> Prescribed-field maps and continuation, using the original field closure.
submodule(sheath_model_field) sheath_model_field_atlas
  implicit none
contains

  module subroutine prepare_field_params(input, params, status, message)
    type(zhao_field_input), intent(in) :: input
    type(zhao_params_type), intent(out) :: params
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    params = zhao_params_type()
    status = SHEATH_INVALID_ARGUMENT
    message = 'branch must be auto, A, B, or C.'
    select case (trim(lower_ascii(input%branch)))
    case ('auto', 'a', 'b', 'c')
    case default
      return
    end select
    message = 'The prescribed field must be finite.'
    if (.not. ieee_is_finite(input%electric_field_v_m)) return
    message = 'Invalid search options; prescribed-field search supports auto, newton, or lm.'
    if (.not. valid_search_options(input%search) .or. trim(lower_ascii(input%search%method)) == 'bracket') return
    call prepare_plasma_params(input, params, status, message)
    params%search = input%search
    params%search%method = lower_ascii(params%search%method)
  end subroutine

  !> Search bounds are per A/B/C type. A table supplies seeds, not accepted answers.
  module subroutine validate_field_search(atlas, max_roots, status, message)
    type(sheath_field_atlas), intent(in), optional :: atlas
    integer, intent(in), optional :: max_roots
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    status = SHEATH_INVALID_ARGUMENT
    message = 'Invalid field atlas controls or max_roots.'
    if (present(atlas)) then
      if (.not. atlas%valid()) return
    end if
    if (present(max_roots)) then
      if (max_roots < 1) return
    end if
    status = SHEATH_OK
    message = ''
  end subroutine

  module subroutine field_atlas_key(params, target, key, shape)
    type(zhao_params_type), intent(in) :: params
    real(dp), intent(in) :: target
    real(dp), intent(out) :: key(7)
    real(dp), allocatable, intent(out) :: shape(:)
    call equilibrium_key(params, key(:6), shape)
    key(7) = asinh(target)
  end subroutine

  !> Recheck the input field, original equations and physical profile before storage.
  module subroutine add_field_to_atlas(input, root, atlas, status, message, component)
    type(zhao_field_input), intent(in) :: input
    type(zhao_field_result), intent(in) :: root
    type(sheath_field_atlas), intent(inout) :: atlas
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    integer, intent(in), optional :: component
    type(zhao_params_type) :: params
    type(zhao_field_root) :: candidate
    real(dp) :: key(7), y(3), raw(3), target
    real(dp), allocatable :: shape(:)
    logical :: valid
    call prepare_field_params(input, params, status, message)
    if (status /= SHEATH_OK) return
    call validate_field_search(atlas, status=status, message=message)
    if (status /= SHEATH_OK) return
    status = SHEATH_INVALID_ARGUMENT
    message = 'A field atlas requires a non-flat root satisfying the specified field and physical profile.'
    if (.not. root%valid .or. index('ABC', root%branch) == 0) return
    if (present(component)) then
      if (component < 1) return
    end if
    target = input%electric_field_v_m/(params%potential_scale_v/params%length_scale_m)
    if (.not. ieee_is_finite(target)) return
    if (root%branch == 'C' .and. target > 0.0_dp) return
    if (root%branch /= 'C' .and. target < 0.0_dp) return
    if (root%branch == 'B' .and. root%minimum_potential_v /= 0.0_dp) return
    if (root%branch == 'C' .and. root%minimum_potential_v /= root%boundary_potential_v) return
    call encode_unknowns(params, root%branch, root%boundary_potential_v, root%minimum_potential_v, &
        root%ambient_electron_density_m3, y, valid)
    if (.not. valid) return
    call evaluate_charge_residual(params, root%branch, target, y, raw, valid)
    if (.not. valid) return
    if (maxval(abs(raw)) > params%search%residual_tolerance) return
    candidate%branch = root%branch
    candidate%phi0_v = root%boundary_potential_v
    candidate%phi_m_v = root%minimum_potential_v
    candidate%ambient_electron_density_m3 = root%ambient_electron_density_m3
    call validate_field_root_profile(params, candidate, target, status, message)
    if (status /= SHEATH_OK) return
    call field_atlas_key(params, target, key, shape)
    call atlas%insert(key, shape, root%branch, y, component)
    status = SHEATH_OK
    message = ''
  end subroutine

  !> report(A/B/C,input) distinguishes stored roots, exclusions and finite-search holes.
  !! The flat E=0 state is checked analytically at every query and is not log encoded.
  module subroutine build_field_atlas(inputs, atlas, status, message, report, deflation, max_roots)
    type(zhao_field_input), intent(in) :: inputs(:)
    type(sheath_field_atlas), intent(inout) :: atlas
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    integer(i32), allocatable, intent(out), optional :: report(:, :)
    logical, intent(in), optional :: deflation
    integer, intent(in), optional :: max_roots
    type(zhao_field_input) :: current
    type(zhao_params_type) :: params
    type(zhao_field_result), allocatable :: candidates(:)
    type(sheath_search_diagnostics) :: diagnostics
    integer(i32) :: outcomes(3, size(inputs)), local_status
    logical :: excluded(3, size(inputs))
    integer :: pass, i, k, j, before
    character(len=1), parameter :: branches(3) = ['A', 'B', 'C']
    outcomes = SHEATH_INVALID_ARGUMENT
    excluded = .false.
    call validate_field_search(atlas, max_roots, status, message)
    if (status /= SHEATH_OK) return
    do pass = 1, 3*size(inputs) + 1
      before = atlas%size()
      do i = 1, size(inputs)
        call prepare_field_params(inputs(i), params, local_status, message)
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
          current = inputs(i)
          current%branch = branches(k)
          call solve_prescribed_field_candidates(current, candidates, local_status, message, diagnostics, &
              atlas=atlas, deflation=deflation, max_roots=max_roots)
          outcomes(k, i) = local_status
          excluded(k, i) = diagnostics%excluded(k)
          if (local_status == SHEATH_INVALID_ARGUMENT) then
            status = local_status
            if (present(report)) report = outcomes
            return
          end if
          if (local_status /= SHEATH_OK) cycle
          do j = 1, size(candidates)
            if (candidates(j)%boundary_potential_v == 0.0_dp .and. &
                candidates(j)%minimum_potential_v == 0.0_dp) cycle
            call add_field_to_atlas(current, candidates(j), atlas, local_status, message)
            if (local_status /= SHEATH_OK) then
              status = local_status
              if (present(report)) report = outcomes
              return
            end if
          end do
        end do
      end do
      if (.not. any((outcomes == SHEATH_NUMERICAL_FAILURE .or. outcomes == SHEATH_NO_PHYSICAL_SOLUTION) &
          .and. .not. excluded) .or. atlas%size() == before) exit
    end do
    if (present(report)) report = outcomes
    status = SHEATH_OK
    message = 'Field atlas built; finite-search holes do not establish absence of solutions.'
  end subroutine

  !> Follow each nearby family; return only roots corrected at the target field.
  module subroutine continue_field_roots(params, branch, target, atlas, roots, diagnostics)
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: target
    type(sheath_field_atlas), intent(in) :: atlas
    type(zhao_field_root), allocatable, intent(out) :: roots(:)
    type(sheath_search_diagnostics), intent(inout) :: diagnostics
    type(sheath_atlas_point) :: point
    type(zhao_field_root) :: root
    integer, allocatable :: indices(:)
    real(dp), allocatable :: shape(:)
    real(dp) :: goal(7), start(7), y(3), raw(3)
    integer :: i, n, k, iterations_before
    logical :: success, valid
    call field_atlas_key(params, target, goal, shape)
    call atlas%neighbors(goal, shape, branch, indices)
    allocate (roots(0))
    n = merge(3, 2, branch == 'A')
    k = index('ABC', branch)
    do i = 1, size(indices)
      point = atlas%point(indices(i))
      start = point%key
      y = 0.0_dp
      iterations_before = diagnostics%iterations(k)
      call continue_guarded_system(n, residual, point%coordinates(:n), params%search, atlas%continuation, &
          diagnostics, k, y(:n), success, accept)
      if (.not. success) then
        diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
        cycle
      end if
      root = zhao_field_root()
      root%branch = branch
      call decode_unknowns(params, branch, y, root%phi0_v, root%phi_m_v, root%ambient_electron_density_m3, valid)
      if (.not. valid) cycle
      call evaluate_charge_residual(params, branch, target, y, raw, valid)
      if (.not. valid) cycle
      root%residual_norm = maxval(abs(raw))
      root%nonlinear_iterations = int(diagnostics%iterations(k) - iterations_before, i32)
      roots = [roots, root]
    end do
  contains
    subroutine path_state(t, current, field, valid)
      real(dp), intent(in) :: t
      type(zhao_params_type), intent(out) :: current
      real(dp), intent(out) :: field
      logical, intent(out) :: valid
      real(dp) :: key(7)
      field = target
      current = params
      valid = .true.
      if (t /= 1.0_dp) then
        key = start + t*(goal - start)
        valid = all(ieee_is_finite(key)) .and. maxval(abs(key)) <= 100.0_dp
        if (.not. valid) return
        call params_from_key(key(:6), params, current, valid)
        field = sinh(key(7))
      end if
      valid = valid .and. ((branch == 'C' .and. field <= 0.0_dp) .or. &
          (branch /= 'C' .and. field >= 0.0_dp))
      if ((branch == 'A' .or. branch == 'C') .and. current%u > 0.0_dp) valid = .false.
    end subroutine
    subroutine residual(value, t, f, valid)
      real(dp), intent(in) :: value(:), t
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: valid
      type(zhao_params_type) :: current
      real(dp) :: field, encoded(3), raw(3)
      call path_state(t, current, field, valid)
      f = 0.0_dp
      if (.not. valid) return
      encoded = 0.0_dp
      encoded(:n) = value
      call evaluate_charge_residual(current, branch, field, encoded, raw, valid)
      f = raw(:n)
    end subroutine
    logical function accept(value, t) result(valid)
      real(dp), intent(in) :: value(:), t
      type(zhao_params_type) :: current
      type(zhao_field_root) :: candidate
      real(dp) :: field, encoded(3)
      integer(i32) :: status
      character(len=256) :: message
      call path_state(t, current, field, valid)
      if (.not. valid) return
      encoded = 0.0_dp
      encoded(:n) = value
      candidate%branch = branch
      call decode_unknowns(current, branch, encoded, candidate%phi0_v, candidate%phi_m_v, &
          candidate%ambient_electron_density_m3, valid)
      if (.not. valid) return
      call validate_field_root_profile(current, candidate, field, status, message)
      valid = status == SHEATH_OK
    end function
  end subroutine
end submodule sheath_model_field_atlas
