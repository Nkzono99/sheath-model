! SPDX-License-Identifier: MIT
!> Search J=0 roots independently of the solar/fixed-entry parameterization.
module sheath_model_equilibrium_search
  use sheath_model_constants, only: dp, i32
  use sheath_model_search, only: sheath_search_diagnostics
  use sheath_model_core, only: zhao_params_type, zhao_residuals_type_a, zhao_residuals_type_b, &
      zhao_residuals_type_c, evaluate_monotonic_stationary_phi
  use sheath_model_coordinates, only: encode_unknowns, decode_unknowns, make_branch_guesses
  use sheath_model_numerics, only: solve_guarded_system
  use sheath_model_equilibrium_physics, only: equilibrium_residual, encoded_equilibrium_residual
  use sheath_model_continuation, only: find_guarded_roots
  use sheath_model_admissibility, only: validate_zhao_profile
  use sheath_model_ions, only: ion_density_ratio, ion_critical_potential
  use sheath_model_status, only: SHEATH_OK, SHEATH_NO_PHYSICAL_SOLUTION
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: search_equilibrium_branch, equilibrium_residual
contains
  !> All J=0 equations normalized to ion density; the A connection additionally
  !! scales with minimum depth^(3/2) so a collapsed segment is not a root.
  subroutine search_equilibrium_branch(p, branch, initial, x, success, diagnostics, atlas_count, all_roots, deflation, max_roots)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: initial(:, :) ! [surface V, minimum V, electron m^-3]
    real(dp), intent(out) :: x(3)
    logical, intent(out) :: success
    type(sheath_search_diagnostics), intent(inout) :: diagnostics
    integer, intent(in), optional :: atlas_count
    real(dp), allocatable, intent(out), optional :: all_roots(:, :)
    logical, intent(in), optional :: deflation
    integer, intent(in), optional :: max_roots
    real(dp) :: defaults(3, 16), y(3), y0(3), norm, trial(3)
    real(dp), allocatable :: guesses(:, :)
    real(dp), allocatable :: collected(:, :), known(:, :), algebraic(:, :)
    logical, allocatable :: atlas_flags(:)
    integer, allocatable :: origins(:)
    integer :: k, n, count, i, default_count, iterations, evaluations, lm_steps
    logical :: valid, converged, scalar_tried
    logical :: enumerate, use_deflation
    integer :: root_count, root_limit, j

    k = index('ABC', branch)
    n = merge(3, 2, branch == 'A')
    diagnostics%searched(k) = .true.
    success = .false.
    scalar_tried = .false.
    x = 0.0_dp
    enumerate = present(all_roots)
    root_count = 0
    root_limit = 16
    if (present(max_roots)) root_limit = max_roots
    allocate (collected(3, root_limit))
    if (enumerate) allocate (all_roots(3, 0))
    use_deflation = .true.
    if (present(deflation)) use_deflation = deflation
    if ((branch == 'A' .or. branch == 'C') .and. p%u > 0.0_dp) then
      diagnostics%excluded(k) = .true.
      return
    end if
    ! Caller seeds precede independent starts; no state is retained in the solver.
    allocate (guesses(3, size(initial, 2) + 16))
    allocate (atlas_flags(size(initial, 2) + 16))
    atlas_flags = .false.
    count = 0
    do i = 1, size(initial, 2)
      call encode_unknowns(p, branch, initial(1, i), initial(2, i), initial(3, i), y0, valid)
      if (.not. valid) cycle
      count = count + 1
      guesses(:, count) = y0
      if (present(atlas_count)) atlas_flags(count) = i > size(initial, 2) - atlas_count
    end do
    default_count = 0
    if (p%search%use_default_guesses) call make_branch_guesses(p, branch, 0.0_dp, defaults, default_count)
    guesses(:, count + 1:count + default_count) = defaults(:, :default_count)
    count = count + default_count

    if (enumerate) then
      if (branch /= 'A' .and. (p%search%method == 'auto' .or. p%search%method == 'bracket')) &
          call scalar_search(x, success)
      if (p%search%method /= 'bracket' .and. root_count < root_limit) then
        allocate (known(n, root_count))
        do j = 1, root_count
          call encode_unknowns(p, branch, collected(1, j), collected(2, j), collected(3, j), y, valid)
          known(:, j) = y(:n)
        end do
        call find_guarded_roots(n, residual, guesses(:n, :count), p%search, diagnostics, k, algebraic, root_limit, &
            use_deflation, known, atlas_flags(:count), origins)
        do j = 1, size(algebraic, 2)
          y = 0.0_dp
          y(:n) = algebraic(:, j)
          call decode_unknowns(p, branch, y, trial(1), trial(2), trial(3), valid)
          if (.not. valid) cycle
          call accept_candidate(trial, success)
          if (success .and. atlas_flags(origins(j))) diagnostics%atlas_hits(k) = diagnostics%atlas_hits(k) + 1
        end do
      end if
      all_roots = collected(:, :root_count)
      success = root_count > 0
      if (success) x = collected(:, 1)
      return
    end if

    ! Explicit methods do not silently fall back. auto keeps caller continuation
    ! seeds first, then uses scalar reduction for monotonic branches.
    do i = 1, min(count, p%search%max_starts)
      if (trim(p%search%method) == 'bracket') exit
      if (i > size(initial, 2) .and. trim(p%search%method) == 'auto' .and. branch /= 'A' .and. .not. scalar_tried) then
        call scalar_search(x, success)
        scalar_tried = .true.
        if (success) return
      end if
      diagnostics%starts(k) = diagnostics%starts(k) + 1
      if (atlas_flags(i)) diagnostics%atlas_starts(k) = diagnostics%atlas_starts(k) + 1
      y = guesses(:, i)
      call solve_guarded_system(n, y(1:n), residual, p%search, y0(1:n), norm, iterations, converged, evaluations, lm_steps)
      diagnostics%evaluations(k) = diagnostics%evaluations(k) + evaluations
      diagnostics%iterations(k) = diagnostics%iterations(k) + iterations
      diagnostics%lm_steps(k) = diagnostics%lm_steps(k) + lm_steps
      diagnostics%best_residual(k) = min(diagnostics%best_residual(k), norm)
      if (.not. converged) then
        diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
        cycle
      end if
      y(1:n) = y0(1:n)
      call decode_unknowns(p, branch, y, trial(1), trial(2), trial(3), valid)
      if (.not. valid) cycle
      call accept_candidate(trial, success)
      if (success) then
        x = trial
        if (atlas_flags(i)) diagnostics%atlas_hits(k) = diagnostics%atlas_hits(k) + 1
        return
      end if
    end do
    if (.not. scalar_tried .and. branch /= 'A' .and. &
        (trim(p%search%method) == 'auto' .or. trim(p%search%method) == 'bracket')) then
      call scalar_search(x, success)
    end if
  contains
    subroutine residual(value, f, ok)
      real(dp), intent(in) :: value(:)
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: ok
      call encoded_equilibrium_residual(p, branch, value, f, ok)
    end subroutine residual

    subroutine accept_candidate(physical, accepted)
      real(dp), intent(in) :: physical(3)
      logical, intent(out) :: accepted
      real(dp) :: minimum_e2, boundary_e2, raw(3)
      real(dp) :: candidate_y(3), other_y(3)
      integer :: other
      logical :: encoded_ok
      integer(i32) :: status
      character(len=256) :: message
      accepted = .false.
      if (enumerate) then
        if (root_count >= root_limit) return
        call encode_unknowns(p, branch, physical(1), physical(2), physical(3), candidate_y, encoded_ok)
        if (.not. encoded_ok) return
        do other = 1, root_count
          call encode_unknowns(p, branch, collected(1, other), collected(2, other), collected(3, other), other_y, encoded_ok)
          if (sqrt(sum((candidate_y - other_y)**2)) < 1e-6_dp) return
        end do
      end if
      call equilibrium_residual(p, branch, physical, raw)
      if (.not. all(ieee_is_finite(raw)) .or. maxval(abs(raw)) > p%search%residual_tolerance) then
        diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
        accepted = .false.
        return
      end if
      call validate_zhao_profile(p, branch, physical(1)/p%potential_scale_v, physical(2)/p%potential_scale_v, &
          physical(3)/p%density_scale_m3, minimum_e2, boundary_e2, status, message)
      accepted = status == SHEATH_OK
      if (accepted) then
        diagnostics%roots_found(k) = diagnostics%roots_found(k) + 1
        if (enumerate) then
          root_count = root_count + 1
          collected(:, root_count) = physical
        end if
      else if (status == SHEATH_NO_PHYSICAL_SOLUTION) then
        diagnostics%rejected(k) = diagnostics%rejected(k) + 1
      else
        diagnostics%profile_failures(k) = diagnostics%profile_failures(k) + 1
      end if
    end subroutine accept_candidate

    subroutine scalar_value(phi, f, density, ok)
      real(dp), intent(in) :: phi
      real(dp), intent(out) :: f, density
      logical, intent(out) :: ok
      real(dp) :: ion
      diagnostics%evaluations(k) = diagnostics%evaluations(k) + 1
      call evaluate_monotonic_stationary_phi(p, branch, phi, f, density, ok)
      f = f/p%n_swi_inf_m3
      ion = ion_density_ratio(max(phi, 0.0_dp), 0.5_dp*p%t_swe_ev*p%mach**2, p%ion_pressure_factor*p%t_swi_ev)
      ok = ok .and. ieee_is_finite(ion)
      if (ok) diagnostics%best_residual(k) = min(diagnostics%best_residual(k), abs(f))
    end subroutine scalar_value

    subroutine scalar_search(physical, accepted)
      real(dp), intent(out) :: physical(3)
      logical, intent(out) :: accepted
      real(dp), allocatable :: grid(:), edges(:)
      real(dp) :: limit, left, right, fleft, fright, density, mid, fmid, lo, hi, flo, temporary
      integer :: point, iteration, j, ngrid
      logical :: left_ok, right_ok, mid_ok, located
      accepted = .false.
      physical = 0.0_dp
      limit = p%search%potential_extent*p%potential_scale_v
      if (branch == 'B') limit = min(limit, nearest(ion_critical_potential(0.5_dp*p%t_swe_ev*p%mach**2, &
          p%ion_pressure_factor*p%t_swi_ev), -1.0_dp))
      if (limit <= 0.0_dp) return
      edges = p%photoelectrons%search_breakpoints()
      allocate (grid(2*p%search%bracket_points + size(edges)))
      ngrid = 0
      do point = 1, p%search%bracket_points
        ngrid = ngrid + 1
        grid(ngrid) = limit*exp(-28.0_dp + 28.0_dp*real(point - 1, dp)/real(p%search%bracket_points - 1, dp))
        ngrid = ngrid + 1
        grid(ngrid) = limit*real(point, dp)/real(p%search%bracket_points, dp)
      end do
      if (branch == 'B') then
        do point = 1, size(edges)
          if (edges(point) <= 0.0_dp .or. edges(point) > limit) cycle
          ngrid = ngrid + 1
          grid(ngrid) = edges(point)
        end do
      end if
      do point = 2, ngrid
        temporary = grid(point)
        j = point - 1
        do while (j >= 1)
          if (grid(j) <= temporary) exit
          grid(j + 1) = grid(j)
          j = j - 1
        end do
        grid(j + 1) = temporary
      end do
      if (branch == 'C') grid(:ngrid) = -grid(:ngrid)
      left_ok = .false.
      left = 0.0_dp
      fleft = 0.0_dp
      do point = 1, ngrid
        right = grid(point)
        call scalar_value(right, fright, density, right_ok)
        located = right_ok .and. abs(fright) <= p%search%residual_tolerance
        if (right_ok .and. left_ok .and. .not. located) then
          if (sign(1.0_dp, fleft) /= sign(1.0_dp, fright)) then
            diagnostics%brackets(k) = diagnostics%brackets(k) + 1
            lo = left
            hi = right
            flo = fleft
            do iteration = 1, p%search%max_iterations
              diagnostics%iterations(k) = diagnostics%iterations(k) + 1
              mid = 0.5_dp*lo + 0.5_dp*hi
              ! A small interval alone is not residual convergence.
              if (mid == lo .or. mid == hi) exit
              call scalar_value(mid, fmid, density, mid_ok)
              if (.not. mid_ok) exit ! do not bridge disconnected valid domains
              if (abs(fmid) <= p%search%residual_tolerance) then
                right = mid
                located = .true.
                exit
              end if
              if (sign(1.0_dp, flo) /= sign(1.0_dp, fmid)) then
                hi = mid
              else
                lo = mid
                flo = fmid
              end if
            end do
            if (.not. located) diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
          end if
        end if
        if (located) then
          physical = [right, merge(0.0_dp, right, branch == 'B'), density]
          call accept_candidate(physical, accepted)
          if (accepted .and. (.not. enumerate .or. root_count >= root_limit)) return
        end if
        left = grid(point)
        fleft = fright
        left_ok = right_ok
      end do
    end subroutine scalar_search
  end subroutine search_equilibrium_branch
end module sheath_model_equilibrium_search
