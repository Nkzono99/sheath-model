! SPDX-License-Identifier: MIT
!> Parameter/pseudo-arclength continuation and shifted deflation for guarded residuals.
module sheath_model_continuation
  use sheath_model_constants, only: dp
  use sheath_model_search, only: sheath_search_options, sheath_search_diagnostics, sheath_continuation_options
  use sheath_model_numerics, only: solve_guarded_system, solve_guarded_linear_system
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: continue_guarded_system, find_guarded_roots
  abstract interface
    subroutine parameter_residual(y, t, f, valid)
      import dp
      real(dp), intent(in) :: y(:), t
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: valid
    end subroutine
    logical function accept_root(y, t) result(valid)
      import dp
      real(dp), intent(in) :: y(:), t
    end function
    subroutine root_residual(y, f, valid)
      import dp
      real(dp), intent(in) :: y(:)
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: valid
    end subroutine
  end interface
contains

  !> Follow F(y,t)=0 from t=0 to t=1; always correct the final fixed parameter.
  subroutine continue_guarded_system(n, residual, start, search, options, diagnostics, k, y_out, success, accept)
    integer, intent(in) :: n, k
    procedure(parameter_residual) :: residual
    real(dp), intent(in) :: start(n)
    type(sheath_search_options), intent(in) :: search
    type(sheath_continuation_options), intent(in) :: options
    type(sheath_search_diagnostics), intent(inout) :: diagnostics
    real(dp), intent(out) :: y_out(n)
    logical, intent(out) :: success
    procedure(accept_root), optional :: accept
    type(sheath_search_options) :: local
    real(dp) :: y(n), t, step, previous_y(n), previous_t, trial(n), trial_t, prediction(n + 1)
    real(dp) :: tangent(n + 1), next_tangent(n + 1), z(n + 1), zp(n + 1), zm(n + 1), jac(n, n + 1)
    real(dp) :: f0(n), fp(n), fm(n), h, augmented(n + 1, n + 1), rhs(n + 1), corrected(n + 1), final_norm
    real(dp) :: end_prediction(n), end_root(n)
    integer :: attempt, column, iterations, evaluations, lm_steps
    logical :: previous, plus_valid, minus_valid, valid, converged, end_ok
    local = search
    if (local%method == 'auto' .or. local%method == 'bracket') local%method = 'newton'
    local%use_default_guesses = .false.
    y = start
    t = 0.0_dp
    step = options%initial_step
    previous = .false.
    tangent = 0.0_dp
    tangent(n + 1) = 1.0_dp
    success = .false.
    y_out = y
    if (.not. valid_root(y, t)) return
    do attempt = 1, options%max_steps
      if (options%method == 'parameter') then
        trial_t = min(1.0_dp, t + step)
        prediction(:n) = y
        if (previous) then
          if (t /= previous_t) prediction(:n) = y + (trial_t - t)/(t - previous_t)*(y - previous_y)
        end if
        call solve_guarded_system(n, prediction(:n), fixed_residual, local, trial, final_norm, iterations, &
            converged, evaluations, lm_steps)
        call count_kernel(iterations, evaluations, lm_steps)
      else
        z(:n) = y
        z(n + 1) = t
        call residual(y, t, f0, valid)
        diagnostics%evaluations(k) = diagnostics%evaluations(k) + 1
        if (.not. valid) return
        do column = 1, n + 1
          h = epsilon(1.0_dp)**(1.0_dp/3.0_dp)*max(1.0_dp, abs(z(column)))
          zp = z
          zm = z
          zp(column) = zp(column) + h
          zm(column) = zm(column) - h
          call residual(zp(:n), zp(n + 1), fp, plus_valid)
          call residual(zm(:n), zm(n + 1), fm, minus_valid)
          diagnostics%evaluations(k) = diagnostics%evaluations(k) + 2
          if (plus_valid .and. minus_valid) then
            jac(:, column) = (fp - fm)/(2.0_dp*h)
          else if (plus_valid) then
            jac(:, column) = (fp - f0)/h
          else if (minus_valid) then
            jac(:, column) = (f0 - fm)/h
          else
            return
          end if
        end do
        if (.not. all(ieee_is_finite(jac))) return
        augmented(:n, :) = jac
        augmented(n + 1, :) = tangent
        rhs = 0.0_dp
        rhs(n + 1) = 1.0_dp
        call solve_guarded_linear_system(n + 1, augmented, rhs, next_tangent, valid)
        if (.not. valid) return
        next_tangent = next_tangent/sqrt(sum(next_tangent**2))
        prediction = z + step*next_tangent
        call solve_guarded_system(n + 1, prediction, augmented_residual, local, corrected, final_norm, iterations, &
            converged, evaluations, lm_steps)
        call count_kernel(iterations, evaluations, lm_steps)
        trial = corrected(:n)
        trial_t = corrected(n + 1)
        converged = converged .and. sqrt(sum((trial - y)**2)) <= options%max_root_distance
        if (converged .and. (t - 1.0_dp)*(trial_t - 1.0_dp) <= 0.0_dp .and. trial_t /= t) then
          end_prediction = y + (1.0_dp - t)/(trial_t - t)*(trial - y)
          trial_t = 1.0_dp
          call solve_guarded_system(n, end_prediction, fixed_residual, local, end_root, final_norm, iterations, &
              end_ok, evaluations, lm_steps)
          call count_kernel(iterations, evaluations, lm_steps)
          if (end_ok .and. sqrt(sum((end_root - y)**2)) <= options%max_root_distance) then
            if (valid_root(end_root, 1.0_dp)) then
              diagnostics%continuation_steps(k) = diagnostics%continuation_steps(k) + 1
              y_out = end_root
              success = .true.
              return
            end if
          end if
          converged = .false.
        end if
      end if
      valid = .false.
      if (converged .and. sqrt(sum((trial - y)**2)) <= options%max_root_distance) valid = valid_root(trial, trial_t)
      if (.not. valid) then
        diagnostics%continuation_retries(k) = diagnostics%continuation_retries(k) + 1
        step = step*0.5_dp
        if (step < options%min_step) return
        cycle
      end if
      previous_y = y
      previous_t = t
      previous = .true.
      y = trial
      t = trial_t
      if (options%method == 'arclength') tangent = next_tangent
      diagnostics%continuation_steps(k) = diagnostics%continuation_steps(k) + 1
      y_out = y
      if (t == 1.0_dp) then
        success = .true.
        return
      end if
      step = min(options%max_step, step*1.5_dp)
    end do
  contains
    logical function valid_root(value, parameter) result(ok)
      real(dp), intent(in) :: value(:), parameter
      real(dp) :: raw(n)
      call residual(value, parameter, raw, ok)
      diagnostics%evaluations(k) = diagnostics%evaluations(k) + 1
      ok = ok .and. all(ieee_is_finite(raw))
      if (ok .and. parameter == 1.0_dp) &
          diagnostics%best_residual(k) = min(diagnostics%best_residual(k), maxval(abs(raw)))
      if (ok) ok = maxval(abs(raw)) <= local%residual_tolerance
      if (ok .and. present(accept)) ok = accept(value, parameter)
    end function
    subroutine fixed_residual(value, f, ok)
      real(dp), intent(in) :: value(:)
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: ok
      call residual(value, trial_t, f, ok)
    end subroutine
    subroutine augmented_residual(value, f, ok)
      real(dp), intent(in) :: value(:)
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: ok
      call residual(value(:n), value(n + 1), f(:n), ok)
      f(n + 1) = dot_product(value - prediction, next_tangent)
    end subroutine
    subroutine count_kernel(it, ev, lm)
      integer, intent(in) :: it, ev, lm
      diagnostics%iterations(k) = diagnostics%iterations(k) + it
      diagnostics%evaluations(k) = diagnostics%evaluations(k) + ev
      diagnostics%lm_steps(k) = diagnostics%lm_steps(k) + lm
    end subroutine
  end subroutine continue_guarded_system

  !> Enumerate algebraic roots with multistart and optional shifted deflation.
  !! Always check the original residual; callers must still validate profiles.
  subroutine find_guarded_roots(n, residual, starts, search, diagnostics, k, roots, max_roots, deflation, &
      known_roots, atlas_flags, origins)
    integer, intent(in) :: n, k, max_roots
    procedure(root_residual) :: residual
    real(dp), intent(in) :: starts(:, :)
    type(sheath_search_options), intent(in) :: search
    type(sheath_search_diagnostics), intent(inout) :: diagnostics
    real(dp), allocatable, intent(out) :: roots(:, :)
    logical, intent(in) :: deflation
    real(dp), intent(in), optional :: known_roots(:, :)
    logical, intent(in), optional :: atlas_flags(:)
    integer, allocatable, intent(out), optional :: origins(:)
    real(dp) :: stored(n, max_roots), value(n), norm, raw(n)
    integer :: stored_origins(max_roots)
    integer :: count, initial_count, i, j, iterations, evaluations, lm_steps
    logical :: valid, success, duplicate
    count = 0
    if (present(known_roots)) then
      count = min(size(known_roots, 2), max_roots)
      stored(:, :count) = known_roots(:, :count)
    end if
    initial_count = count
    do i = 1, size(starts, 2)
      do while (diagnostics%starts(k) < search%max_starts .and. count < max_roots)
        diagnostics%starts(k) = diagnostics%starts(k) + 1
        if (present(atlas_flags)) then
          if (atlas_flags(i)) diagnostics%atlas_starts(k) = diagnostics%atlas_starts(k) + 1
        end if
        call solve_guarded_system(n, starts(:, i), modified_residual, search, value, norm, &
            iterations, success, evaluations, lm_steps)
        diagnostics%iterations(k) = diagnostics%iterations(k) + iterations
        diagnostics%evaluations(k) = diagnostics%evaluations(k) + evaluations + 1
        diagnostics%lm_steps(k) = diagnostics%lm_steps(k) + lm_steps
        call residual(value, raw, valid)
        if (.not. success .or. .not. valid .or. .not. all(ieee_is_finite(raw))) then
          diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
          exit
        end if
        norm = maxval(abs(raw))
        diagnostics%best_residual(k) = min(diagnostics%best_residual(k), norm)
        duplicate = .false.
        do j = 1, count
          if (sqrt(sum((stored(:, j) - value)**2)) < 1e-6_dp) duplicate = .true.
        end do
        if (norm > search%residual_tolerance .or. duplicate) exit
        count = count + 1
        stored(:, count) = value
        stored_origins(count) = i
        if (.not. deflation) exit
        diagnostics%deflations(k) = diagnostics%deflations(k) + 1
      end do
    end do
    roots = stored(:, initial_count + 1:count)
    if (present(origins)) origins = stored_origins(initial_count + 1:count)
  contains
    subroutine modified_residual(y, f, ok)
      real(dp), intent(in) :: y(:)
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: ok
      real(dp) :: distance2, log_factor
      integer :: root
      call residual(y, f, ok)
      if (.not. ok .or. .not. deflation) return
      log_factor = 0.0_dp
      do root = 1, count
        distance2 = sum((y - stored(:, root))**2)
        if (distance2 < 1e-20_dp) then
          ok = .false.
          return
        end if
        log_factor = log_factor + log(1.0_dp + 1.0_dp/distance2)
      end do
      ok = log_factor <= 200.0_dp
      if (ok) f = f*exp(log_factor)
    end subroutine
  end subroutine find_guarded_roots
end module sheath_model_continuation
