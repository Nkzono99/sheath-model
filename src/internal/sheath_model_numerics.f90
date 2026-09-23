! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: extracted generic nonlinear solvers with residual callbacks.
!> Numerical algorithms without sheath parameters, branches, or status codes.
module sheath_model_numerics
  use sheath_model_constants, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite

  implicit none

  private

  public :: solve_nonlinear_system, try_guarded_newton_solve, residual_norm, NONLINEAR_TOL

  real(dp), parameter :: NONLINEAR_TOL = 1.0d-10
  integer, parameter :: NONLINEAR_MAX_ITER = 60
  integer, parameter :: NONLINEAR_MAX_BACKTRACK = 20
  integer, parameter :: GUARDED_MAX_ITERATIONS = 60
  integer, parameter :: GUARDED_MAX_BACKTRACKS = 24
  real(dp), parameter :: GUARDED_TOLERANCE = 1.0e-9_dp

  abstract interface
    subroutine nonlinear_residual(x, f)
      import :: dp
      real(dp), intent(in) :: x(:)
      real(dp), intent(out) :: f(:)
    end subroutine nonlinear_residual

    subroutine guarded_residual(x, f, valid)
      import :: dp
      real(dp), intent(in) :: x(:)
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: valid
    end subroutine guarded_residual
  end interface

contains

  ! Forward differences and Euclidean norm for unconstrained residual callbacks.
  subroutine solve_nonlinear_system(n, guesses, residual_fn, x_best, success)
    integer, intent(in) :: n
    real(dp), intent(in) :: guesses(:, :)
    procedure(nonlinear_residual) :: residual_fn
    real(dp), intent(out) :: x_best(n)
    logical, intent(out) :: success

    integer :: guess_idx
    real(dp) :: x_trial(n), best_norm, trial_norm
    logical :: trial_success

    success = .false.
    x_best = 0.0_dp
    if (size(guesses, 1) /= n .or. size(guesses, 2) == 0) return

    x_best = guesses(:, 1)
    best_norm = huge(1.0d0)
    do guess_idx = 1, size(guesses, 2)
      call try_newton_solve(n, guesses(:, guess_idx), residual_fn, x_trial, trial_norm, trial_success)
      if (trial_norm < best_norm) then
        best_norm = trial_norm
        x_best = x_trial
      end if
      if (trial_success .and. trial_norm < NONLINEAR_TOL) then
        success = .true.
        x_best = x_trial
        return
      end if
    end do

    success = best_norm < NONLINEAR_TOL
  end subroutine solve_nonlinear_system

  subroutine try_newton_solve( &
      n, x0, residual_fn, &
      x_out, final_norm, success &
      )
    integer, intent(in) :: n
    real(dp), intent(in) :: x0(n)
    procedure(nonlinear_residual) :: residual_fn
    real(dp), intent(out) :: x_out(n)
    real(dp), intent(out) :: final_norm
    logical, intent(out) :: success

    integer :: iter, backtrack
    real(dp) :: x(n), f(n), jac(n, n), dx(n), x_trial(n), f_trial(n), step_scale, fnorm, trial_norm
    logical :: linear_ok, improved

    x = x0
    call residual_fn(x, f)
    fnorm = residual_norm(f)

    do iter = 1, NONLINEAR_MAX_ITER
      if (fnorm < NONLINEAR_TOL) exit

      call numerical_jacobian(n, x, f, residual_fn, jac)
      call solve_small_linear_system(n, jac, -f, dx, linear_ok)
      if (.not. linear_ok) exit

      step_scale = 1.0d0
      improved = .false.
      do backtrack = 1, NONLINEAR_MAX_BACKTRACK
        x_trial = x + step_scale*dx
        call residual_fn(x_trial, f_trial)
        trial_norm = residual_norm(f_trial)
        if (trial_norm < fnorm) then
          x = x_trial
          f = f_trial
          fnorm = trial_norm
          improved = .true.
          exit
        end if
        step_scale = 0.5d0*step_scale
      end do
      if (.not. improved) exit
    end do

    x_out = x
    final_norm = fnorm
    success = fnorm < NONLINEAR_TOL
  end subroutine try_newton_solve

  subroutine numerical_jacobian(n, x, f0, residual_fn, jac)
    integer, intent(in) :: n
    real(dp), intent(in) :: x(n), f0(n)
    procedure(nonlinear_residual) :: residual_fn
    real(dp), intent(out) :: jac(n, n)

    integer :: j
    real(dp) :: h, xh(n), fh(n)

    do j = 1, n
      h = 1.0d-6*max(1.0d0, abs(x(j)))
      xh = x
      xh(j) = xh(j) + h
      call residual_fn(xh, fh)
      jac(:, j) = (fh - f0)/h
    end do
  end subroutine numerical_jacobian

  subroutine solve_small_linear_system(n, a_in, b_in, x, ok)
    integer, intent(in) :: n
    real(dp), intent(in) :: a_in(n, n), b_in(n)
    real(dp), intent(out) :: x(n)
    logical, intent(out) :: ok

    integer :: i, j, k, pivot_row
    real(dp) :: a(n, n), b(n), factor, pivot_abs, tmp_row(n), tmp_val

    a = a_in
    b = b_in
    ok = .true.

    do k = 1, n
      pivot_row = k
      pivot_abs = abs(a(k, k))
      do i = k + 1, n
        if (abs(a(i, k)) > pivot_abs) then
          pivot_abs = abs(a(i, k))
          pivot_row = i
        end if
      end do
      if (pivot_abs <= 1.0d-18) then
        ok = .false.
        x = 0.0d0
        return
      end if

      if (pivot_row /= k) then
        tmp_row = a(k, :)
        a(k, :) = a(pivot_row, :)
        a(pivot_row, :) = tmp_row
        tmp_val = b(k)
        b(k) = b(pivot_row)
        b(pivot_row) = tmp_val
      end if

      do i = k + 1, n
        factor = a(i, k)/a(k, k)
        a(i, k:n) = a(i, k:n) - factor*a(k, k:n)
        b(i) = b(i) - factor*b(k)
      end do
    end do

    x = 0.0d0
    do i = n, 1, -1
      x(i) = b(i)
      do j = i + 1, n
        x(i) = x(i) - a(i, j)*x(j)
      end do
      x(i) = x(i)/a(i, i)
    end do
  end subroutine solve_small_linear_system

  real(dp) function residual_norm(f) result(norm2)
    real(dp), intent(in) :: f(:)

    if (.not. all(ieee_is_finite(f))) then
      norm2 = huge(1.0d0)
      return
    end if

    norm2 = sqrt(sum(f*f))
  end function residual_norm

  ! Central differences (one-sided at domain limits) and maximum residual norm.
  subroutine try_guarded_newton_solve( &
      n, y0, residual_fn, &
      y_out, final_norm, iterations, &
      success &
      )
    integer, intent(in) :: n
    real(dp), intent(in) :: y0(n)
    procedure(guarded_residual) :: residual_fn
    real(dp), intent(out) :: y_out(n), final_norm
    integer, intent(out) :: iterations
    logical, intent(out) :: success

    real(dp) :: y(n), f(n), jac(n, n), delta(n), trial(n), trial_f(n)
    real(dp) :: norm, trial_norm, step
    integer :: iteration, backtrack
    logical :: valid, jacobian_ok, linear_ok, trial_valid

    y = y0
    call residual_fn(y, f, valid)
    if (.not. valid) then
      y_out = y
      final_norm = huge(1.0_dp)
      iterations = 0
      success = .false.
      return
    end if

    norm = maxval(abs(f(1:n)))
    success = .false.
    do iteration = 0, GUARDED_MAX_ITERATIONS
      if (norm <= GUARDED_TOLERANCE) then
        success = .true.
        exit
      end if
      if (iteration == GUARDED_MAX_ITERATIONS) exit

      call guarded_numerical_jacobian(n, y, f, residual_fn, jac, jacobian_ok)
      if (.not. jacobian_ok) exit
      call solve_guarded_linear_system(n, jac, -f, delta, linear_ok)
      if (.not. linear_ok) exit

      step = 1.0_dp
      do backtrack = 1, GUARDED_MAX_BACKTRACKS
        trial = y + step*delta
        call residual_fn(trial, trial_f, trial_valid)
        if (trial_valid) then
          trial_norm = maxval(abs(trial_f(1:n)))
          if (trial_norm < norm) then
            y = trial
            f = trial_f
            norm = trial_norm
            exit
          end if
        end if
        step = 0.5_dp*step
      end do
      if (backtrack > GUARDED_MAX_BACKTRACKS) exit
    end do

    y_out = y
    final_norm = norm
    iterations = iteration
  end subroutine try_guarded_newton_solve

  subroutine guarded_numerical_jacobian(n, y, f0, residual_fn, jac, success)
    integer, intent(in) :: n
    real(dp), intent(in) :: y(n), f0(n)
    procedure(guarded_residual) :: residual_fn
    real(dp), intent(out) :: jac(n, n)
    logical, intent(out) :: success

    real(dp) :: yp(n), ym(n), fp(n), fm(n), h
    integer :: column
    logical :: plus_valid, minus_valid

    jac = 0.0_dp
    success = .true.
    do column = 1, n
      h = epsilon(1.0_dp)**(1.0_dp/3.0_dp)*max(1.0_dp, abs(y(column)))
      yp = y
      ym = y
      yp(column) = yp(column) + h
      ym(column) = ym(column) - h
      call residual_fn(yp, fp, plus_valid)
      call residual_fn(ym, fm, minus_valid)

      if (plus_valid .and. minus_valid) then
        jac(1:n, column) = (fp(1:n) - fm(1:n))/(2.0_dp*h)
      else if (plus_valid) then
        jac(1:n, column) = (fp(1:n) - f0(1:n))/h
      else if (minus_valid) then
        jac(1:n, column) = (f0(1:n) - fm(1:n))/h
      else
        success = .false.
        return
      end if
    end do

    success = all(ieee_is_finite(jac(1:n, 1:n)))
  end subroutine guarded_numerical_jacobian

  subroutine solve_guarded_linear_system(n, a_in, b_in, x, success)
    integer, intent(in) :: n
    real(dp), intent(in) :: a_in(n, n), b_in(n)
    real(dp), intent(out) :: x(n)
    logical, intent(out) :: success

    real(dp) :: a(n, n), b(n), factor, pivot_value, tmp
    integer :: i, j, k, pivot

    a = a_in
    b = b_in
    x = 0.0_dp
    success = .false.

    do k = 1, n
      pivot = k
      do i = k + 1, n
        if (abs(a(i, k)) > abs(a(pivot, k))) then
          pivot = i
        end if
      end do
      if (.not. ieee_is_finite(a(pivot, k)) .or. abs(a(pivot, k)) <= 1.0e-14_dp) return

      if (pivot /= k) then
        do j = k, n
          tmp = a(k, j)
          a(k, j) = a(pivot, j)
          a(pivot, j) = tmp
        end do
        tmp = b(k)
        b(k) = b(pivot)
        b(pivot) = tmp
      end if

      pivot_value = a(k, k)
      do i = k + 1, n
        factor = a(i, k)/pivot_value
        a(i, k:n) = a(i, k:n) - factor*a(k, k:n)
        b(i) = b(i) - factor*b(k)
      end do
    end do

    do i = n, 1, -1
      x(i) = b(i)
      do j = i + 1, n
        x(i) = x(i) - a(i, j)*x(j)
      end do
      x(i) = x(i)/a(i, i)
    end do

    success = all(ieee_is_finite(x(1:n)))
  end subroutine solve_guarded_linear_system

end module sheath_model_numerics
