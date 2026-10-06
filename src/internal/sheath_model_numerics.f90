! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: extracted generic nonlinear solvers with residual callbacks.
!> Numerical algorithms without sheath parameters, branches, or status codes.
module sheath_model_numerics
  use sheath_model_constants, only: dp
  use sheath_model_search, only: sheath_search_options
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite

  implicit none

  private

  public :: solve_guarded_system, residual_norm, solve_guarded_linear_system

  abstract interface
    subroutine guarded_residual(x, f, valid)
      import :: dp
      real(dp), intent(in) :: x(:)
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: valid
    end subroutine guarded_residual
  end interface

contains

  !> Return the Euclidean norm of f, or huge() if any component is non-finite.
  real(dp) function residual_norm(f) result(norm2)
    real(dp), intent(in) :: f(:)

    if (.not. all(ieee_is_finite(f))) then
      norm2 = huge(1.0d0)
      return
    end if

    norm2 = sqrt(sum(f*f))
  end function residual_norm

  !> Newton with backtracking or Levenberg-Marquardt in dimensionless coordinates.
  !! auto tries an LM step when a Newton step cannot reduce the residual.
  !! Convergence always requires the residual tolerance, never a small step alone.
  subroutine solve_guarded_system(n, y0, residual_fn, options, y_out, final_norm, iterations, success, evaluations, lm_steps)
    integer, intent(in) :: n
    real(dp), intent(in) :: y0(n)
    procedure(guarded_residual) :: residual_fn
    type(sheath_search_options), intent(in) :: options
    real(dp), intent(out) :: y_out(n), final_norm
    integer, intent(out) :: iterations
    logical, intent(out) :: success
    integer, intent(out), optional :: evaluations, lm_steps

    real(dp) :: y(n), f(n), jac(n, n), delta(n), trial(n), trial_f(n), a(n, n), rhs(n)
    real(dp) :: norm, trial_norm, step, damping, diagonal(n)
    integer :: iteration, backtrack, attempt, i, neval, nlm
    logical :: valid, jacobian_ok, linear_ok, trial_valid, improved, use_lm

    neval = 0
    nlm = 0
    iterations = 0
    y = y0
    call counted_residual(y, f, valid)
    norm = huge(1.0_dp)
    success = .false.
    if (valid) then
      norm = maxval(abs(f))
      do iteration = 0, options%max_iterations
        iterations = iteration
        if (norm <= options%residual_tolerance) then
          success = .true.
          exit
        end if
        if (iteration == options%max_iterations) exit
        call guarded_numerical_jacobian(n, y, f, counted_residual, jac, jacobian_ok)
        if (.not. jacobian_ok) exit
        improved = .false.
        if (trim(options%method) /= 'lm') then
          call solve_guarded_linear_system(n, jac, -f, delta, linear_ok)
          if (linear_ok) call accept_step(delta, improved)
        end if
        use_lm = trim(options%method) == 'lm' .or. (trim(options%method) == 'auto' .and. .not. improved)
        if (use_lm) then
          a = matmul(transpose(jac), jac)
          rhs = -matmul(transpose(jac), f)
          do i = 1, n
            diagonal(i) = max(a(i, i), 1e-12_dp)
          end do
          damping = 1e-3_dp
          do attempt = 1, options%max_backtracks
            a = matmul(transpose(jac), jac)
            do i = 1, n
              a(i, i) = a(i, i) + damping*diagonal(i)
            end do
            call solve_guarded_linear_system(n, a, rhs, delta, linear_ok)
            if (linear_ok) call accept_step(delta, improved)
            if (improved) then
              nlm = nlm + 1
              exit
            end if
            damping = damping*10.0_dp
          end do
        end if
        if (.not. improved) exit
      end do
    end if
    y_out = y
    final_norm = norm
    if (present(evaluations)) evaluations = neval
    if (present(lm_steps)) lm_steps = nlm

  contains
    subroutine counted_residual(value, residual, ok)
      real(dp), intent(in) :: value(:)
      real(dp), intent(out) :: residual(:)
      logical, intent(out) :: ok
      neval = neval + 1
      call residual_fn(value, residual, ok)
      ok = ok .and. all(ieee_is_finite(residual))
    end subroutine counted_residual

    subroutine accept_step(direction, accepted)
      real(dp), intent(in) :: direction(n)
      logical, intent(out) :: accepted
      accepted = .false.
      step = 1.0_dp
      do backtrack = 1, options%max_backtracks
        trial = y + step*direction
        call counted_residual(trial, trial_f, trial_valid)
        if (trial_valid) then
          trial_norm = maxval(abs(trial_f))
          if (trial_norm < norm) then
            y = trial
            f = trial_f
            norm = trial_norm
            accepted = .true.
            return
          end if
        end if
        step = step*0.5_dp
      end do
    end subroutine accept_step
  end subroutine solve_guarded_system

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
