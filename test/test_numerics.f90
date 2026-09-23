! SPDX-License-Identifier: MIT
! Exercise numerical kernels with analytic problems independent of sheath physics.
program test_numerics
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
  use sheath_model_constants, only: dp
  use sheath_model_numerics, only: solve_nonlinear_system, try_guarded_newton_solve, residual_norm
  implicit none
  real(dp) :: x(3), scalar(1), norm, nan
  integer :: iterations
  logical :: success

  ! The zero leading diagonal requires pivoting; the exact solution is known.
  call solve_nonlinear_system(3, reshape([0.0_dp, 0.0_dp, 0.0_dp], [3, 1]), linear_residual, x, success)
  call check(success .and. maxval(abs(x - [1.0_dp, 2.0_dp, -1.0_dp])) < 1e-8_dp, 'pivoted linear system')
  call try_guarded_newton_solve(3, [0.0_dp, 0.0_dp, 0.0_dp], guarded_linear_residual, x, norm, iterations, success)
  call check(success .and. maxval(abs(x - [1.0_dp, 2.0_dp, -1.0_dp])) < 1e-8_dp, 'guarded pivoted system')

  ! The first start stalls; multistart must continue to the second guess.
  call solve_nonlinear_system(1, reshape([0.0_dp, 1.0_dp], [1, 2]), quadratic_residual, scalar, success)
  call check(success .and. abs(scalar(1) - sqrt(2.0_dp)) < 1e-9_dp, 'nonlinear multistart')
  call solve_nonlinear_system(1, reshape([0.0_dp], [1, 1]), constant_residual, scalar, success)
  call check(.not. success .and. all(ieee_is_finite(scalar)), 'singular Jacobian failure')

  ! At x=0 the central difference crosses the domain boundary; use one side.
  call try_guarded_newton_solve(1, [0.0_dp], square_root_residual, scalar, norm, iterations, success)
  call check(success .and. abs(scalar(1) - 4.0_dp) < 1e-8_dp, 'one-sided derivative at domain boundary')
  ! A full Newton step from x=10 leaves the log domain; backtracking recovers it.
  call try_guarded_newton_solve(1, [10.0_dp], logarithm_residual, scalar, norm, iterations, success)
  call check(success .and. abs(scalar(1) - 1.0_dp) < 1e-8_dp, 'backtracking rejects invalid trials')

  call try_guarded_newton_solve(1, [-1.0_dp], square_root_residual, scalar, norm, iterations, success)
  call check(.not. success .and. iterations == 0 .and. scalar(1) == -1.0_dp, 'invalid initial state')
  call check(norm == huge(1.0_dp), 'invalid state residual is unresolved')
  call try_guarded_newton_solve(1, [0.0_dp], isolated_residual, scalar, norm, iterations, success)
  call check(.not. success .and. iterations == 0 .and. norm == 1.0_dp, 'no valid derivative direction')

  nan = ieee_value(0.0_dp, ieee_quiet_nan)
  call check(residual_norm([3.0_dp, 4.0_dp]) == 5.0_dp, 'Euclidean norm')
  call check(residual_norm([nan]) == huge(1.0_dp), 'non-finite residual cannot converge')
  print *, 'Generic numerical kernel checks passed.'
contains
  subroutine linear_residual(value, residual)
    real(dp), intent(in) :: value(:)
    real(dp), intent(out) :: residual(:)
    residual(1) = 2.0_dp*value(2) + value(3) - 3.0_dp
    residual(2) = value(1) - value(2) + value(3) + 2.0_dp
    residual(3) = 2.0_dp*value(1) + value(2) + 3.0_dp*value(3) - 1.0_dp
  end subroutine linear_residual

  subroutine guarded_linear_residual(value, residual, valid)
    real(dp), intent(in) :: value(:)
    real(dp), intent(out) :: residual(:)
    logical, intent(out) :: valid
    call linear_residual(value, residual)
    valid = all(ieee_is_finite(residual))
  end subroutine guarded_linear_residual

  subroutine quadratic_residual(value, residual)
    real(dp), intent(in) :: value(:)
    real(dp), intent(out) :: residual(:)
    residual = value*value - 2.0_dp
  end subroutine quadratic_residual

  subroutine constant_residual(value, residual)
    real(dp), intent(in) :: value(:)
    real(dp), intent(out) :: residual(:)
    residual = 1.0_dp + 0.0_dp*value
  end subroutine constant_residual

  subroutine square_root_residual(value, residual, valid)
    real(dp), intent(in) :: value(:)
    real(dp), intent(out) :: residual(:)
    logical, intent(out) :: valid
    residual = 0.0_dp
    valid = all(value >= 0.0_dp)
    if (valid) residual = sqrt(value) - 2.0_dp
  end subroutine square_root_residual

  subroutine logarithm_residual(value, residual, valid)
    real(dp), intent(in) :: value(:)
    real(dp), intent(out) :: residual(:)
    logical, intent(out) :: valid
    residual = 0.0_dp
    valid = all(value > 0.0_dp)
    if (valid) residual = log(value)
  end subroutine logarithm_residual

  subroutine isolated_residual(value, residual, valid)
    real(dp), intent(in) :: value(:)
    real(dp), intent(out) :: residual(:)
    logical, intent(out) :: valid
    residual = 1.0_dp
    valid = all(value == 0.0_dp)
  end subroutine isolated_residual

  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, 'FAIL: ', label
      error stop 1
    end if
  end subroutine check
end program test_numerics
