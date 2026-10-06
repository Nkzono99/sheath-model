! SPDX-License-Identifier: MIT
program test_scalar_search
  use sheath_model_constants, only: dp
  use sheath_model_search, only: sheath_search_options, sheath_search_diagnostics
  use sheath_model_scalar_search, only: find_scalar_roots
  implicit none
  type(sheath_search_options) :: options
  type(sheath_search_diagnostics) :: diagnostics
  real(dp), allocatable :: roots(:)
  integer :: problem
  options%residual_tolerance = 1e-14_dp
  options%max_iterations = 120
  ! Two roots lie inside one initial interval with the same endpoint signs.
  problem = 1
  call find_scalar_roots(residual, [0.0_dp, 1.0_dp], options, roots, diagnostics, 2, 16)
  call check(size(roots) == 2, 'narrow pair enumerated')
  call check(any(abs(roots - 0.37099_dp) < 1e-8_dp), 'left root')
  call check(any(abs(roots - 0.37101_dp) < 1e-8_dp), 'right root')
  ! A rejected mathematical root must not consume the accepted-root budget.
  diagnostics = sheath_search_diagnostics()
  call find_scalar_roots(residual, [0.0_dp, 1.0_dp], options, roots, diagnostics, 2, 1, accept_right)
  call check(size(roots) == 1, 'accepted-root budget')
  call check(abs(roots(1) - 0.37101_dp) < 1e-8_dp, 'search continues after rejection')
  ! A double root has no sign change.
  problem = 2
  diagnostics = sheath_search_diagnostics()
  call find_scalar_roots(residual, [0.0_dp, 1.0_dp], options, roots, diagnostics, 2, 16)
  call check(size(roots) == 1 .and. diagnostics%tangencies(2) > 0, 'tangential root')
  call check(abs(roots(1) - 0.371_dp) < 1e-8_dp, 'tangent location refined')
  problem = 6
  diagnostics = sheath_search_diagnostics()
  call find_scalar_roots(residual, [0.0_dp, 1.0_dp], options, roots, diagnostics, 2, 16)
  call check(size(roots) == 1 .and. diagnostics%tangencies(2) > 0, 'nonquadratic tangency')
  call check(abs(roots(1) - 0.371_dp) < 1e-8_dp, 'minimizer refines an interpolated extremum')
  ! Opposite signs across an invalid domain do not form a root.
  problem = 3
  diagnostics = sheath_search_diagnostics()
  call find_scalar_roots(residual, [0.0_dp, 1.0_dp], options, roots, diagnostics, 2, 16)
  call check(size(roots) == 0 .and. diagnostics%invalid_evaluations(2) > 0, 'invalid gap retained')
  ! A discontinuity never meets the original residual tolerance.
  problem = 4
  diagnostics = sheath_search_diagnostics()
  call find_scalar_roots(residual, [0.0_dp, 1.0_dp], options, roots, diagnostics, 2, 16)
  call check(size(roots) == 0 .and. diagnostics%unconverged(2) > 0, 'jump is unresolved, not a root')
  ! A small positive minimum is not a tangency.
  problem = 5
  diagnostics = sheath_search_diagnostics()
  call find_scalar_roots(residual, [0.0_dp, 1.0_dp], options, roots, diagnostics, 2, 16)
  call check(size(roots) == 0, 'stationary residual is not convergence')
  print *, 'Scalar pairs, tangencies and disconnected domains passed.'
contains
  subroutine accept_right(x, iterations, accepted)
    real(dp), intent(in) :: x
    integer, intent(in) :: iterations
    logical, intent(out) :: accepted
    call check(iterations > 0, 'refinement count passed to acceptance')
    accepted = x > 0.371_dp
  end subroutine
  subroutine residual(x, f, valid)
    real(dp), intent(in) :: x
    real(dp), intent(out) :: f
    logical, intent(out) :: valid
    valid = .true.
    select case (problem)
    case (1)
      f = (x - 0.371_dp)**2 - 1e-10_dp
    case (2)
      f = (x - 0.371_dp)**2
    case (3)
      valid = abs(x - 0.5_dp) > 0.05_dp
      f = x - 0.5_dp
    case (4)
      f = sign(1.0_dp, x - 0.371_dp)
    case (6)
      f = (x - 0.371_dp)**2*(1.0_dp + 0.5_dp*x)
    case default
      f = (x - 0.371_dp)**2 + 1e-8_dp
    end select
  end subroutine
  subroutine check(condition, message)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: message
    if (.not. condition) then
      print *, 'FAIL: ', message
      print *, 'roots=', roots
      error stop 1
    end if
  end subroutine
end program
