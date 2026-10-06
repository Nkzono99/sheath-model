! SPDX-License-Identifier: MIT
!> Finite scalar root enumeration with domain guards and residual-based acceptance.
module sheath_model_scalar_search
  use sheath_model_constants, only: dp
  use sheath_model_search, only: sheath_search_options, sheath_search_diagnostics
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: find_scalar_roots
  abstract interface
    subroutine scalar_residual(x, f, valid)
      import :: dp
      real(dp), intent(in) :: x
      real(dp), intent(out) :: f
      logical, intent(out) :: valid
    end subroutine
    subroutine scalar_candidate(x, iterations, accepted)
      import :: dp
      real(dp), intent(in) :: x
      integer, intent(in) :: iterations
      logical, intent(out) :: accepted
    end subroutine
  end interface
contains
  !> Search ascending knots, including supplied nonsmooth boundaries.
  !! Midpoint curvature drives bounded subdivision. Quadratic extrema expose
  !! close pairs; minimization can locate tangencies without a sign change.
  !! Invalid samples split domains; neither a small interval nor a local minimum
  !! alone establishes a root. No finite sampling proves completeness.
  subroutine find_scalar_roots(residual_fn, knots, options, roots, diagnostics, branch_index, max_roots, candidate_fn)
    procedure(scalar_residual) :: residual_fn
    real(dp), intent(in) :: knots(:)
    type(sheath_search_options), intent(in) :: options
    real(dp), allocatable, intent(out) :: roots(:)
    type(sheath_search_diagnostics), intent(inout) :: diagnostics
    integer, intent(in) :: branch_index, max_roots
    procedure(scalar_candidate), optional :: candidate_fn
    real(dp), allocatable :: tested(:)
    real(dp) :: found(max_roots), left, fl, right, fr
    integer :: n, point, k
    logical :: vl, vr
    n = 0
    allocate (tested(0))
    k = branch_index
    diagnostics%starts(k) = diagnostics%starts(k) + 1
    if (size(knots) < 2) then
      allocate (roots(0))
      return
    end if
    left = knots(1)
    call evaluate(left, fl, vl)
    do point = 2, size(knots)
      right = knots(point)
      if (right <= left) cycle
      call evaluate(right, fr, vr)
      call inspect(left, fl, vl, right, fr, vr, 0)
      left = right
      fl = fr
      vl = vr
      if (n >= max_roots) exit
    end do
    roots = found(:n)
  contains
    subroutine evaluate(x, f, valid)
      real(dp), intent(in) :: x
      real(dp), intent(out) :: f
      logical, intent(out) :: valid
      call residual_fn(x, f, valid)
      diagnostics%evaluations(k) = diagnostics%evaluations(k) + 1
      valid = valid .and. ieee_is_finite(f)
      if (valid) then
        diagnostics%best_residual(k) = min(diagnostics%best_residual(k), abs(f))
      else
        diagnostics%invalid_evaluations(k) = diagnostics%invalid_evaluations(k) + 1
      end if
    end subroutine

    recursive subroutine inspect(a, fa, va, b, fb, vb, depth)
      real(dp), intent(in) :: a, fa, b, fb
      logical, intent(in) :: va, vb
      integer, intent(in) :: depth
      real(dp) :: m, fm, q, linear, t, v, fv, predicted, scale
      logical :: vm, vv, refine
      if (n >= max_roots) return
      m = 0.5_dp*a + 0.5_dp*b
      if (m == a .or. m == b) then
        if (va .and. vb) call bisect(a, fa, b, fb)
        return
      end if
      call evaluate(m, fm, vm)
      refine = depth < 1
      if (va .and. vm .and. vb) then
        scale = max(abs(fa), abs(fm), abs(fb), tiny(1.0_dp))
        refine = refine .or. abs(fm - 0.5_dp*fa - 0.5_dp*fb) > 0.1_dp*scale
      else
        refine = refine .or. ((va .or. vm .or. vb) .and. .not. (va .and. vm .and. vb))
      end if
      if (refine .and. depth < options%scalar_max_depth) then
        diagnostics%subdivisions(k) = diagnostics%subdivisions(k) + 1
        call inspect(a, fa, va, m, fm, vm, depth + 1)
        call inspect(m, fm, vm, b, fb, vb, depth + 1)
        return
      end if
      if (va .and. vm) call bisect(a, fa, m, fm)
      if (vm .and. vb) call bisect(m, fm, b, fb)
      if (.not. (va .and. vm .and. vb)) return
      if (crossing(fa, fm) .or. crossing(fm, fb)) return
      ! Interpolated extremum inside the interval, even when the sampled
      ! midpoint lies outside a narrow pair of roots.
      q = 2.0_dp*(fa + fb - 2.0_dp*fm)
      linear = fb - fa - q
      if (q /= 0.0_dp) then
        t = -linear/(2.0_dp*q)
        if (t > 0.0_dp .and. t < 1.0_dp) then
          predicted = fa + t*(linear + q*t)
          if (abs(predicted) < 0.5_dp*min(abs(fa), abs(fb))) then
            v = a + (b - a)*t
            call evaluate(v, fv, vv)
            if (.not. vv) return
            if (fv == 0.0_dp .and. sign(1.0_dp, fa) == sign(1.0_dp, fb)) then
              call remember(v, fv, .true.)
              return
            end if
            if (crossing(fa, fv) .or. crossing(fv, fb)) then
              call bisect(a, fa, v, fv)
              call bisect(v, fv, b, fb)
              return
            end if
            if (abs(fv) < min(abs(fa), abs(fb))) call minimize(a, fa, b, fb)
          end if
        end if
      end if
      if (abs(fm) < min(abs(fa), abs(fb))) call minimize(a, fa, b, fb)
    end subroutine

    logical function crossing(fa, fb)
      real(dp), intent(in) :: fa, fb
      crossing = fa == 0.0_dp .or. fb == 0.0_dp .or. sign(1.0_dp, fa) /= sign(1.0_dp, fb)
    end function

    subroutine remember(x, f, tangent, iterations)
      real(dp), intent(in) :: x, f
      logical, intent(in), optional :: tangent
      integer, intent(in), optional :: iterations
      integer :: j, count
      logical :: accepted
      if (abs(f) > options%residual_tolerance .or. n >= max_roots) return
      do j = 1, size(tested)
        if (abs(x - tested(j)) <= 1e-8_dp*max(1.0_dp, abs(x), abs(tested(j)))) return
      end do
      tested = [tested, x]
      if (present(candidate_fn)) then
        count = 0
        if (present(iterations)) count = iterations
        call candidate_fn(x, count, accepted)
        if (.not. accepted) return
      end if
      n = n + 1
      found(n) = x
      if (present(tangent)) then
        if (tangent) diagnostics%tangencies(k) = diagnostics%tangencies(k) + 1
      end if
    end subroutine

    subroutine bisect(a, fa, b, fb)
      real(dp), intent(in) :: a, fa, b, fb
      real(dp) :: lo, hi, flo, mid, fmid
      integer :: iteration
      logical :: valid
      if (.not. crossing(fa, fb) .or. n >= max_roots) return
      diagnostics%brackets(k) = diagnostics%brackets(k) + 1
      if (abs(fa) <= options%residual_tolerance) then
        call remember(a, fa)
        return
      end if
      if (abs(fb) <= options%residual_tolerance) then
        call remember(b, fb)
        return
      end if
      lo = a
      hi = b
      flo = fa
      do iteration = 1, options%max_iterations
        diagnostics%iterations(k) = diagnostics%iterations(k) + 1
        mid = 0.5_dp*lo + 0.5_dp*hi
        if (mid == lo .or. mid == hi) exit
        call evaluate(mid, fmid, valid)
        if (.not. valid) exit ! never bridge an invalid gap
        if (abs(fmid) <= options%residual_tolerance) then
          call remember(mid, fmid, iterations=iteration)
          return
        end if
        if (crossing(flo, fmid)) then
          hi = mid
        else
          lo = mid
          flo = fmid
        end if
      end do
      diagnostics%unconverged(k) = diagnostics%unconverged(k) + 1
    end subroutine

    subroutine minimize(a, fa, b, fb)
      real(dp), intent(in) :: a, fa, b, fb
      real(dp), parameter :: fraction = 0.3819660112501051_dp
      real(dp) :: lo, hi, x1, x2, f1, f2, best_x, best_f
      integer :: iteration
      logical :: valid1, valid2
      lo = a
      hi = b
      x1 = lo + fraction*(hi - lo)
      x2 = hi - fraction*(hi - lo)
      call evaluate(x1, f1, valid1)
      call evaluate(x2, f2, valid2)
      if (.not. (valid1 .and. valid2)) return
      best_x = x1
      best_f = f1
      do iteration = 1, options%max_iterations
        diagnostics%iterations(k) = diagnostics%iterations(k) + 1
        if (abs(f1) < abs(best_f)) then
          best_x = x1
          best_f = f1
        end if
        if (abs(f2) < abs(best_f)) then
          best_x = x2
          best_f = f2
        end if
        if (f1 /= 0.0_dp .and. f2 /= 0.0_dp .and. sign(1.0_dp, f1) /= sign(1.0_dp, f2)) then
          ! The minimum is a close pair, rather than a tangency.
          if (crossing(fa, f1)) then
            call bisect(a, fa, x1, f1)
            call bisect(x1, f1, b, fb)
          else
            call bisect(a, fa, x2, f2)
            call bisect(x2, f2, b, fb)
          end if
          return
        end if
        if (abs(f1) <= abs(f2)) then
          hi = x2
          x2 = x1
          f2 = f1
          x1 = lo + fraction*(hi - lo)
          if (x1 == lo .or. x1 == x2) exit
          call evaluate(x1, f1, valid1)
          if (.not. valid1) return
        else
          lo = x1
          x1 = x2
          f1 = f2
          x2 = hi - fraction*(hi - lo)
          if (x2 == hi .or. x2 == x1) exit
          call evaluate(x2, f2, valid2)
          if (.not. valid2) return
        end if
      end do
      call remember(best_x, best_f, .true., min(iteration, options%max_iterations))
    end subroutine

  end subroutine
end module
