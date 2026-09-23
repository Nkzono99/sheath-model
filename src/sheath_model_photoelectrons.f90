! SPDX-License-Identifier: MIT
!> Boundary photoelectron distributions and orbit-conserving moments.
module sheath_model_photoelectrons
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sheath_model_constants, only: dp, i32, pi, qe
  use sheath_model_status, only: SHEATH_OK, SHEATH_INVALID_ARGUMENT
  implicit none
  private
  public :: photoelectron_source, maxwellian_photoelectrons, binned_photoelectrons
  public :: validate_photoelectrons, photoelectron_density, photoelectron_fluxes
  public :: photoelectron_density_integral, photoelectron_sqrt_coefficient

  !> Outward boundary distribution: analytic Maxwellian or piecewise constant dGamma/dK.
  !! Construct with maxwellian_photoelectrons or binned_photoelectrons; no history is stored.
  type :: photoelectron_source
    private
    logical :: binned = .false.
    real(dp) :: density_m3 = 0.0_dp
    real(dp) :: temperature_ev = 2.2_dp
    real(dp), allocatable :: edges_ev(:), flux_m2_s(:)
  contains
    procedure :: is_binned
    procedure :: potential_scale
  end type

contains

  !> Construct an analytic source from Maxwellian normalization [m^-3] and temperature [eV].
  !! Validity is checked by the model entry points, including zero emission.
  pure function maxwellian_photoelectrons(density_m3, temperature_ev) result(source)
    real(dp), intent(in) :: density_m3, temperature_ev
    type(photoelectron_source) :: source
    source%density_m3 = density_m3
    source%temperature_ev = temperature_ev
  end function

  !> Construct a source from ascending normal-energy edges [eV] and integrated bin fluxes [m^-2 s^-1].
  !! There must be one more edge than flux. dGamma/dK is constant in each bin and zero outside the edges.
  pure function binned_photoelectrons(energy_edges_ev, bin_flux_m2_s) result(source)
    real(dp), intent(in) :: energy_edges_ev(:), bin_flux_m2_s(:)
    type(photoelectron_source) :: source
    source%binned = .true.
    source%edges_ev = energy_edges_ev
    source%flux_m2_s = bin_flux_m2_s
  end function

  !> True for a piecewise constant energy-flux source.
  pure logical function is_binned(self)
    class(photoelectron_source), intent(in) :: self
    is_binned = self%binned
  end function

  !> Numerical potential scale [V]: Maxwellian temperature, or a power of two near the supplied spectral scale.
  !! Binary scaling preserves physical bin edges exactly when converting potentials to and from internal units.
  pure real(dp) function potential_scale(self, spectrum_scale_v) result(scale)
    class(photoelectron_source), intent(in) :: self
    real(dp), intent(in) :: spectrum_scale_v
    scale = self%temperature_ev
    if (self%binned) then
      scale = 2.0_dp**(exponent(spectrum_scale_v) - 1)
    end if
  end function

  !> Validate finite nonnegative source strength, positive Maxwellian temperature, and strict bin ordering.
  subroutine validate_photoelectrons(source, status, message)
    type(photoelectron_source), intent(in) :: source
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    integer :: n
    status = SHEATH_INVALID_ARGUMENT
    message = 'Invalid Maxwellian photoelectron density or temperature.'
    if (.not. source%binned) then
      if (.not. all(ieee_is_finite([source%density_m3, source%temperature_ev]))) return
      if (source%density_m3 < 0.0_dp .or. source%temperature_ev <= 0.0_dp) return
    else
      message = 'Spectrum needs N+1 finite, nonnegative ascending edges and N nonnegative integrated fluxes.'
      if (.not. allocated(source%edges_ev) .or. .not. allocated(source%flux_m2_s)) return
      n = size(source%flux_m2_s)
      if (n < 1 .or. size(source%edges_ev) /= n + 1) return
      if (.not. all(ieee_is_finite(source%edges_ev)) .or. .not. all(ieee_is_finite(source%flux_m2_s))) return
      if (any(source%edges_ev < 0.0_dp) .or. any(source%flux_m2_s < 0.0_dp)) return
      if (any(source%edges_ev(2:) <= source%edges_ev(:n))) return
      if (.not. ieee_is_finite(sum(source%flux_m2_s))) return
      if (.not. all(ieee_is_finite(source%flux_m2_s/(source%edges_ev(2:) - source%edges_ev(:n))))) return
    end if
    status = SHEATH_OK
    message = ''
  end subroutine

  !> Return outward, escaping and returning number fluxes [m^-2 s^-1] for a validated source.
  !! barrier_v >= 0 is the boundary-to-minimum potential drop [V]; return counts one inward crossing.
  pure subroutine photoelectron_fluxes(source, mass_kg, barrier_v, outward, escape, returning)
    type(photoelectron_source), intent(in) :: source
    real(dp), intent(in) :: mass_kg, barrier_v
    real(dp), intent(out) :: outward, escape, returning
    real(dp) :: a, b
    integer :: i
    if (source%binned) then
      outward = sum(source%flux_m2_s)
      escape = 0.0_dp
      returning = 0.0_dp
      do i = 1, size(source%flux_m2_s)
        a = source%edges_ev(i)
        b = source%edges_ev(i + 1)
        escape = escape + source%flux_m2_s(i)*max(0.0_dp, b - max(a, barrier_v))/(b - a)
        returning = returning + source%flux_m2_s(i)*max(0.0_dp, min(b, barrier_v) - a)/(b - a)
      end do
    else
      outward = source%density_m3*sqrt(2.0_dp*qe*source%temperature_ev/mass_kg)/(2.0_dp*sqrt(pi))
      escape = outward*exp(-barrier_v/source%temperature_ev)
      returning = outward*one_minus_exp(-barrier_v/source%temperature_ev)
    end if
  end subroutine

  !> Return escaping and captured densities [m^-3] at phi_v for a validated source and physical orbit domain.
  !! Potentials are in V with minimum_v <= min(boundary_v,0); lower_side includes both legs of returning orbits.
  pure subroutine photoelectron_density(source, mass_kg, boundary_v, minimum_v, phi_v, lower_side, free, captured)
    type(photoelectron_source), intent(in) :: source
    real(dp), intent(in) :: mass_kg, boundary_v, minimum_v, phi_v
    logical, intent(in) :: lower_side
    real(dp), intent(out) :: free, captured
    real(dp) :: barrier, shift, a, b, g, velocity, x
    integer :: i
    free = 0.0_dp
    captured = 0.0_dp
    barrier = boundary_v - minimum_v
    if (.not. source%binned) then
      x = sqrt(max(0.0_dp, (phi_v - minimum_v)/source%temperature_ev))
      free = 0.5_dp*source%density_m3*exp(-barrier/source%temperature_ev)*erfc_scaled(x)
      if (lower_side) then
        captured = source%density_m3*exp((phi_v - boundary_v)/source%temperature_ev)*erf(x)
      end if
      return
    end if
    velocity = sqrt(2.0_dp*qe/mass_kg)
    shift = boundary_v - phi_v
    do i = 1, size(source%flux_m2_s)
      a = source%edges_ev(i)
      b = source%edges_ev(i + 1)
      g = source%flux_m2_s(i)/(b - a)
      free = free + 2.0_dp*g/velocity*root_interval(max(a, barrier), b, shift)
      if (lower_side) then
        captured = captured + 4.0_dp*g/velocity*root_interval(a, min(b, barrier), shift)
      end if
    end do
  end subroutine

  !> Analytic integral of binned PE number density over potential [m^-3 V], with the same orbit populations.
  !! Requires a validated binned source; the Maxwellian path is integrated with the other smooth densities.
  pure real(dp) function photoelectron_density_integral(source, mass_kg, boundary_v, minimum_v, lo_v, hi_v, lower_side) &
      result(value)
    type(photoelectron_source), intent(in) :: source
    real(dp), intent(in) :: mass_kg, boundary_v, minimum_v, lo_v, hi_v
    logical, intent(in) :: lower_side
    real(dp) :: a, b, g, barrier, lo, hi, velocity
    integer :: i
    value = 0.0_dp
    if (lo_v == hi_v) return
    lo = min(lo_v, hi_v)
    hi = max(lo_v, hi_v)
    barrier = boundary_v - minimum_v
    velocity = sqrt(2.0_dp*qe/mass_kg)
    do i = 1, size(source%flux_m2_s)
      a = source%edges_ev(i)
      b = source%edges_ev(i + 1)
      g = source%flux_m2_s(i)/(b - a)
      value = value + g*bin_integral(max(a, barrier), b, boundary_v, lo, hi)
      if (lower_side) then
        value = value + 2.0_dp*g*bin_integral(a, min(b, barrier), boundary_v, lo, hi)
      end if
    end do
    value = sign(1.0_dp, hi_v - lo_v)*value/velocity
  end function

  !> PE coefficient of sqrt(phi) in Type-B density [m^-3 V^-1/2] near upstream phi=0+.
  !! At an energy-bin edge, 2*g_left-g_right accounts separately for returning and escaping orbits.
  pure real(dp) function photoelectron_sqrt_coefficient(source, mass_kg, boundary_v) result(coefficient)
    type(photoelectron_source), intent(in) :: source
    real(dp), intent(in) :: mass_kg, boundary_v
    real(dp) :: left, right, a, b, g
    integer :: i
    if (.not. source%binned) then
      coefficient = source%density_m3*exp(-boundary_v/source%temperature_ev)/sqrt(pi*source%temperature_ev)
      return
    end if
    left = 0.0_dp
    right = 0.0_dp
    do i = 1, size(source%flux_m2_s)
      a = source%edges_ev(i)
      b = source%edges_ev(i + 1)
      g = source%flux_m2_s(i)/(b - a)
      if (a < boundary_v .and. boundary_v <= b) then
        left = g
      end if
      if (a <= boundary_v .and. boundary_v < b) then
        right = g
      end if
    end do
    coefficient = 2.0_dp*(2.0_dp*left - right)/sqrt(2.0_dp*qe/mass_kg)
  end function

  pure real(dp) function root_interval(lo, hi, shift) result(value)
    real(dp), intent(in) :: lo, hi, shift
    real(dp) :: a
    a = max(lo, shift)
    value = 0.0_dp
    if (hi <= a) return
    value = (hi - a)/(sqrt(max(0.0_dp, hi - shift)) + sqrt(max(0.0_dp, a - shift)))
  end function

  ! Difference (x+h)^(3/2)-x^(3/2), avoiding subtraction of close powers.
  pure real(dp) function power_difference(x, h) result(value)
    real(dp), intent(in) :: x, h
    real(dp) :: a, b
    a = sqrt(max(0.0_dp, x))
    b = sqrt(max(0.0_dp, x + h))
    value = 0.0_dp
    if (h > 0.0_dp) then
      value = h*(a*a + a*b + b*b)/(a + b)
    end if
  end function

  ! Mixed difference of x^(3/2), stable even for narrow bins and short potential intervals.
  pure real(dp) function mixed_power_difference(x, w, h) result(value)
    real(dp), intent(in) :: x, w, h
    real(dp) :: a, b, c, d, db, dd
    a = sqrt(max(0.0_dp, x))
    b = sqrt(max(0.0_dp, x + w))
    c = sqrt(max(0.0_dp, x + h))
    d = sqrt(max(0.0_dp, x + w + h))
    db = w/(b + a)
    dd = w/(d + c)
    value = h*(dd + db - (c*c*db + a*a*dd + (c + a)*dd*db)/((d + b)*(c + a)))
  end function

  pure real(dp) function bin_integral(klo, khi, boundary, lo, hi) result(value)
    real(dp), intent(in) :: klo, khi, boundary, lo, hi
    real(dp) :: a, b, x
    value = 0.0_dp
    if (khi <= klo) return
    a = max(lo, boundary - khi)
    b = min(hi, boundary - klo)
    if (b > a) then
      value = power_difference(max(0.0_dp, khi - boundary + a), b - a)
    end if
    a = max(lo, boundary - klo)
    if (hi > a) then
      x = max(0.0_dp, klo - boundary + a)
      value = value + mixed_power_difference(x, khi - klo, hi - a)
    end if
    value = (4.0_dp/3.0_dp)*value
  end function

  pure real(dp) function one_minus_exp(x) result(value)
    real(dp), intent(in) :: x
    if (abs(x) < 1e-5_dp) then
      value = -x*(1.0_dp + x*(0.5_dp + x*(1.0_dp/6.0_dp + x/24.0_dp)))
    else
      value = 1.0_dp - exp(x)
    end if
  end function
end module
