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
  public :: photoelectron_upper_delta_scaled
  public :: log_energy_edges, energy_bin, observed_outflow_source
  public :: write_photoelectron_source, read_photoelectron_source

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
    procedure :: search_breakpoints
    procedure :: potential_scale
    procedure :: atlas_shape
    procedure :: with_outward_flux
    procedure :: mean_normal_energy
  end type

  !> Immutable zero-emission default for opaque sources nested in physical inputs.
  !! Explicit component initialization keeps enclosing constructors portable across compilers.
  type(photoelectron_source), parameter, public :: DEFAULT_PHOTOELECTRONS = photoelectron_source()

contains

  !> Dimensionless spectral shape for identifying compatible atlas entries.
  !! Maxwellians use an empty shape; their temperature is the potential scale.
  pure function atlas_shape(self, potential_scale_v) result(shape)
    class(photoelectron_source), intent(in) :: self
    real(dp), intent(in) :: potential_scale_v
    real(dp), allocatable :: shape(:)
    real(dp) :: total
    integer :: n
    allocate (shape(0))
    if (.not. self%binned) return
    n = size(self%flux_m2_s)
    deallocate (shape)
    allocate (shape(2*n + 1))
    shape(:n + 1) = self%edges_ev/potential_scale_v
    shape(n + 2:) = 0.0_dp
    total = sum(self%flux_m2_s)
    if (total > 0.0_dp) shape(n + 2:) = self%flux_m2_s/total
  end function atlas_shape

  !> Change source amplitude while preserving its energy distribution.
  pure function with_outward_flux(self, total_flux_m2_s, mass_kg) result(source)
    class(photoelectron_source), intent(in) :: self
    real(dp), intent(in) :: total_flux_m2_s, mass_kg
    type(photoelectron_source) :: source
    real(dp) :: total
    source = self
    if (self%binned) then
      total = sum(self%flux_m2_s)
      if (total > 0.0_dp) source%flux_m2_s = self%flux_m2_s*(total_flux_m2_s/total)
    else
      source%density_m3 = total_flux_m2_s*2.0_dp*sqrt(pi)/sqrt(2.0_dp*qe*self%temperature_ev/mass_kg)
    end if
  end function with_outward_flux

  !> Flux-weighted mean outward normal energy [eV]: the temperature for a half-Maxwellian,
  !! bin midpoints for a spectrum (dGamma/dK is constant in each bin), and 0 for an empty spectrum.
  pure real(dp) function mean_normal_energy(self) result(energy_ev)
    class(photoelectron_source), intent(in) :: self
    real(dp) :: total
    integer :: n
    energy_ev = self%temperature_ev
    if (.not. self%binned) return
    energy_ev = 0.0_dp
    n = size(self%flux_m2_s)
    total = sum(self%flux_m2_s)
    if (total > 0.0_dp) energy_ev = sum(self%flux_m2_s*0.5_dp*(self%edges_ev(:n) + self%edges_ev(2:)))/total
  end function mean_normal_energy

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

  !> Normal-energy edges [eV]: [0, lowest) followed by bins_per_decade log bins per decade up to lowest*10**decades.
  !! Defaults give 1 meV to 1 keV with 24 bins per decade (145 bins). Nonpositive arguments return no edges.
  pure function log_energy_edges(lowest_edge_ev, decades, bins_per_decade) result(edges_ev)
    real(dp), intent(in), optional :: lowest_edge_ev
    integer, intent(in), optional :: decades, bins_per_decade
    real(dp), allocatable :: edges_ev(:)
    real(dp) :: lowest
    integer :: n_decades, per_decade, j
    lowest = 1.0e-3_dp
    n_decades = 6
    per_decade = 24
    if (present(lowest_edge_ev)) lowest = lowest_edge_ev
    if (present(decades)) n_decades = decades
    if (present(bins_per_decade)) per_decade = bins_per_decade
    allocate (edges_ev(0))
    if (.not. ieee_is_finite(lowest) .or. lowest <= 0.0_dp .or. n_decades < 1 .or. per_decade < 1) return
    deallocate (edges_ev)
    allocate (edges_ev(n_decades*per_decade + 2))
    edges_ev(1) = 0.0_dp
    do j = 0, n_decades*per_decade
      edges_ev(j + 2) = lowest*10.0_dp**(real(j, dp)/real(per_decade, dp))
    end do
  end function log_energy_edges

  !> Bin of a normal kinetic energy [eV] on ascending edges; values outside the edges (and NaN) go to the end bins.
  !! Returns 0 when there are no bins.
  pure integer function energy_bin(edges_ev, energy_ev) result(bin)
    real(dp), intent(in) :: edges_ev(:), energy_ev
    integer :: n, low, high, middle
    n = size(edges_ev) - 1
    bin = 0
    if (n < 1) return
    bin = 1
    if (n == 1 .or. .not. energy_ev >= edges_ev(2)) return
    bin = n
    if (energy_ev >= edges_ev(n)) return
    ! Invariant: edges(low) <= energy < edges(high).
    low = 2
    high = n
    do while (high - low > 1)
      middle = (low + high)/2
      if (energy_ev >= edges_ev(middle)) then
        low = middle
      else
        high = middle
      end if
    end do
    bin = low
  end function energy_bin

  !> Build a binned emitted source from the outward flux spectrum [m^-2 s^-1 per bin] observed on a plane.
  !! Empty end bins are trimmed and interior empty bins kept. The total must be positive.
  subroutine observed_outflow_source(edges_ev, flux_m2_s, source, status, message)
    real(dp), intent(in) :: edges_ev(:), flux_m2_s(:)
    type(photoelectron_source), intent(out) :: source
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    integer :: n, first, last
    status = SHEATH_INVALID_ARGUMENT
    message = 'Outflow spectrum needs N+1 edges and N finite, nonnegative fluxes with a positive total.'
    n = size(flux_m2_s)
    if (n < 1 .or. size(edges_ev) /= n + 1) return
    if (.not. all(ieee_is_finite(flux_m2_s)) .or. any(flux_m2_s < 0.0_dp)) return
    if (.not. sum(flux_m2_s) > 0.0_dp) return
    first = 1
    do while (flux_m2_s(first) <= 0.0_dp)
      first = first + 1
    end do
    last = n
    do while (flux_m2_s(last) <= 0.0_dp)
      last = last - 1
    end do
    source = binned_photoelectrons(edges_ev(first:last + 1), flux_m2_s(first:last))
    call validate_photoelectrons(source, status, message)
  end subroutine observed_outflow_source

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

  !> Exact normal-energy bin edges for numerical searches; empty for a Maxwellian.
  pure function search_breakpoints(self) result(edges)
    class(photoelectron_source), intent(in) :: self
    real(dp), allocatable :: edges(:)
    allocate (edges(0))
    if (allocated(self%edges_ev)) edges = self%edges_ev
  end function search_breakpoints

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

  !> [n_PE(-depth*f)-n_PE(0)]/sqrt(depth) on an A upper segment [m^-3 V^-1/2].
  !! The Maxwellian expression is for depth/T << 1; bin differences are exact.
  !! Factored square-root differences retain small depths and narrow bin widths.
  pure real(dp) function photoelectron_upper_delta_scaled(source, mass_kg, boundary, depth, f) result(value)
    type(photoelectron_source), intent(in) :: source
    real(dp), intent(in) :: mass_kg, boundary, depth, f
    real(dp) :: a, b, sa, sb, ta, tb, g, w, ratio, psi, s, s0, polynomial, erf_delta, exp_delta
    integer :: i
    value = 0.0_dp
    if (.not. source%binned) then
      ratio = depth/source%temperature_ev
      psi = -ratio*f
      s0 = sqrt(ratio)
      s = s0*sqrt(1.0_dp - f)
      polynomial = 1.0_dp - (s*s + s*s0 + s0*s0)/3.0_dp + &
          (s**4 + s**3*s0 + s*s*s0*s0 + s*s0**3 + s0**4)/10.0_dp
      erf_delta = -2.0_dp*f*polynomial/(sqrt(pi*source%temperature_ev)*(1.0_dp + sqrt(1.0_dp - f)))
      exp_delta = -sqrt(depth)*f/source%temperature_ev*(1.0_dp + psi*(0.5_dp + psi*(1.0_dp/6.0_dp + psi/24.0_dp)))
      value = 0.5_dp*source%density_m3*exp(-boundary/source%temperature_ev)* &
          (exp_delta*erfc(s0) - exp(psi)*erf_delta)
      return
    end if
    do i = 1, size(source%flux_m2_s)
      a = max(source%edges_ev(i) - boundary, depth)
      b = source%edges_ev(i + 1) - boundary
      if (b <= a) cycle
      w = b - a
      if (source%edges_ev(i) - boundary >= depth) w = source%edges_ev(i + 1) - source%edges_ev(i)
      sa = sqrt(a)
      sb = sqrt(b)
      ta = sqrt(max(0.0_dp, a - depth*f))
      tb = sqrt(max(0.0_dp, b - depth*f))
      g = source%flux_m2_s(i)/(source%edges_ev(i + 1) - source%edges_ev(i))
      value = value + 2.0_dp*g/sqrt(2.0_dp*qe/mass_kg)* &
          (sqrt(depth)*f/(ta + sa))*(w/(tb + sb))*(1.0_dp/(ta + tb) + 1.0_dp/(sa + sb))
    end do
  end function

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

  !> Write a source to an unformatted unit; the caller owns the unit and its access mode.
  subroutine write_photoelectron_source(source, unit, iostat)
    type(photoelectron_source), intent(in) :: source
    integer, intent(in) :: unit
    integer, intent(out) :: iostat
    integer :: n
    n = 0
    if (source%binned) n = size(source%flux_m2_s)
    write (unit, iostat=iostat) source%binned, source%density_m3, source%temperature_ev, n
    if (iostat /= 0 .or. n == 0) return
    write (unit, iostat=iostat) source%edges_ev, source%flux_m2_s
  end subroutine write_photoelectron_source

  !> Read a source written by write_photoelectron_source. The result is not validated here.
  subroutine read_photoelectron_source(source, unit, iostat)
    type(photoelectron_source), intent(out) :: source
    integer, intent(in) :: unit
    integer, intent(out) :: iostat
    integer :: n
    read (unit, iostat=iostat) source%binned, source%density_m3, source%temperature_ev, n
    if (iostat /= 0) return
    if (n < 0 .or. (source%binned .neqv. n > 0)) then
      iostat = -1
      return
    end if
    if (n == 0) return
    allocate (source%edges_ev(n + 1), source%flux_m2_s(n))
    read (unit, iostat=iostat) source%edges_ev, source%flux_m2_s
  end subroutine read_photoelectron_source
end module
