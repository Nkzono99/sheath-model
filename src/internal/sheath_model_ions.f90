! SPDX-License-Identifier: MIT
!> Upstream-connected ion fluid root with constant pressure coefficient; Sana & Mishra (2026), Eq. (17).
module sheath_model_ions
  use sheath_model_constants, only: dp
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
  implicit none
  private
  public :: ion_density_ratio, ion_critical_potential

contains

  !> Critical positive potential [V]; energies [eV] are m_i*u_0^2/(2e) and pressure_factor*T_i.
  !! NaN for invalid inputs or an upstream speed at/below the ion thermal sound speed.
  pure real(dp) function ion_critical_potential(entry_energy_ev, pressure_energy_ev) result(value)
    real(dp), intent(in) :: entry_energy_ev, pressure_energy_ev
    real(dp) :: gap, difference
    integer :: n
    value = ieee_value(0.0_dp, ieee_quiet_nan)
    if (.not. all(ieee_is_finite([entry_energy_ev, pressure_energy_ev]))) return
    if (entry_energy_ev <= 0.0_dp .or. pressure_energy_ev < 0.0_dp) return
    if (2.0_dp*entry_energy_ev <= pressure_energy_ev) return
    value = entry_energy_ev
    if (pressure_energy_ev == 0.0_dp) return
    gap = (2.0_dp*entry_energy_ev - pressure_energy_ev)/pressure_energy_ev
    if (gap < 1e-3_dp) then
      difference = 0.0_dp
      do n = 2, 8
        difference = difference + (-1.0_dp)**n*gap**n/real(n, dp)
      end do
    else
      difference = gap - log(1.0_dp + gap)
    end if
    value = 0.5_dp*pressure_energy_ev*difference
  end function ion_critical_potential

  !> n_i/n_i,infinity on the supersonic root continuous from density ratio one at phi=0.
  !! NaN above the sonic turning potential, or at the divergent cold-ion endpoint.
  pure real(dp) function ion_density_ratio(potential_v, entry_energy_ev, pressure_energy_ev) result(value)
    real(dp), intent(in) :: potential_v, entry_energy_ev, pressure_energy_ev
    real(dp) :: critical, lower, upper, midpoint, residual, argument, exponential_difference
    integer :: iteration
    value = ieee_value(0.0_dp, ieee_quiet_nan)
    critical = ion_critical_potential(entry_energy_ev, pressure_energy_ev)
    if (.not. ieee_is_finite(critical) .or. .not. ieee_is_finite(potential_v)) return
    if (potential_v > critical) return
    if (pressure_energy_ev == 0.0_dp) then
      if (potential_v >= critical) return
      value = 1.0_dp/sqrt(1.0_dp - potential_v/entry_energy_ev)
      return
    end if
    if (potential_v == 0.0_dp) then
      value = 1.0_dp
      return
    end if
    if (potential_v == critical) then
      value = sqrt(2.0_dp*entry_energy_ev/pressure_energy_ev)
      return
    end if
    lower = 0.0_dp
    upper = 0.5_dp*log(2.0_dp*entry_energy_ev/pressure_energy_ev)
    if (potential_v < 0.0_dp) then
      lower = -0.5_dp*log(1.0_dp + 2.0_dp*abs(potential_v)/entry_energy_ev) - 1.0_dp
      upper = 0.0_dp
    end if
    do iteration = 1, 64
      midpoint = 0.5_dp*(lower + upper)
      argument = -2.0_dp*midpoint
      ! exp(x)-1 loses the weak-potential response; use its Taylor series near zero.
      if (abs(argument) < 1e-3_dp) then
        exponential_difference = argument*(1.0_dp + argument*(0.5_dp + argument*(1.0_dp/6.0_dp + &
            argument*(1.0_dp/24.0_dp + argument*(1.0_dp/120.0_dp + argument/720.0_dp)))))
      else
        exponential_difference = exp(argument) - 1.0_dp
      end if
      residual = entry_energy_ev*exponential_difference + pressure_energy_ev*midpoint + potential_v
      if (residual > 0.0_dp) then
        lower = midpoint
      else
        upper = midpoint
      end if
    end do
    value = exp(0.5_dp*(lower + upper))
  end function ion_density_ratio
end module sheath_model_ions
