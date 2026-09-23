! SPDX-License-Identifier: MIT AND Apache-2.0
! Zero-drift evaluation adapted from BEACH 3897f38; see NOTICE.
! Incoming electrons are defined at infinity and transported by energy conservation.
module sheath_model_orbits
  use sheath_model_constants, only: dp, pi
  implicit none

  private

  public :: electron_density, gauss_x, gauss_w
  !> Abscissae of the 16-point Gauss-Legendre quadrature on [-1,1].
  real(dp), parameter :: gauss_x(16) = [ &
      -0.9894009349916499_dp, -0.9445750230732326_dp, -0.8656312023878318_dp, -0.7554044083550030_dp, &
      -0.6178762444026438_dp, -0.4580167776572274_dp, -0.2816035507792589_dp, -0.0950125098376374_dp, &
      0.0950125098376374_dp, 0.2816035507792589_dp, 0.4580167776572274_dp, 0.6178762444026438_dp, &
      0.7554044083550030_dp, 0.8656312023878318_dp, 0.9445750230732326_dp, 0.9894009349916499_dp]
  !> Dimensionless quadrature weights corresponding to gauss_x; their sum is two.
  real(dp), parameter :: gauss_w(16) = [ &
      0.0271524594117541_dp, 0.0622535239386479_dp, 0.0951585116824928_dp, 0.1246289712555339_dp, &
      0.1495959888165767_dp, 0.1691565193950025_dp, 0.1826034150449236_dp, 0.1894506104550685_dp, &
      0.1894506104550685_dp, 0.1826034150449236_dp, 0.1691565193950025_dp, 0.1495959888165767_dp, &
      0.1246289712555339_dp, 0.0951585116824928_dp, 0.0622535239386479_dp, 0.0271524594117541_dp]

contains

  !> Return free and reflected electron densities divided by the upstream Maxwellian normalization.
  !! psi and barrier are local and minimum potentials divided by T_e [eV]; u is inward drift / thermal speed.
  pure subroutine electron_density(psi, barrier, u, free, reflected)
    real(dp), intent(in) :: psi, barrier, u
    real(dp), intent(out) :: free, reflected

    real(dp) :: cutoff, upper, amax, amin

    cutoff = sqrt(max(0.0_dp, psi - barrier))
    if (u == 0.0_dp) then
      free = 0.5_dp*exp(barrier)*erfc_scaled(cutoff)
      if (psi <= 0.0_dp) then
        reflected = exp(psi)*erf(cutoff)
      else
        reflected = max(0.0_dp, erfc_scaled(sqrt(psi)) - 2.0_dp*free)
      end if
      return
    end if
    amin = max(0.0_dp, u - 10.0_dp)
    amax = max(sqrt(max(0.0_dp, -barrier)), u, 0.0_dp) + 10.0_dp
    upper = sqrt(max(0.0_dp, amax*amax + psi))
    free = local_integral(sqrt(max(0.0_dp, psi - barrier, amin*amin + psi)), upper, psi, u)
    reflected = 2.0_dp*local_integral(sqrt(max(0.0_dp, amin*amin + psi)), cutoff, psi, u)
  end subroutine electron_density

  pure real(dp) function local_integral(lower, upper, psi, u) result(value)
    real(dp), intent(in) :: lower, upper, psi, u

    real(dp) :: t, w, upstream, width
    integer :: panel, j

    value = 0.0_dp
    width = upper - lower
    if (width <= 0.0_dp) return

    ! w=lower+width*t^2 removes the square-root endpoint at w^2=psi.
    do panel = 0, 7
      do j = 1, 16
        t = (real(panel, dp) + 0.5_dp*(1.0_dp + gauss_x(j)))/8.0_dp
        w = lower + width*t*t
        upstream = sqrt(max(0.0_dp, w*w - psi))
        value = value + gauss_w(j)*2.0_dp*width*t*exp(-(upstream - u)**2)
      end do
    end do
    value = value/(16.0_dp*sqrt(pi))
  end function local_integral

end module sheath_model_orbits
