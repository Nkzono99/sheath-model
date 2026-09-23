! SPDX-License-Identifier: MIT
module sheath_model_admissibility
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sheath_model_constants, only: dp, i32, sheath_ok, sheath_no_physical_solution, sheath_numerical_failure
  use sheath_model_core, only: zhao_params_type, integrate_zhao_rho, evaluate_zhao_rho_hat
  implicit none
  private
  public :: validate_zhao_profile
contains
  subroutine validate_zhao_profile(p, branch, phi0, phim, density, minimum_e2, boundary_e2, status, message)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: phi0, phim, density ! dimensionless
    real(dp), intent(out) :: minimum_e2, boundary_e2
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    real(dp) :: phi, e2, upper_e2, rho, fraction
    integer :: j, segment
    character(len=9) :: side
    status = sheath_no_physical_solution
    minimum_e2 = huge(1.0_dp)
    boundary_e2 = huge(1.0_dp)
    message = 'Inward electron drift with reflected slow electrons cannot approach neutral zero-field infinity.'
    ! For A/C, n_e(-h)-n_e(0) contains +const*u*h*log(1/h).
    ! For u>0 this dominates the regular ion/PE terms and makes E^2<0 arbitrarily near infinity.
    if ((branch == 'A' .or. branch == 'C') .and. p%u > 0.0_dp) return
    message = 'The root blocks the cold ion beam.'
    if (1.0_dp - 2.0_dp*max(phi0, 0.0_dp)/(p%tau*p%mach**2) <= 0.0_dp) return
    side = 'monotonic'
    if (branch == 'A') side = 'upper'
    call evaluate_zhao_rho_hat(p, branch, side, 0.0_dp, phi0, phim, density, rho)
    message = 'The root does not approach a neutral upstream state.'
    if (abs(rho) > 1e-7_dp*max(1.0_dp, density, p%n_swi_inf_m3/p%n_phe_ref_m3)) return
    if (branch == 'A') then
      upper_e2 = -2.0_dp*integrate_zhao_rho(p, branch, 'upper', phim, 0.0_dp, phi0, phim, density)
      message = 'The internal minimum does not connect to zero field at infinity.'
      if (abs(upper_e2) > 1e-7_dp) return
    end if
    minimum_e2 = 0.0_dp
    do segment = 1, merge(2, 1, branch == 'A')
      do j = 0, 128
        fraction = real(j, dp)/128.0_dp
        if (branch == 'A') then
          side = 'lower'
          phi = phim + (phi0 - phim)*fraction
          if (segment == 2) then
            side = 'upper'
            phi = phim*(1.0_dp - fraction)
          end if
          e2 = -2.0_dp*integrate_zhao_rho(p, branch, side, phim, phi, phi0, phim, density)
          if (segment == 1 .and. j == 128) boundary_e2 = e2
        else
          phi = phi0*(1.0_dp - fraction)
          e2 = 2.0_dp*integrate_zhao_rho(p, branch, 'monotonic', phi, 0.0_dp, phi0, phim, density)
          if (j == 0) boundary_e2 = e2
        end if
        if (.not. ieee_is_finite(e2)) then
          status = sheath_numerical_failure
          message = 'Profile field integral is non-finite.'
          return
        end if
        minimum_e2 = min(minimum_e2, e2)
      end do
    end do
    message = 'The algebraic root has no real connecting field profile.'
    if (minimum_e2 < -1e-8_dp*max(1.0_dp, abs(boundary_e2))) return
    status = sheath_ok
    message = ''
  end subroutine validate_zhao_profile
end module sheath_model_admissibility
