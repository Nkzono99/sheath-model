! SPDX-License-Identifier: MIT
module sheath_model_admissibility
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sheath_model_constants, only: dp, i32, pi
  use sheath_model_status, only: SHEATH_OK, SHEATH_NO_PHYSICAL_SOLUTION, SHEATH_NUMERICAL_FAILURE
  use sheath_model_core, only: zhao_params_type, integrate_zhao_rho, evaluate_zhao_rho_hat
  use sheath_model_photoelectrons, only: photoelectron_sqrt_coefficient

  implicit none

  private

  public :: validate_zhao_profile

contains

  !> Check that an algebraic root connects to a neutral, zero-field upstream state with a real field profile.
  !! phi0 and phim are potentials / p%potential_scale_v; density is electron normalization / p%density_scale_m3.
  !! Returns sampled minimum_e2 and boundary_e2 in units of (p%potential_scale_v/p%length_scale_m)^2, plus status and message.
  subroutine validate_zhao_profile( &
      p, branch, phi0, phim, density, &
      minimum_e2, boundary_e2, &
      status, message &
      )
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: phi0, phim, density ! dimensionless
    real(dp), intent(out) :: minimum_e2, boundary_e2
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    real(dp) :: phi, e2, upper_e2, rho, fraction
    real(dp) :: ambient_edge, photo_edge, connection_residual
    integer :: j, segment
    character(len=9) :: side

    status = SHEATH_NO_PHYSICAL_SOLUTION
    minimum_e2 = huge(1.0_dp)
    boundary_e2 = huge(1.0_dp)
    message = 'Inward electron drift with reflected slow electrons cannot approach neutral zero-field infinity.'
    if (density <= 0.0_dp) then
      message = 'The electron normalization must be positive.'
      return
    end if
    ! For A/C, n_e(-h)-n_e(0) contains +const*u*h*log(1/h).
    ! For u>0 this dominates the regular ion/PE terms and makes E^2<0 arbitrarily near infinity.
    if ((branch == 'A' .or. branch == 'C') .and. p%u > 0.0_dp) return

    if (branch == 'B' .and. phi0 > 0.0_dp) then
      ambient_edge = density*p%density_scale_m3*exp(-p%u**2)/sqrt(pi*p%t_swe_ev)
      photo_edge = photoelectron_sqrt_coefficient(p%photoelectrons, p%m_e_kg, phi0*p%potential_scale_v)
      if (ambient_edge - photo_edge > 128.0_dp*epsilon(1.0_dp)*max(abs(ambient_edge), abs(photo_edge))) then
        message = 'Type B has negative field squared arbitrarily near upstream infinity.'
        return
      end if
    end if

    message = 'The root blocks the cold ion beam.'
    if (1.0_dp - 2.0_dp*max(phi0, 0.0_dp)/(p%tau*p%mach**2) <= 0.0_dp) return

    side = 'monotonic'
    if (branch == 'A') then
      side = 'upper'
    end if
    call evaluate_zhao_rho_hat(p, branch, side, 0.0_dp, phi0, phim, density, rho)
    message = 'The root does not approach a neutral upstream state.'
    if (abs(rho) > 1e-7_dp*max(1.0_dp, density, p%n_swi_inf_m3/p%density_scale_m3)) return

    if (branch == 'A') then
      upper_e2 = -2.0_dp*integrate_zhao_rho(p, branch, 'upper', phim, 0.0_dp, phi0, phim, density)
      message = 'The internal minimum does not connect to zero field at infinity.'
      connection_residual = upper_e2
      if (p%photoelectrons%is_binned()) then
        connection_residual = upper_e2/(-phim)**1.5_dp
      end if
      if (.not. ieee_is_finite(connection_residual)) then
        status = SHEATH_NUMERICAL_FAILURE
        message = 'The upper connection residual is non-finite.'
        return
      end if
      if (abs(connection_residual) > 1e-7_dp) return
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

          if (segment == 1 .and. j == 128) then
            boundary_e2 = e2
          end if
        else
          phi = phi0*(1.0_dp - fraction)
          e2 = 2.0_dp*integrate_zhao_rho(p, branch, 'monotonic', phi, 0.0_dp, phi0, phim, density)

          if (j == 0) then
            boundary_e2 = e2
          end if
        end if

        if (.not. ieee_is_finite(e2)) then
          status = SHEATH_NUMERICAL_FAILURE
          message = 'Profile field integral is non-finite.'
          return
        end if

        minimum_e2 = min(minimum_e2, e2)
      end do
    end do

    message = 'The algebraic root has no real connecting field profile.'
    if (minimum_e2 < -1e-8_dp*max(1.0_dp, abs(boundary_e2))) return

    status = SHEATH_OK
    message = ''
  end subroutine validate_zhao_profile

end module sheath_model_admissibility
