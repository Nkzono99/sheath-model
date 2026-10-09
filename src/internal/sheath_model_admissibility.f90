! SPDX-License-Identifier: MIT
module sheath_model_admissibility
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sheath_model_constants, only: dp, i32, pi
  use sheath_model_status, only: SHEATH_OK, SHEATH_NO_PHYSICAL_SOLUTION, SHEATH_NUMERICAL_FAILURE
  use sheath_model_core, only: zhao_params_type, integrate_zhao_rho, evaluate_zhao_rho_hat, type_a_connection_residual
  use sheath_model_photoelectrons, only: photoelectron_sqrt_coefficient
  use sheath_model_ions, only: ion_density_ratio

  implicit none

  private

  public :: validate_zhao_profile

contains

  !> Check that an algebraic root connects to a neutral, zero-field upstream state with a real field profile.
  !! phi0 and phim are potentials / p%potential_scale_v; density is electron normalization / p%density_scale_m3.
  !! Returns sampled minimum_e2 and boundary_e2 in units of (p%potential_scale_v/p%length_scale_m)^2, plus status and message.
  !! With inward electron drift, A/C are accepted only if p%search%upstream_band_tolerance>0 and E^2<0 is confined
  !! to a band next to upstream no wider than that fraction of |phim| (A) or |phi0| (C); negative_band returns its
  !! width in units of p%potential_scale_v (0 otherwise).
  subroutine validate_zhao_profile( &
      p, branch, phi0, phim, density, &
      minimum_e2, boundary_e2, &
      status, message, negative_band &
      )
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: phi0, phim, density ! dimensionless
    real(dp), intent(out) :: minimum_e2, boundary_e2
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    real(dp), intent(out), optional :: negative_band

    integer, parameter :: n_samples = 129, n_geometric = 48, n_bisection = 40
    real(dp) :: phi, e2, rho, fraction, threshold, reference, band, outside, lower_minimum, mid
    real(dp) :: ambient_edge, photo_edge, connection_residual
    real(dp) :: depths(n_samples + n_geometric), upstream_e2s(n_samples + n_geometric)
    integer :: j, segment, n_upstream
    logical :: drifting_reflection
    character(len=9) :: side

    status = SHEATH_NO_PHYSICAL_SOLUTION
    minimum_e2 = huge(1.0_dp)
    boundary_e2 = huge(1.0_dp)
    if (present(negative_band)) negative_band = 0.0_dp
    message = 'Inward electron drift with reflected slow electrons cannot approach neutral zero-field infinity.'
    if (density <= 0.0_dp) then
      message = 'The electron normalization must be positive.'
      return
    end if
    ! For A/C, n_e(-h)-n_e(0) contains +const*u*h*log(1/h).
    ! For u>0 this dominates the regular ion/PE terms and makes E^2<0 arbitrarily near infinity.
    drifting_reflection = (branch == 'A' .or. branch == 'C') .and. p%u > 0.0_dp
    if (drifting_reflection .and. p%search%upstream_band_tolerance <= 0.0_dp) return

    if (branch == 'B' .and. phi0 > 0.0_dp) then
      ambient_edge = density*p%density_scale_m3*exp(-p%u**2)/sqrt(pi*p%t_swe_ev)
      photo_edge = photoelectron_sqrt_coefficient(p%photoelectrons, p%m_e_kg, phi0*p%potential_scale_v)
      if (ambient_edge - photo_edge > 128.0_dp*epsilon(1.0_dp)*max(abs(ambient_edge), abs(photo_edge))) then
        message = 'Type B has negative field squared arbitrarily near upstream infinity.'
        return
      end if
    end if

    message = 'The root blocks the upstream-connected ion flow.'
    if (.not. ieee_is_finite(ion_density_ratio(max(phi0, 0.0_dp)*p%potential_scale_v, &
        0.5_dp*p%t_swe_ev*p%mach**2, p%ion_pressure_factor*p%t_swi_ev))) return

    side = 'monotonic'
    if (branch == 'A') then
      side = 'upper'
    end if
    call evaluate_zhao_rho_hat(p, branch, side, 0.0_dp, phi0, phim, density, rho)
    message = 'The root does not approach a neutral upstream state.'
    if (abs(rho) > 1e-7_dp*max(1.0_dp, density, p%n_swi_inf_m3/p%density_scale_m3)) return

    if (branch == 'A') then
      message = 'The internal minimum does not connect to zero field at infinity.'
      connection_residual = type_a_connection_residual(p, phi0, phim, density)
      if (.not. ieee_is_finite(connection_residual)) then
        status = SHEATH_NUMERICAL_FAILURE
        message = 'The upper connection residual is non-finite.'
        return
      end if
      if (abs(connection_residual) > 1e-7_dp) return
    end if

    minimum_e2 = 0.0_dp
    lower_minimum = 0.0_dp
    n_upstream = 0
    do segment = 1, merge(2, 1, branch == 'A')
      do j = 0, n_samples - 1
        fraction = real(j, dp)/real(n_samples - 1, dp)

        if (branch == 'A') then
          side = 'lower'
          phi = phim + (phi0 - phim)*fraction

          if (segment == 2) then
            side = 'upper'
            phi = phim*(1.0_dp - fraction)
          end if

          e2 = -2.0_dp*integrate_zhao_rho(p, branch, side, phim, phi, phi0, phim, density)

          if (segment == 1 .and. j == n_samples - 1) then
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
        if (branch == 'A' .and. segment == 1) then
          lower_minimum = min(lower_minimum, e2)
        else
          n_upstream = n_upstream + 1
          depths(n_upstream) = -phi
          upstream_e2s(n_upstream) = e2
        end if
      end do
    end do

    message = 'The algebraic root has no real connecting field profile.'
    threshold = -1e-8_dp*max(1.0_dp, abs(boundary_e2))
    if (.not. drifting_reflection) then
      if (minimum_e2 < threshold) return
      status = SHEATH_OK
      message = ''
      return
    end if

    ! Locate the outermost depth below upstream with E^2<0; the log term makes the band touch upstream.
    ! The band is measured by sign, since for C it can be shallower than the acceptance threshold.
    if (lower_minimum < threshold) return
    reference = merge(-phim, -phi0, branch == 'A')
    do j = 1, n_geometric
      phi = -reference*0.5_dp**j
      e2 = upstream_e2(phi)
      if (.not. ieee_is_finite(e2)) then
        status = SHEATH_NUMERICAL_FAILURE
        message = 'Profile field integral is non-finite.'
        return
      end if
      minimum_e2 = min(minimum_e2, e2)
      n_upstream = n_upstream + 1
      depths(n_upstream) = -phi
      upstream_e2s(n_upstream) = e2
    end do
    band = 0.0_dp
    do j = 1, n_upstream
      if (upstream_e2s(j) < 0.0_dp) band = max(band, depths(j))
    end do
    if (band > 0.0_dp) then
      outside = reference
      do j = 1, n_upstream
        if (depths(j) > band) outside = min(outside, depths(j))
      end do
      do j = 1, n_bisection
        mid = 0.5_dp*(band + outside)
        if (upstream_e2(-mid) < 0.0_dp) then
          band = mid
        else
          outside = mid
        end if
      end do
    end if
    message = 'Negative E^2 next to upstream is wider than upstream_band_tolerance allows.'
    if (band > p%search%upstream_band_tolerance*reference) return
    if (present(negative_band)) negative_band = band

    status = SHEATH_OK
    message = ''

  contains

    real(dp) function upstream_e2(phi_value) result(value)
      real(dp), intent(in) :: phi_value
      if (branch == 'A') then
        value = -2.0_dp*integrate_zhao_rho(p, branch, 'upper', phim, phi_value, phi0, phim, density)
      else
        value = 2.0_dp*integrate_zhao_rho(p, branch, 'monotonic', phi_value, 0.0_dp, phi0, phim, density)
      end if
    end function upstream_e2
  end subroutine validate_zhao_profile

end module sheath_model_admissibility
