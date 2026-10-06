! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
!> Common dimensionless coordinates and physically scaled multistart guesses.
module sheath_model_coordinates
  use sheath_model_constants, only: dp
  use sheath_model_core, only: zhao_params_type, neutral_electron_density
  use sheath_model_ions, only: ion_critical_potential
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: encode_unknowns, decode_unknowns, make_branch_guesses
contains
  subroutine encode_unknowns( &
      params, branch, &
      phi0_v, phi_m_v, density_m3, &
      y, valid &
      )
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: phi0_v, phi_m_v, density_m3
    real(dp), intent(out) :: y(3)
    logical, intent(out) :: valid

    y = 0.0_dp
    valid = density_m3 > 0.0_dp .and. params%n_swi_inf_m3 > 0.0_dp .and. params%potential_scale_v > 0.0_dp
    if (.not. valid) return

    select case (branch)
    case ('A')
      valid = phi_m_v < min(phi0_v, 0.0_dp)
      if (.not. valid) return

      y(1) = log((phi0_v - phi_m_v)/params%potential_scale_v)
      y(2) = log(-phi_m_v/params%potential_scale_v)
      y(3) = log(density_m3/params%n_swi_inf_m3)
    case ('B')
      valid = phi0_v > 0.0_dp
      if (.not. valid) return

      y(1) = log(phi0_v/params%potential_scale_v)
      y(2) = log(density_m3/params%n_swi_inf_m3)
    case ('C')
      valid = phi0_v < 0.0_dp
      if (.not. valid) return

      y(1) = log(-phi0_v/params%potential_scale_v)
      y(2) = log(density_m3/params%n_swi_inf_m3)
    case default
      valid = .false.
    end select

    valid = valid .and. all(ieee_is_finite(y))
  end subroutine encode_unknowns

  subroutine decode_unknowns( &
      params, branch, y, &
      phi0_v, phi_m_v, density_m3, &
      valid &
      )
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: y(3)
    real(dp), intent(out) :: phi0_v, phi_m_v, density_m3
    logical, intent(out) :: valid

    phi0_v = 0.0_dp
    phi_m_v = 0.0_dp
    density_m3 = 0.0_dp
    valid = all(ieee_is_finite(y))
    if (.not. valid .or. y(1) < -50.0_dp .or. y(1) > log(params%search%potential_extent)) then
      valid = .false.
      return
    end if

    select case (branch)
    case ('A')
      if (y(2) < -50.0_dp .or. y(2) > log(params%search%potential_extent) .or. &
          y(3) < -30.0_dp .or. y(3) > log(1.0e6_dp)) then
        valid = .false.
        return
      end if
      phi_m_v = -params%potential_scale_v*exp(y(2))
      phi0_v = phi_m_v + params%potential_scale_v*exp(y(1))
      density_m3 = params%n_swi_inf_m3*exp(y(3))
    case ('B')
      if (y(2) < -30.0_dp .or. y(2) > log(1.0e6_dp)) then
        valid = .false.
        return
      end if
      phi0_v = params%potential_scale_v*exp(y(1))
      phi_m_v = 0.0_dp
      density_m3 = params%n_swi_inf_m3*exp(y(2))
    case ('C')
      if (y(2) < -30.0_dp .or. y(2) > log(1.0e6_dp)) then
        valid = .false.
        return
      end if
      phi0_v = -params%potential_scale_v*exp(y(1))
      phi_m_v = phi0_v
      density_m3 = params%n_swi_inf_m3*exp(y(2))
    case default
      valid = .false.
    end select

    valid = valid .and. all(ieee_is_finite([phi0_v, phi_m_v, density_m3]))
  end subroutine decode_unknowns

  subroutine make_branch_guesses(params, branch, target_field_hat, guesses, count)
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: target_field_hat
    real(dp), intent(out) :: guesses(3, 16)
    integer, intent(out) :: count

    real(dp), parameter :: gaps(8) = [2.0_dp, 1.0_dp, 0.5_dp, 3.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 0.02_dp]
    real(dp), parameter :: depths(8) = [0.2_dp, 0.05_dp, 0.5_dp, 0.8_dp, 1.0_dp, 2.0_dp, 4.0_dp, 0.5_dp]
    real(dp), parameter :: voltages(8) = [0.002_dp, 0.02_dp, 0.2_dp, 0.6_dp, 1.5_dp, 4.0_dp, 12.0_dp, 50.0_dp]
    real(dp) :: source_ratio, source_shift, field_voltage, ion_limit, phi0, phim
    integer :: i

    ! Every start scales with potential_scale_v and n_i. Additional starts respond to the
    ! emission strength and dimensionless prescribed field, never to SI constants.
    guesses = 0.0_dp
    count = 0
    source_ratio = params%emission_density_scale_m3/params%n_swi_inf_m3
    source_shift = log(max(1.0_dp, 0.5_dp*source_ratio))
    field_voltage = max(1e-10_dp, min(100.0_dp, abs(target_field_hat)*sqrt(params%tau)))
    ion_limit = ion_critical_potential(0.5_dp*params%t_swe_ev*params%mach**2, &
        params%ion_pressure_factor*params%t_swi_ev)/params%potential_scale_v

    select case (branch)
    case ('A')
      do i = 1, size(gaps)
        call add_guess(gaps(i) - depths(i), -depths(i))
        call add_guess(gaps(i) + source_shift - depths(i), -depths(i)*sqrt(1.0_dp + field_voltage))
      end do
    case ('B', 'C')
      do i = 1, size(voltages)
        phi0 = voltages(i)
        if (branch == 'B') then
          call add_guess(phi0, 0.0_dp)
          call add_guess(source_shift + phi0*field_voltage, 0.0_dp)
        else
          call add_guess(-phi0, -phi0)
          phim = -min(0.9_dp*params%search%potential_extent, phi0*max(field_voltage, source_shift))
          call add_guess(phim, phim)
        end if
      end do
    end select

  contains

    subroutine add_guess(surface_hat, minimum_hat)
      real(dp), intent(in) :: surface_hat, minimum_hat
      real(dp) :: surface, minimum, density, encoded(3)
      logical :: valid
      integer :: j
      surface = min(surface_hat, 0.8_dp*ion_limit, 0.9_dp*params%search%potential_extent)
      minimum = minimum_hat
      if (branch == 'A') then
        minimum = max(-0.9_dp*params%search%potential_extent, min(minimum, surface - 1e-6_dp))
      else if (branch == 'C') then
        surface = max(-0.9_dp*params%search%potential_extent, surface)
        minimum = surface
      else
        minimum = 0.0_dp
      end if
      density = neutral_electron_density(params, branch, surface*params%potential_scale_v, &
          minimum*params%potential_scale_v)/params%n_swi_inf_m3
      if (.not. ieee_is_finite(density)) return
      density = max(0.1_dp, min(1e5_dp, density))

      call encode_unknowns(params, branch, surface*params%potential_scale_v, minimum*params%potential_scale_v, &
          density*params%n_swi_inf_m3, encoded, valid)

      if (.not. valid) return

      do j = 1, count
        if (maxval(abs(encoded - guesses(:, j))) < 1e-10_dp) return
      end do

      count = count + 1
      guesses(:, count) = encoded
    end subroutine add_guess
  end subroutine make_branch_guesses

end module sheath_model_coordinates
