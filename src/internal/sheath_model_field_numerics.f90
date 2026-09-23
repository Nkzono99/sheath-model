! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone modules; status-returning public facade in sheath_model.
!> Zhao の初期推定と、汎用 Newton 法へ渡す物理残差の接続。
!! 数学的な反復処理は sheath_model_numerics、残差の定義と許容条件は physics が担当する。
submodule(sheath_model_field) sheath_model_field_numerics
  use sheath_model_numerics, only: try_guarded_newton_solve
  implicit none

contains

  module subroutine make_field_branch_guesses(params, branch, target_field_hat, guesses, count)
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: target_field_hat
    real(dp), intent(out) :: guesses(3, default_field_starts)
    integer, intent(out) :: count

    real(dp), parameter :: gaps(8) = [2.0_dp, 1.0_dp, 0.5_dp, 3.0_dp, 1.0_dp, 1.0_dp, 1.0_dp, 0.02_dp]
    real(dp), parameter :: depths(8) = [0.2_dp, 0.05_dp, 0.5_dp, 0.8_dp, 1.0_dp, 2.0_dp, 4.0_dp, 0.5_dp]
    real(dp), parameter :: voltages(8) = [0.002_dp, 0.02_dp, 0.2_dp, 0.6_dp, 1.5_dp, 4.0_dp, 12.0_dp, 50.0_dp]
    real(dp) :: source_ratio, source_shift, field_voltage, ion_limit, phi0, phim
    integer :: i

    ! Every start scales with T_pe and n_i. Additional starts respond to the
    ! emission strength and dimensionless prescribed field, never to SI constants.
    guesses = 0.0_dp
    count = 0
    source_ratio = params%n_phe0_m3/params%n_swi_inf_m3
    source_shift = log(max(1.0_dp, 0.5_dp*source_ratio))
    field_voltage = max(1e-10_dp, min(100.0_dp, abs(target_field_hat)*sqrt(params%tau)))
    ion_limit = 0.5_dp*params%tau*params%mach**2
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
          phim = -min(180.0_dp, phi0*max(field_voltage, source_shift))
          call add_guess(phim, phim)
        end if
      end do
    end select
  contains
    subroutine add_guess(surface_hat, minimum_hat)
      real(dp), intent(in) :: surface_hat, minimum_hat
      real(dp) :: surface, minimum, coefficient, photo_density, density, encoded(3)
      logical :: valid
      integer :: j
      surface = min(surface_hat, 0.8_dp*ion_limit, 180.0_dp)
      minimum = minimum_hat
      if (branch == 'A') then
        minimum = max(-180.0_dp, min(minimum, surface - 1e-6_dp))
      else if (branch == 'C') then
        surface = max(-180.0_dp, surface)
        minimum = surface
      else
        minimum = 0.0_dp
      end if
      coefficient = 0.5_dp*(1.0_dp + 2.0_dp*erf(params%u) + &
          erf(sqrt(max(0.0_dp, -minimum/params%tau)) - params%u))
      photo_density = 0.5_dp*source_ratio*exp(-surface)*erfc(sqrt(max(0.0_dp, -minimum)))
      ! Initialize N_e from neutrality where possible; leave potential adjustment
      ! to Newton if a trial voltage would require a nonpositive normalization.
      density = max(0.1_dp, min(1e5_dp, (1.0_dp - photo_density)/max(coefficient, 1e-12_dp)))
      call encode_field_unknowns(params, branch, surface*params%t_phe_ev, minimum*params%t_phe_ev, &
          density*params%n_swi_inf_m3, encoded, valid)
      if (.not. valid) return
      do j = 1, count
        if (maxval(abs(encoded - guesses(:, j))) < 1e-10_dp) return
      end do
      count = count + 1
      guesses(:, count) = encoded
    end subroutine add_guess
  end subroutine make_field_branch_guesses

  module subroutine newton_field_branch( &
      params, branch, target_field_hat, y0, y_out, final_norm, iterations, success &
      )
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: target_field_hat, y0(3)
    real(dp), intent(out) :: y_out(3), final_norm
    integer, intent(out) :: iterations
    logical, intent(out) :: success

    integer :: n

    n = merge(3, 2, branch == 'A')
    y_out = y0
    call try_guarded_newton_solve(n, y0(1:n), field_residual, y_out(1:n), final_norm, iterations, success)
  contains
    subroutine field_residual(y, f, valid)
      real(dp), intent(in) :: y(:)
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: valid
      real(dp) :: encoded(3), residual(3)

      encoded = y0
      encoded(1:n) = y
      call evaluate_charge_residual(params, branch, target_field_hat, encoded, residual, valid)
      f = residual(1:n)
    end subroutine field_residual
  end subroutine newton_field_branch

end submodule sheath_model_field_numerics
