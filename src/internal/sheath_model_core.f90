! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone modules; status-returning public facade in sheath_model.
!> Zhao 系シースの物理量、残差、および枝ごとの初期値・探索。
module sheath_model_core
  use sheath_model_photoelectrons, only: photoelectron_source, photoelectron_density, photoelectron_fluxes, &
      photoelectron_density_integral
  use sheath_model_orbits, only: electron_density, gauss_x, gauss_w
  use sheath_model_constants, only: dp
  use sheath_model_numerics, only: solve_nonlinear_system, residual_norm, NONLINEAR_TOL
  use sheath_model_constants, only: pi, eps0, qe
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
  implicit none
  private

  !> Internal plasma parameters and derived scales, prepared by the public model wrappers.
  !! Physical fields use their suffix units; mach, u and tau are dimensionless ratios.
  type :: zhao_params_type
    type(photoelectron_source) :: photoelectrons
    real(dp) :: alpha_rad = 0.0d0
    real(dp) :: n_swi_inf_m3 = 0.0d0
    real(dp) :: density_scale_m3 = 0.0d0
    real(dp) :: emission_density_scale_m3 = 0.0d0
    real(dp) :: t_swe_ev = 0.0d0
    real(dp) :: potential_scale_v = 0.0d0
    real(dp) :: v_d_electron_mps = 0.0d0
    real(dp) :: v_d_ion_mps = 0.0d0
    real(dp) :: m_i_kg = 0.0d0
    real(dp) :: m_e_kg = 0.0d0
    real(dp) :: v_swe_th_mps = 0.0d0
    real(dp) :: velocity_scale_mps = 0.0d0
    real(dp) :: cs_mps = 0.0d0
    real(dp) :: mach = 0.0d0
    real(dp) :: u = 0.0d0
    real(dp) :: tau = 0.0d0
    real(dp) :: length_scale_m = 0.0d0
  end type zhao_params_type

  public :: neutral_electron_density, evaluate_zhao_fluxes
  public :: zhao_params_type
  public :: try_solve_zhao_unknowns
  public :: evaluate_zhao_density_hat
  public :: evaluate_zhao_rho_hat
  public :: zhao_residuals_type_a
  public :: zhao_residuals_type_b
  public :: zhao_residuals_type_c
  public :: swe_free_current_term
  public :: type_a_e2_sum_at_infinity, integrate_zhao_rho

contains

  !> Return net charge density rho_hat = rho/(qe*p%density_scale_m3) on the selected branch and side.
  !! Potentials are divided by p%potential_scale_v, and n_swe_inf_hat is electron normalization / p%density_scale_m3.
  subroutine evaluate_zhao_rho_hat(p, branch, side, phi_hat, phi0_hat, phi_m_hat, n_swe_inf_hat, rho_hat)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    character(len=*), intent(in) :: side
    real(dp), intent(in) :: phi_hat, phi0_hat, phi_m_hat, n_swe_inf_hat
    real(dp), intent(out) :: rho_hat

    real(dp) :: n_swi_hat, n_swe_f_hat, n_swe_r_hat, n_phe_f_hat, n_phe_c_hat

    call evaluate_zhao_density_hat( &
        p, branch, side, phi_hat, phi0_hat, phi_m_hat, n_swe_inf_hat, &
        n_swi_hat, n_swe_f_hat, n_swe_r_hat, n_phe_f_hat, n_phe_c_hat &
        )
    rho_hat = n_swi_hat - n_swe_f_hat - n_swe_r_hat - n_phe_f_hat - n_phe_c_hat
  end subroutine evaluate_zhao_rho_hat

  !> Return ion, free/reflected electron, and free/captured photoelectron densities / p%density_scale_m3.
  !! Potentials are divided by p%potential_scale_v; n_swe_inf_hat uses the same density normalization as the outputs.
  !! branch is A/B/C; Type A requires side='lower' or 'upper'. A blocked ion beam produces NaN densities.
  subroutine evaluate_zhao_density_hat( &
      p, branch, side, phi_hat, phi0_hat, phi_m_hat, n_swe_inf_hat, &
      n_swi_hat, n_swe_f_hat, n_swe_r_hat, n_phe_f_hat, n_phe_c_hat &
      )
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    character(len=*), intent(in) :: side
    real(dp), intent(in) :: phi_hat, phi0_hat, phi_m_hat, n_swe_inf_hat
    real(dp), intent(out) :: n_swi_hat, n_swe_f_hat, n_swe_r_hat, n_phe_f_hat, n_phe_c_hat

    real(dp) :: photo_free, photo_captured
    logical :: lower_side

    call evaluate_background_hat(p, branch, side, phi_hat, phi_m_hat, n_swe_inf_hat, n_swi_hat, n_swe_f_hat, n_swe_r_hat)
    lower_side = branch == 'B' .or. (branch == 'A' .and. side == 'lower')
    call photoelectron_density(p%photoelectrons, p%m_e_kg, phi0_hat*p%potential_scale_v, &
        phi_m_hat*p%potential_scale_v, phi_hat*p%potential_scale_v, lower_side, photo_free, photo_captured)
    n_phe_f_hat = photo_free/p%density_scale_m3
    n_phe_c_hat = photo_captured/p%density_scale_m3
  end subroutine evaluate_zhao_density_hat

  !> 指定枝の代数根を探索する。物理解の判定と auto 選択は公開窓口で行う。
  !! model は zhao_a/zhao_b/zhao_c。電位 [V]、電子規格化密度 [m^-3]、枝と収束成否を返す。
  !! 最小電位は B では上流の 0 V、C では境界電位と一致する。
  subroutine try_solve_zhao_unknowns(model, p, phi0_v, phi_m_v, n_swe_inf_m3, branch, success)
    character(len=*), intent(in) :: model
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(out) :: phi0_v, phi_m_v, n_swe_inf_m3
    character(len=1), intent(out) :: branch
    logical, intent(out) :: success

    real(dp) :: x3(3), x2(2)

    phi0_v = 0.0_dp
    phi_m_v = 0.0_dp
    n_swe_inf_m3 = 0.0_dp
    branch = ' '
    success = .false.

    select case (trim(model))
    case ('zhao_a')
      call try_solve_zhao_branch_a(p, x3, success)
      if (success) then
        phi0_v = x3(1)
        phi_m_v = x3(2)
        n_swe_inf_m3 = x3(3)
        branch = 'A'
      end if
      return
    case ('zhao_b')
      call try_solve_zhao_branch_b(p, x2, success)
      if (success) then
        phi0_v = x2(1)
        phi_m_v = 0.0_dp
        n_swe_inf_m3 = x2(2)
        branch = 'B'
      end if
      return
    case ('zhao_c')
      call try_solve_zhao_branch_c(p, x2, success)
      if (success) then
        phi0_v = x2(1)
        phi_m_v = x2(1)
        n_swe_inf_m3 = x2(2)
        branch = 'C'
      end if
      return
    end select

  end subroutine try_solve_zhao_unknowns

  subroutine try_solve_zhao_branch_a(p, x, success)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(out) :: x(3)
    logical, intent(out) :: success

    real(dp) :: guesses(3, 6)

    guesses(:, 1) = [1.6_dp*p%potential_scale_v, -0.3_dp*p%potential_scale_v, 0.9_dp*p%n_swi_inf_m3]
    guesses(:, 2) = [0.5_dp*p%potential_scale_v, -0.5_dp*p%potential_scale_v, 0.9_dp*p%n_swi_inf_m3]
    guesses(:, 3) = [-0.2_dp*p%potential_scale_v, -0.8_dp*p%potential_scale_v, 0.9_dp*p%n_swi_inf_m3]
    guesses(:, 4) = [-p%potential_scale_v, -2.0_dp*p%potential_scale_v, p%n_swi_inf_m3]
    guesses(:, 5) = [3.0_dp*p%potential_scale_v, -0.1_dp*p%potential_scale_v, p%n_swi_inf_m3]
    guesses(:, 6) = [-3.0_dp*p%potential_scale_v, -4.0_dp*p%potential_scale_v, p%n_swi_inf_m3]
    call solve_nonlinear_system(3, guesses, residual_a, x, success)

  contains

    subroutine residual_a(xa, fa)
      real(dp), intent(in) :: xa(:)
      real(dp), intent(out) :: fa(:)

      call zhao_residuals_type_a(p, xa, fa)
      fa(1:2) = fa(1:2)/p%density_scale_m3
    end subroutine residual_a

  end subroutine try_solve_zhao_branch_a

  subroutine try_solve_zhao_branch_b(p, x, success)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(out) :: x(2)
    logical, intent(out) :: success

    real(dp) :: guesses(2, 3)

    call try_solve_zhao_monotonic_scalar(p, 'B', x, success)
    if (success) return
    guesses(:, 1) = [1.3d0, 7.0d6]
    guesses(:, 2) = [0.8d0, 6.5d6]
    guesses(:, 3) = [2.0d0, 7.8d6]
    call solve_nonlinear_system(2, guesses, residual_b, x, success)

  contains

    subroutine residual_b(xb, fb)
      real(dp), intent(in) :: xb(:)
      real(dp), intent(out) :: fb(:)

      call zhao_residuals_type_b(p, xb, fb)
      fb(1:2) = fb(1:2)/p%density_scale_m3
    end subroutine residual_b

  end subroutine try_solve_zhao_branch_b

  subroutine try_solve_zhao_branch_c(p, x, success)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(out) :: x(2)
    logical, intent(out) :: success

    real(dp) :: guesses(2, 5)

    call try_solve_zhao_monotonic_scalar(p, 'C', x, success)
    if (success) return
    guesses(:, 1) = [-0.5d0, 6.0d6]
    guesses(:, 2) = [-2.0d0, 7.0d6]
    guesses(:, 3) = [-5.0d0, 8.0d6]
    guesses(:, 4) = [-10.0d0, 8.2d6]
    guesses(:, 5) = [-15.0d0, 8.5d6]
    call solve_nonlinear_system(2, guesses, residual_c, x, success)

  contains

    subroutine residual_c(xc, fc)
      real(dp), intent(in) :: xc(:)
      real(dp), intent(out) :: fc(:)

      call zhao_residuals_type_c(p, xc, fc)
      fc(1:2) = fc(1:2)/p%density_scale_m3
    end subroutine residual_c

  end subroutine try_solve_zhao_branch_c

  !> Type-B/C の定常電流式から ambient density を消去し、電位だけをbracketする。
  subroutine try_solve_zhao_monotonic_scalar(p, branch, x, success)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), intent(out) :: x(2)
    logical, intent(out) :: success

    integer :: iteration
    real(dp) :: phi_left, phi_right, phi_mid, residual_left, residual_right, residual_mid
    real(dp) :: density_left, density_right, density_mid, voltage_scale, phi_limit
    real(dp) :: residual(2), residual_scale
    logical :: left_ok, right_ok, mid_ok

    x = 0.0_dp
    success = .false.
    voltage_scale = max(p%potential_scale_v, p%t_swe_ev, 1.0_dp)
    residual_scale = max(p%n_swi_inf_m3, p%emission_density_scale_m3, 1.0_dp)
    select case (branch)
    case ('B')
      phi_left = 128.0_dp*epsilon(1.0_dp)*voltage_scale
      phi_right = voltage_scale
      phi_limit = 100.0_dp*voltage_scale
    case ('C')
      phi_left = -voltage_scale
      phi_right = -128.0_dp*epsilon(1.0_dp)*voltage_scale
      phi_limit = -100.0_dp*voltage_scale
    case default
      return
    end select

    call evaluate_monotonic_stationary_phi(p, branch, phi_left, residual_left, density_left, left_ok)
    call evaluate_monotonic_stationary_phi(p, branch, phi_right, residual_right, density_right, right_ok)
    if (.not. left_ok .or. .not. right_ok) return
    do iteration = 1, 16
      if (residual_left == 0.0_dp .or. residual_right == 0.0_dp .or. &
          sign(1.0_dp, residual_left) /= sign(1.0_dp, residual_right)) exit
      if (branch == 'B') then
        phi_right = min(phi_limit, 2.0_dp*phi_right)
        call evaluate_monotonic_stationary_phi( &
            p, branch, phi_right, residual_right, density_right, right_ok &
            )
        if (.not. right_ok .or. phi_right >= phi_limit) exit
      else
        phi_left = max(phi_limit, 2.0_dp*phi_left)
        call evaluate_monotonic_stationary_phi( &
            p, branch, phi_left, residual_left, density_left, left_ok &
            )
        if (.not. left_ok .or. phi_left <= phi_limit) exit
      end if
    end do
    if (.not. left_ok .or. .not. right_ok) return
    if (residual_left /= 0.0_dp .and. residual_right /= 0.0_dp) then
      if (sign(1.0_dp, residual_left) == sign(1.0_dp, residual_right)) return
    end if

    if (residual_left == 0.0_dp) then
      x = [phi_left, density_left]
    else if (residual_right == 0.0_dp) then
      x = [phi_right, density_right]
    else
      do iteration = 1, 160
        phi_mid = phi_left + 0.5_dp*(phi_right - phi_left)
        call evaluate_monotonic_stationary_phi( &
            p, branch, phi_mid, residual_mid, density_mid, mid_ok &
            )
        if (.not. mid_ok) return
        if (residual_mid == 0.0_dp) then
          phi_left = phi_mid
          phi_right = phi_mid
          density_left = density_mid
          density_right = density_mid
          exit
        end if
        if (sign(1.0_dp, residual_left) /= sign(1.0_dp, residual_mid)) then
          phi_right = phi_mid
          residual_right = residual_mid
          density_right = density_mid
        else
          phi_left = phi_mid
          residual_left = residual_mid
          density_left = density_mid
        end if
        if (abs(phi_right - phi_left) <= &
            256.0_dp*epsilon(1.0_dp)*max(1.0_dp, abs(phi_left), abs(phi_right))) exit
      end do
      phi_mid = phi_left + 0.5_dp*(phi_right - phi_left)
      call evaluate_monotonic_stationary_phi( &
          p, branch, phi_mid, residual_mid, density_mid, mid_ok &
          )
      if (.not. mid_ok) return
      x = [phi_mid, density_mid]
    end if

    if (branch == 'B') then
      call zhao_residuals_type_b(p, x, residual)
    else
      call zhao_residuals_type_c(p, x, residual)
    end if
    success = all(ieee_is_finite(x)) .and. all(ieee_is_finite(residual)) .and. &
        x(2) > 0.0_dp .and. residual_norm(residual) <= &
        max(NONLINEAR_TOL, 1024.0_dp*epsilon(1.0_dp)*residual_scale)
  end subroutine try_solve_zhao_monotonic_scalar

  subroutine evaluate_monotonic_stationary_phi(p, branch, phi_v, residual_v, density_m3, success)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: phi_v
    real(dp), intent(out) :: residual_v, density_m3
    logical, intent(out) :: success

    real(dp) :: cutoff, ion_term, source_current_term, coefficient, residual(2)

    real(dp) :: outward, escape, returning

    residual_v = huge(1.0_dp)
    density_m3 = 0.0_dp
    success = .false.

    select case (branch)
    case ('B')
      if (phi_v <= 0.0_dp) return
      cutoff = -p%u
      call photoelectron_fluxes(p%photoelectrons, p%m_e_kg, phi_v, outward, escape, returning)
      source_current_term = escape/(p%velocity_scale_mps/(2.0_dp*sqrt(pi)))
    case ('C')
      if (phi_v >= 0.0_dp) return
      cutoff = sqrt(max(0.0_dp, -phi_v/p%t_swe_ev)) - p%u
      call photoelectron_fluxes(p%photoelectrons, p%m_e_kg, 0.0_dp, outward, escape, returning)
      source_current_term = escape/(p%velocity_scale_mps/(2.0_dp*sqrt(pi)))
    case default
      return
    end select

    ion_term = p%n_swi_inf_m3*sqrt( &
        2.0_dp*pi*p%t_swe_ev/p%potential_scale_v*p%m_e_kg/p%m_i_kg &
        )*p%mach
    coefficient = swe_free_current_term(p, 1.0_dp, cutoff)

    if (.not. all(ieee_is_finite([source_current_term, ion_term, coefficient])) .or. &
        coefficient <= 0.0_dp) return

    density_m3 = (source_current_term + ion_term)/coefficient

    if (.not. ieee_is_finite(density_m3) .or. density_m3 <= 0.0_dp) return

    if (branch == 'B') then
      call zhao_residuals_type_b(p, [phi_v, density_m3], residual)
    else
      call zhao_residuals_type_c(p, [phi_v, density_m3], residual)
    end if

    if (.not. all(ieee_is_finite(residual))) return

    residual_v = residual(1)
    success = .true.
  end subroutine evaluate_monotonic_stationary_phi

  !> Evaluate Type-A neutrality, zero-current, and upstream field-squared residuals.
  !! x = [surface potential (V), minimum potential (V), electron normalization (m^-3)].
  !! f(1:2) have density units [m^-3]; f(3) is normalized E^2. Invalid trial states receive a penalty.
  subroutine zhao_residuals_type_a(p, x, f)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(in) :: x(:)
    real(dp), intent(out) :: f(:)
    real(dp) :: phi0, phim, density, rho, electron, ion, outward, escape, returning
    phi0 = x(1)
    phim = x(2)
    density = x(3)
    if (phim >= min(phi0, 0.0_dp) .or. density <= 0.0_dp) then
      f = 1.0e6_dp
      return
    end if
    call evaluate_zhao_rho_hat(p, 'A', 'upper', 0.0_dp, phi0/p%potential_scale_v, &
        phim/p%potential_scale_v, density/p%density_scale_m3, rho)
    call evaluate_zhao_fluxes(p, phi0, phim, density, electron, ion, outward, escape, returning)
    f(1) = -rho*p%density_scale_m3
    f(2) = (escape - electron + ion)/(p%velocity_scale_mps/(2.0_dp*sqrt(pi)))
    f(3) = type_a_e2_sum_at_infinity(p, phi0, phim, density)
  end subroutine zhao_residuals_type_a

  !> Evaluate Type-B neutrality and zero-current residuals f(1:2) in density units [m^-3].
  !! x = [positive surface potential (V), electron normalization (m^-3)]; invalid trials receive a penalty.
  subroutine zhao_residuals_type_b(p, x, f)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(in) :: x(:)
    real(dp), intent(out) :: f(:)
    real(dp) :: phi0, phim, density, rho, electron, ion, outward, escape, returning
    phi0 = x(1)
    phim = 0.0_dp
    density = x(2)
    if (phi0 <= 0.0_dp .or. density <= 0.0_dp) then
      f = 1.0e6_dp
      return
    end if
    call evaluate_zhao_rho_hat(p, 'B', 'upper', 0.0_dp, phi0/p%potential_scale_v, &
        phim/p%potential_scale_v, density/p%density_scale_m3, rho)
    call evaluate_zhao_fluxes(p, phi0, phim, density, electron, ion, outward, escape, returning)
    f(1) = -rho*p%density_scale_m3
    f(2) = (escape - electron + ion)/(p%velocity_scale_mps/(2.0_dp*sqrt(pi)))
  end subroutine zhao_residuals_type_b

  !> Evaluate Type-C neutrality and zero-current residuals f(1:2) in density units [m^-3].
  !! x = [negative surface potential (V), electron normalization (m^-3)]; invalid trials receive a penalty.
  subroutine zhao_residuals_type_c(p, x, f)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(in) :: x(:)
    real(dp), intent(out) :: f(:)
    real(dp) :: phi0, phim, density, rho, electron, ion, outward, escape, returning
    phi0 = x(1)
    phim = x(1)
    density = x(2)
    if (phi0 >= 0.0_dp .or. density <= 0.0_dp) then
      f = 1.0e6_dp
      return
    end if
    call evaluate_zhao_rho_hat(p, 'C', 'upper', 0.0_dp, phi0/p%potential_scale_v, &
        phim/p%potential_scale_v, density/p%density_scale_m3, rho)
    call evaluate_zhao_fluxes(p, phi0, phim, density, electron, ion, outward, escape, returning)
    f(1) = -rho*p%density_scale_m3
    f(2) = (escape - electron + ion)/(p%velocity_scale_mps/(2.0_dp*sqrt(pi)))
  end subroutine zhao_residuals_type_c

  !> Return the incoming-electron flux term in density units [m^-3] for normalization n_swe_inf_m3.
  !! a_swe is the thermal-speed cutoff minus p%u; multiply term by p%velocity_scale_mps/(2*sqrt(pi)) for flux.
  real(dp) function swe_free_current_term(p, n_swe_inf_m3, a_swe) result(term)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(in) :: n_swe_inf_m3, a_swe

    term = n_swe_inf_m3*(sqrt(p%t_swe_ev/p%potential_scale_v)*exp(-(a_swe*a_swe)) + &
        sqrt(pi)*(p%v_d_electron_mps/p%velocity_scale_mps)*erfc(a_swe))
  end function swe_free_current_term

  !> Return normalized upstream E^2 from the Type-A upper segment; a connecting root requires zero.
  !! Inputs are boundary/minimum potentials [V] and upstream electron normalization [m^-3].
  real(dp) function type_a_e2_sum_at_infinity(p, phi0_v, phi_m_v, n_swe_inf_m3) result(e2_sum)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(in) :: phi0_v, phi_m_v, n_swe_inf_m3

    ! Integrate the same orbit densities used in Poisson's equation. No 1/u term.
    e2_sum = -2.0_dp*integrate_zhao_rho(p, 'A', 'upper', phi_m_v/p%potential_scale_v, 0.0_dp, &
        phi0_v/p%potential_scale_v, phi_m_v/p%potential_scale_v, n_swe_inf_m3/p%density_scale_m3)
  end function type_a_e2_sum_at_infinity

  !> Integrate dimensionless charge density over potential from lo to hi on the selected branch and side.
  !! lo, hi, phi0 and phim are potentials / p%potential_scale_v; density is electron normalization / p%density_scale_m3.
  !! The result is dimensionless and changes sign when the integration bounds are reversed.
  real(dp) function integrate_zhao_rho(p, branch, side, lo, hi, phi0, phim, density) result(value)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    character(len=*), intent(in) :: side
    real(dp), intent(in) :: lo, hi, phi0, phim, density

    real(dp) :: t, phi, rho, ni, ne, nr, pe_integral
    logical :: lower_side
    integer :: panel, j
    value = 0.0_dp

    if (lo == hi) return

    do panel = 0, 3
      do j = 1, 16
        t = (real(panel, dp) + 0.5_dp*(1.0_dp + gauss_x(j)))/4.0_dp
        phi = lo + (hi - lo)*sin(0.5_dp*pi*t)**2
        if (p%photoelectrons%is_binned()) then
          call evaluate_background_hat(p, branch, side, phi, phim, density, ni, ne, nr)
          rho = ni - ne - nr
        else
          call evaluate_zhao_rho_hat(p, branch, side, phi, phi0, phim, density, rho)
        end if
        value = value + gauss_w(j)*rho*(hi - lo)*0.5_dp*pi*sin(pi*t)
      end do
    end do
    value = value/8.0_dp
    if (p%photoelectrons%is_binned()) then
      lower_side = branch == 'B' .or. (branch == 'A' .and. side == 'lower')
      pe_integral = photoelectron_density_integral(p%photoelectrons, p%m_e_kg, phi0*p%potential_scale_v, &
          phim*p%potential_scale_v, lo*p%potential_scale_v, hi*p%potential_scale_v, lower_side)
      value = value - pe_integral/(p%density_scale_m3*p%potential_scale_v)
    end if
  end function integrate_zhao_rho

  !> Electron normalization [m^-3] imposed by upstream neutrality at the supplied potentials [V].
  real(dp) function neutral_electron_density(p, branch, phi0, phim) result(density)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: phi0, phim
    real(dp) :: free, reflected, pe, captured, coefficient
    call electron_density(0.0_dp, phim/p%t_swe_ev, p%u, free, reflected)
    if (branch == 'B') then
      reflected = 0.0_dp
    end if
    coefficient = free + reflected
    call photoelectron_density(p%photoelectrons, p%m_e_kg, phi0, phim, 0.0_dp, .false., pe, captured)
    density = ieee_value(0.0_dp, ieee_quiet_nan)
    if (coefficient > 0.0_dp) then
      density = (p%n_swi_inf_m3 - pe)/coefficient
    end if
  end function

  !> Common boundary number fluxes [m^-2 s^-1] for potentials [V] and electron normalization [m^-3].
  subroutine evaluate_zhao_fluxes(p, phi0, phim, density, electron, ion, outward, escape, returning)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(in) :: phi0, phim, density
    real(dp), intent(out) :: electron, ion, outward, escape, returning
    real(dp) :: cutoff
    cutoff = sqrt(max(0.0_dp, -phim/p%t_swe_ev)) - p%u
    electron = p%velocity_scale_mps/(2.0_dp*sqrt(pi))*swe_free_current_term(p, density, cutoff)
    ion = p%n_swi_inf_m3*p%v_d_ion_mps
    call photoelectron_fluxes(p%photoelectrons, p%m_e_kg, phi0 - phim, outward, escape, returning)
  end subroutine

  subroutine evaluate_background_hat(p, branch, side, phi, phim, density, ni, ne, nr)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    character(len=*), intent(in) :: side
    real(dp), intent(in) :: phi, phim, density
    real(dp), intent(out) :: ni, ne, nr
    real(dp) :: arg, free, reflected
    arg = 1.0_dp - 2.0_dp*phi/(p%tau*p%mach**2)
    ni = ieee_value(0.0_dp, ieee_quiet_nan)
    ne = ni
    nr = ni
    if (arg <= 0.0_dp) return
    ni = (p%n_swi_inf_m3/p%density_scale_m3)/sqrt(arg)
    call electron_density(phi/p%tau, phim/p%tau, p%u, free, reflected)
    ne = density*free
    nr = density*reflected
    if (branch == 'B' .or. (branch == 'A' .and. side == 'lower')) then
      nr = 0.0_dp
    end if
  end subroutine
end module sheath_model_core
