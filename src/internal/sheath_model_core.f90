! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone modules; status-returning public facade in sheath_model.
!> Zhao 系シース数値モデルの core 実装。
module sheath_model_core
  use sheath_model_constants, only: dp
  use sheath_model_constants, only: pi, eps0, qe
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_value, ieee_quiet_nan
  implicit none
  private

  real(dp), parameter :: nonlinear_tol = 1.0d-5
  integer, parameter :: nonlinear_max_iter = 60
  integer, parameter :: nonlinear_max_backtrack = 20

  abstract interface
    subroutine nonlinear_residual(x, f)
      import :: dp
      real(dp), intent(in) :: x(:)
      real(dp), intent(out) :: f(:)
    end subroutine nonlinear_residual
  end interface

  type :: zhao_params_type
    real(dp) :: alpha_rad = 0.0d0
    real(dp) :: n_swi_inf_m3 = 0.0d0
    real(dp) :: n_phe_ref_m3 = 0.0d0
    real(dp) :: n_phe0_m3 = 0.0d0
    real(dp) :: t_swe_ev = 0.0d0
    real(dp) :: t_phe_ev = 0.0d0
    real(dp) :: v_d_electron_mps = 0.0d0
    real(dp) :: v_d_ion_mps = 0.0d0
    real(dp) :: m_i_kg = 0.0d0
    real(dp) :: m_e_kg = 0.0d0
    real(dp) :: v_swe_th_mps = 0.0d0
    real(dp) :: v_phe_th_mps = 0.0d0
    real(dp) :: cs_mps = 0.0d0
    real(dp) :: mach = 0.0d0
    real(dp) :: u = 0.0d0
    real(dp) :: tau = 0.0d0
    real(dp) :: lambda_d_phe_ref_m = 0.0d0
  end type zhao_params_type

  public :: zhao_params_type
  public :: try_solve_zhao_unknowns
  public :: evaluate_zhao_density_hat
  public :: evaluate_zhao_rho_hat
  public :: zhao_residuals_type_a
  public :: zhao_residuals_type_b
  public :: zhao_residuals_type_c
  public :: swe_free_current_term
  public :: type_a_e2_sum_at_infinity

contains

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

  subroutine evaluate_zhao_density_hat( &
    p, branch, side, phi_hat, phi0_hat, phi_m_hat, n_swe_inf_hat, &
    n_swi_hat, n_swe_f_hat, n_swe_r_hat, n_phe_f_hat, n_phe_c_hat &
    )
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    character(len=*), intent(in) :: side
    real(dp), intent(in) :: phi_hat, phi0_hat, phi_m_hat, n_swe_inf_hat
    real(dp), intent(out) :: n_swi_hat, n_swe_f_hat, n_swe_r_hat, n_phe_f_hat, n_phe_c_hat

    real(dp) :: arg_ion, s_swe, s_phe, source_density_hat

    source_density_hat = p%n_phe0_m3/p%n_phe_ref_m3
    arg_ion = 1.0d0 - 2.0d0*phi_hat/(p%tau*p%mach*p%mach)
    n_swi_hat = ieee_value(0.0_dp, ieee_quiet_nan)
    n_swe_f_hat = n_swi_hat
    n_swe_r_hat = n_swi_hat
    n_phe_f_hat = n_swi_hat
    n_phe_c_hat = n_swi_hat
    if (arg_ion <= 0.0_dp) return
    n_swi_hat = (p%n_swi_inf_m3/p%n_phe_ref_m3)*arg_ion**(-0.5d0)

    select case (branch)
    case ('A')
      s_swe = sqrt(max(0.0d0, (phi_hat - phi_m_hat)/p%tau))
      s_phe = sqrt(max(0.0d0, phi_hat - phi_m_hat))
      n_swe_f_hat = 0.5d0*n_swe_inf_hat*exp(phi_hat/p%tau)*(1.0d0 - erf(s_swe - p%u))
      n_phe_f_hat = 0.5d0*source_density_hat*exp(phi_hat - phi0_hat)*(1.0d0 - erf(s_phe))
      if (trim(side) == 'lower') then
        n_swe_r_hat = 0.0d0
        n_phe_c_hat = source_density_hat*exp(phi_hat - phi0_hat)*erf(s_phe)
      else if (trim(side) == 'upper') then
        n_swe_r_hat = n_swe_inf_hat*exp(phi_hat/p%tau)*(erf(s_swe - p%u) + erf(p%u))
        n_phe_c_hat = 0.0d0
      else
        return
      end if
    case ('B')
      s_phe = sqrt(max(0.0d0, phi_hat))
      n_swe_f_hat = 0.5d0*n_swe_inf_hat*exp(phi_hat/p%tau)*(1.0d0 + erf(p%u))
      n_swe_r_hat = 0.0d0
      n_phe_f_hat = 0.5d0*source_density_hat*exp(phi_hat - phi0_hat)*(1.0d0 - erf(s_phe))
      n_phe_c_hat = source_density_hat*exp(phi_hat - phi0_hat)*erf(s_phe)
    case ('C')
      s_swe = sqrt(max(0.0d0, (phi_hat - phi0_hat)/p%tau))
      s_phe = sqrt(max(0.0d0, phi_hat - phi0_hat))
      n_swe_f_hat = 0.5d0*n_swe_inf_hat*exp(phi_hat/p%tau)*(1.0d0 - erf(s_swe - p%u))
      n_swe_r_hat = n_swe_inf_hat*exp(phi_hat/p%tau)*(erf(s_swe - p%u) + erf(p%u))
      n_phe_f_hat = 0.5d0*source_density_hat*exp(phi_hat - phi0_hat)*erfc(s_phe)
      n_phe_c_hat = 0.0d0
    case default
      return
    end select
  end subroutine evaluate_zhao_density_hat

  !> Zhao の零電流定常根を fail-closed な status 付きで探索する。
  subroutine try_solve_zhao_unknowns(model, p, phi0_v, phi_m_v, n_swe_inf_m3, branch, success)
    character(len=*), intent(in) :: model
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(out) :: phi0_v, phi_m_v, n_swe_inf_m3
    character(len=1), intent(out) :: branch
    logical, intent(out) :: success

    real(dp) :: x3(3), x2(2)
    character(len=1), dimension(3) :: order
    integer :: i

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
        phi_m_v = x2(1)
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
    case ('zhao_auto')
      if (p%alpha_rad*180.0d0/pi < 20.0d0) then
        order = ['C', 'A', 'B']
      else
        order = ['A', 'B', 'C']
      end if
    case default
      return
    end select

    do i = 1, size(order)
      select case (order(i))
      case ('A')
        call try_solve_zhao_branch_a(p, x3, success)
        if (success) then
          phi0_v = x3(1)
          phi_m_v = x3(2)
          n_swe_inf_m3 = x3(3)
          branch = 'A'
          success = .true.
          return
        end if
      case ('B')
        call try_solve_zhao_branch_b(p, x2, success)
        if (success) then
          phi0_v = x2(1)
          phi_m_v = x2(1)
          n_swe_inf_m3 = x2(2)
          branch = 'B'
          success = .true.
          return
        end if
      case ('C')
        call try_solve_zhao_branch_c(p, x2, success)
        if (success) then
          phi0_v = x2(1)
          phi_m_v = x2(1)
          n_swe_inf_m3 = x2(2)
          branch = 'C'
          success = .true.
          return
        end if
      end select
    end do

  end subroutine try_solve_zhao_unknowns

  subroutine try_solve_zhao_branch_a(p, x, success)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(out) :: x(3)
    logical, intent(out) :: success

    real(dp) :: guesses(3, 3)

    guesses(:, 1) = [3.6d0, -0.5d0, 8.2d6]
    guesses(:, 2) = [2.8d0, -0.3d0, 8.0d6]
    guesses(:, 3) = [4.5d0, -0.8d0, 8.4d6]
    call solve_nonlinear_system(3, guesses, residual_a, x, success)

  contains

    subroutine residual_a(xa, fa)
      real(dp), intent(in) :: xa(:)
      real(dp), intent(out) :: fa(:)

      call zhao_residuals_type_a(p, xa, fa)
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
    voltage_scale = max(p%t_phe_ev, p%t_swe_ev, 1.0_dp)
    residual_scale = max(p%n_swi_inf_m3, p%n_phe0_m3, 1.0_dp)
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
              max(nonlinear_tol, 1024.0_dp*epsilon(1.0_dp)*residual_scale)
  end subroutine try_solve_zhao_monotonic_scalar

  subroutine evaluate_monotonic_stationary_phi(p, branch, phi_v, residual_v, density_m3, success)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: phi_v
    real(dp), intent(out) :: residual_v, density_m3
    logical, intent(out) :: success

    real(dp) :: cutoff, ion_term, source_current_term, coefficient, residual(2)

    residual_v = huge(1.0_dp)
    density_m3 = 0.0_dp
    success = .false.
    select case (branch)
    case ('B')
      if (phi_v <= 0.0_dp) return
      cutoff = -p%u
      source_current_term = p%n_phe0_m3*exp(-phi_v/p%t_phe_ev)
    case ('C')
      if (phi_v >= 0.0_dp) return
      cutoff = sqrt(max(0.0_dp, -phi_v/p%t_swe_ev)) - p%u
      source_current_term = p%n_phe0_m3
    case default
      return
    end select
    ion_term = p%n_swi_inf_m3*sqrt( &
               2.0_dp*pi*p%t_swe_ev/p%t_phe_ev*p%m_e_kg/p%m_i_kg &
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

  subroutine zhao_residuals_type_a(p, x, f)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(in) :: x(:)
    real(dp), intent(out) :: f(:)

    real(dp) :: phi0_v, phi_m_v, n_swe_inf_m3, a_swe, a_phe, ion_term

    phi0_v = x(1)
    phi_m_v = x(2)
    n_swe_inf_m3 = x(3)
    if (phi0_v <= 0.0d0 .or. phi_m_v >= 0.0d0 .or. phi_m_v >= phi0_v .or. n_swe_inf_m3 <= 0.0d0) then
      f = 1.0d6
      return
    end if

    a_swe = sqrt(max(0.0d0, -phi_m_v/p%t_swe_ev)) - p%u
    a_phe = sqrt(max(0.0d0, -phi_m_v/p%t_phe_ev))
    ion_term = p%n_swi_inf_m3*sqrt(2.0d0*pi*p%t_swe_ev/p%t_phe_ev*p%m_e_kg/p%m_i_kg)*p%mach

    f(1) = 0.5d0*n_swe_inf_m3*(1.0d0 + 2.0d0*erf(p%u) + erf(a_swe)) + &
           0.5d0*p%n_phe0_m3* &
           exp(-phi0_v/p%t_phe_ev)*(1.0d0 - erf(a_phe)) - p%n_swi_inf_m3
    f(2) = p%n_phe0_m3*exp((phi_m_v - phi0_v)/p%t_phe_ev) - swe_free_current_term(p, n_swe_inf_m3, a_swe) + ion_term
    f(3) = type_a_e2_sum_at_infinity(p, phi0_v, phi_m_v, n_swe_inf_m3)
  end subroutine zhao_residuals_type_a

  subroutine zhao_residuals_type_b(p, x, f)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(in) :: x(:)
    real(dp), intent(out) :: f(:)

    real(dp) :: phi0_v, n_swe_inf_m3, ion_term

    phi0_v = x(1)
    n_swe_inf_m3 = x(2)
    if (phi0_v <= 0.0d0 .or. n_swe_inf_m3 <= 0.0d0) then
      f = 1.0d6
      return
    end if

    ion_term = p%n_swi_inf_m3*sqrt(2.0d0*pi*p%t_swe_ev/p%t_phe_ev*p%m_e_kg/p%m_i_kg)*p%mach
    f(1) = 0.5d0*n_swe_inf_m3*(1.0d0 + erf(p%u)) + &
           0.5d0*p%n_phe0_m3*exp(-phi0_v/p%t_phe_ev) - p%n_swi_inf_m3
    f(2) = p%n_phe0_m3*exp(-phi0_v/p%t_phe_ev) - swe_free_current_term(p, n_swe_inf_m3, -p%u) + ion_term
  end subroutine zhao_residuals_type_b

  subroutine zhao_residuals_type_c(p, x, f)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(in) :: x(:)
    real(dp), intent(out) :: f(:)

    real(dp) :: phi0_v, n_swe_inf_m3, a_swe, a_phe, ion_term

    phi0_v = x(1)
    n_swe_inf_m3 = x(2)
    if (phi0_v >= 0.0d0 .or. n_swe_inf_m3 <= 0.0d0) then
      f = 1.0d6
      return
    end if

    a_swe = sqrt(max(0.0d0, -phi0_v/p%t_swe_ev)) - p%u
    a_phe = sqrt(max(0.0d0, -phi0_v/p%t_phe_ev))
    ion_term = p%n_swi_inf_m3*sqrt(2.0d0*pi*p%t_swe_ev/p%t_phe_ev*p%m_e_kg/p%m_i_kg)*p%mach

    f(1) = 0.5d0*n_swe_inf_m3*(1.0d0 + 2.0d0*erf(p%u) + erf(a_swe)) + &
           0.5d0*p%n_phe0_m3* &
           exp(-phi0_v/p%t_phe_ev)*erfc(a_phe) - p%n_swi_inf_m3
    f(2) = p%n_phe0_m3 - swe_free_current_term(p, n_swe_inf_m3, a_swe) + ion_term
  end subroutine zhao_residuals_type_c

  real(dp) function swe_free_current_term(p, n_swe_inf_m3, a_swe) result(term)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(in) :: n_swe_inf_m3, a_swe

    term = n_swe_inf_m3*(sqrt(p%t_swe_ev/p%t_phe_ev)*exp(-(a_swe*a_swe)) + &
                         sqrt(pi)*(p%v_d_electron_mps/p%v_phe_th_mps)*erfc(a_swe))
  end function swe_free_current_term

  real(dp) function type_a_e2_sum_at_infinity(p, phi0_v, phi_m_v, n_swe_inf_m3) result(e2_sum)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(in) :: phi0_v, phi_m_v, n_swe_inf_m3

    real(dp) :: phi, s_swe, s_phe, e2_swe_f, e2_swe_r, e2_phe_f, e2_swi, arg_phi, arg_m

    if (abs(p%u) <= 1.0d-12) then
      e2_sum = 1.0d30
      return
    end if

    phi = 0.0d0
    s_swe = sqrt(max(0.0d0, (phi - phi_m_v)/p%t_swe_ev))
    s_phe = sqrt(max(0.0d0, (phi - phi_m_v)/p%t_phe_ev))

    e2_swe_f = (p%t_swe_ev/p%t_phe_ev)*(n_swe_inf_m3/p%n_phe_ref_m3)*( &
               exp(phi/p%t_swe_ev)*(1.0d0 - erf(s_swe - p%u)) - &
               exp(phi_m_v/p%t_swe_ev)*(1.0d0 - erf(-p%u)) + &
               (1.0d0/(sqrt(pi)*p%u))*exp(phi_m_v/p%t_swe_ev - p%u*p%u)*(exp(2.0d0*p%u*s_swe) - 1.0d0) &
               )

    e2_swe_r = 2.0d0*(p%t_swe_ev/p%t_phe_ev)*(n_swe_inf_m3/p%n_phe_ref_m3)*( &
               exp(phi/p%t_swe_ev)*(erf(s_swe - p%u) + erf(p%u)) - &
               (1.0d0/(sqrt(pi)*p%u))*exp(phi_m_v/p%t_swe_ev - p%u*p%u)*(exp(2.0d0*p%u*s_swe) - 1.0d0) &
               )

    e2_phe_f = (p%n_phe0_m3/p%n_phe_ref_m3)*( &
               exp((phi - phi0_v)/p%t_phe_ev)*(1.0d0 - erf(s_phe)) - &
               exp((phi_m_v - phi0_v)/p%t_phe_ev)*(1.0d0 - 2.0d0*s_phe/sqrt(pi)) &
               )

    arg_phi = 1.0d0 - 2.0d0*phi/(p%t_swe_ev*p%mach*p%mach)
    arg_m = 1.0d0 - 2.0d0*phi_m_v/(p%t_swe_ev*p%mach*p%mach)
    if (arg_phi <= 0.0d0 .or. arg_m <= 0.0d0) then
      e2_sum = 1.0d30
      return
    end if

    e2_swi = 2.0d0*(p%t_swe_ev/p%t_phe_ev)*(p%n_swi_inf_m3/p%n_phe_ref_m3)*p%mach*p%mach*( &
             sqrt(arg_phi) - sqrt(arg_m) &
             )
    e2_sum = e2_swe_f + e2_swe_r + e2_phe_f + e2_swi
  end function type_a_e2_sum_at_infinity

  subroutine solve_nonlinear_system(n, guesses, residual_fn, x_best, success)
    integer, intent(in) :: n
    real(dp), intent(in) :: guesses(:, :)
    procedure(nonlinear_residual) :: residual_fn
    real(dp), intent(out) :: x_best(n)
    logical, intent(out) :: success

    integer :: guess_idx
    real(dp) :: x_trial(n), best_norm, trial_norm
    logical :: trial_success

    success = .false.
    x_best = 0.0_dp
    if (size(guesses, 1) /= n .or. size(guesses, 2) == 0) return
    x_best = guesses(:, 1)
    best_norm = huge(1.0d0)
    do guess_idx = 1, size(guesses, 2)
      call try_newton_solve(n, guesses(:, guess_idx), residual_fn, x_trial, trial_norm, trial_success)
      if (trial_norm < best_norm) then
        best_norm = trial_norm
        x_best = x_trial
      end if
      if (trial_success .and. trial_norm < nonlinear_tol) then
        success = .true.
        x_best = x_trial
        return
      end if
    end do

    success = best_norm < nonlinear_tol
  end subroutine solve_nonlinear_system

  subroutine try_newton_solve(n, x0, residual_fn, x_out, final_norm, success)
    integer, intent(in) :: n
    real(dp), intent(in) :: x0(n)
    procedure(nonlinear_residual) :: residual_fn
    real(dp), intent(out) :: x_out(n)
    real(dp), intent(out) :: final_norm
    logical, intent(out) :: success

    integer :: iter, backtrack
    real(dp) :: x(n), f(n), jac(n, n), dx(n), x_trial(n), f_trial(n), step_scale, fnorm, trial_norm
    logical :: linear_ok, improved

    x = x0
    call residual_fn(x, f)
    fnorm = residual_norm(f)
    do iter = 1, nonlinear_max_iter
      if (fnorm < nonlinear_tol) exit
      call numerical_jacobian(n, x, f, residual_fn, jac)
      call solve_small_linear_system(n, jac, -f, dx, linear_ok)
      if (.not. linear_ok) exit

      step_scale = 1.0d0
      improved = .false.
      do backtrack = 1, nonlinear_max_backtrack
        x_trial = x + step_scale*dx
        call residual_fn(x_trial, f_trial)
        trial_norm = residual_norm(f_trial)
        if (trial_norm < fnorm) then
          x = x_trial
          f = f_trial
          fnorm = trial_norm
          improved = .true.
          exit
        end if
        step_scale = 0.5d0*step_scale
      end do
      if (.not. improved) exit
    end do

    x_out = x
    final_norm = fnorm
    success = fnorm < nonlinear_tol
  end subroutine try_newton_solve

  subroutine numerical_jacobian(n, x, f0, residual_fn, jac)
    integer, intent(in) :: n
    real(dp), intent(in) :: x(n), f0(n)
    procedure(nonlinear_residual) :: residual_fn
    real(dp), intent(out) :: jac(n, n)

    integer :: j
    real(dp) :: h, xh(n), fh(n)

    do j = 1, n
      h = 1.0d-6*max(1.0d0, abs(x(j)))
      xh = x
      xh(j) = xh(j) + h
      call residual_fn(xh, fh)
      jac(:, j) = (fh - f0)/h
    end do
  end subroutine numerical_jacobian

  subroutine solve_small_linear_system(n, a_in, b_in, x, ok)
    integer, intent(in) :: n
    real(dp), intent(in) :: a_in(n, n), b_in(n)
    real(dp), intent(out) :: x(n)
    logical, intent(out) :: ok

    integer :: i, j, k, pivot_row
    real(dp) :: a(n, n), b(n), factor, pivot_abs, tmp_row(n), tmp_val

    a = a_in
    b = b_in
    ok = .true.

    do k = 1, n
      pivot_row = k
      pivot_abs = abs(a(k, k))
      do i = k + 1, n
        if (abs(a(i, k)) > pivot_abs) then
          pivot_abs = abs(a(i, k))
          pivot_row = i
        end if
      end do
      if (pivot_abs <= 1.0d-18) then
        ok = .false.
        x = 0.0d0
        return
      end if
      if (pivot_row /= k) then
        tmp_row = a(k, :)
        a(k, :) = a(pivot_row, :)
        a(pivot_row, :) = tmp_row
        tmp_val = b(k)
        b(k) = b(pivot_row)
        b(pivot_row) = tmp_val
      end if
      do i = k + 1, n
        factor = a(i, k)/a(k, k)
        a(i, k:n) = a(i, k:n) - factor*a(k, k:n)
        b(i) = b(i) - factor*b(k)
      end do
    end do

    x = 0.0d0
    do i = n, 1, -1
      x(i) = b(i)
      do j = i + 1, n
        x(i) = x(i) - a(i, j)*x(j)
      end do
      x(i) = x(i)/a(i, i)
    end do
  end subroutine solve_small_linear_system

  real(dp) function residual_norm(f) result(norm2)
    real(dp), intent(in) :: f(:)

    if (.not. all(ieee_is_finite(f))) then
      norm2 = huge(1.0d0)
      return
    end if
    norm2 = sqrt(sum(f*f))
  end function residual_norm

end module sheath_model_core
