! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone modules; status-returning public facade in sheath_model.
!> Zhao の初期推定、減衰 Newton 法、差分 Jacobian、小規模線形解法。
!! 残差の定義と物理的な許容条件は physics、最終的な根の選択は roots が担当する。
submodule(sheath_model_field) sheath_model_field_numerics
  implicit none

  integer, parameter :: root_max_iterations = 60
  integer, parameter :: root_max_backtracks = 24
  real(dp), parameter :: root_tolerance = 1.0e-9_dp

contains

  module procedure make_field_branch_guesses
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
  end procedure make_field_branch_guesses

  module procedure newton_field_branch

    real(dp) :: y(3), f(3), jac(3, 3), delta(3), trial(3), trial_f(3)
    real(dp) :: norm, trial_norm, step
    integer :: n, iteration, backtrack
    logical :: valid, jacobian_ok, linear_ok, trial_valid

    n = merge(3, 2, branch == 'A')
    y = y0
    call evaluate_charge_residual(params, branch, target_field_hat, y, f, valid)
    if (.not. valid) then
      y_out = y
      final_norm = huge(1.0_dp)
      iterations = 0
      success = .false.
      return
    end if
    norm = maxval(abs(f(1:n)))
    success = .false.
    do iteration = 0, root_max_iterations
      if (norm <= root_tolerance) then
        success = .true.
        exit
      end if
      if (iteration == root_max_iterations) exit
      call field_numerical_jacobian( &
          params, branch, target_field_hat, y, f, n, jac, jacobian_ok &
          )
      if (.not. jacobian_ok) exit
      call solve_field_small_system(jac, -f, n, delta, linear_ok)
      if (.not. linear_ok) exit
      step = 1.0_dp
      do backtrack = 1, root_max_backtracks
        trial = y + step*delta
        call evaluate_charge_residual( &
            params, branch, target_field_hat, trial, trial_f, trial_valid &
            )
        if (trial_valid) then
          trial_norm = maxval(abs(trial_f(1:n)))
          if (trial_norm < norm) then
            y = trial
            f = trial_f
            norm = trial_norm
            exit
          end if
        end if
        step = 0.5_dp*step
      end do
      if (backtrack > root_max_backtracks) exit
    end do
    y_out = y
    final_norm = norm
    iterations = iteration
  end procedure newton_field_branch

  subroutine field_numerical_jacobian( &
      params, branch, target_field_hat, y, f0, n, jac, success &
      )
    type(zhao_params_type), intent(in) :: params
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: target_field_hat, y(3), f0(3)
    integer, intent(in) :: n
    real(dp), intent(out) :: jac(3, 3)
    logical, intent(out) :: success

    real(dp) :: yp(3), ym(3), fp(3), fm(3), h
    integer :: column
    logical :: plus_valid, minus_valid

    jac = 0.0_dp
    success = .true.
    do column = 1, n
      h = epsilon(1.0_dp)**(1.0_dp/3.0_dp)*max(1.0_dp, abs(y(column)))
      yp = y
      ym = y
      yp(column) = yp(column) + h
      ym(column) = ym(column) - h
      call evaluate_charge_residual(params, branch, target_field_hat, yp, fp, plus_valid)
      call evaluate_charge_residual(params, branch, target_field_hat, ym, fm, minus_valid)
      if (plus_valid .and. minus_valid) then
        jac(1:n, column) = (fp(1:n) - fm(1:n))/(2.0_dp*h)
      else if (plus_valid) then
        jac(1:n, column) = (fp(1:n) - f0(1:n))/h
      else if (minus_valid) then
        jac(1:n, column) = (f0(1:n) - fm(1:n))/h
      else
        success = .false.
        return
      end if
    end do
    success = all(ieee_is_finite(jac(1:n, 1:n)))
  end subroutine field_numerical_jacobian

  subroutine solve_field_small_system(a_in, b_in, n, x, success)
    real(dp), intent(in) :: a_in(3, 3), b_in(3)
    integer, intent(in) :: n
    real(dp), intent(out) :: x(3)
    logical, intent(out) :: success

    real(dp) :: a(3, 3), b(3), factor, pivot_value, tmp
    integer :: i, j, k, pivot

    a = a_in
    b = b_in
    x = 0.0_dp
    success = .false.
    do k = 1, n
      pivot = k
      do i = k + 1, n
        if (abs(a(i, k)) > abs(a(pivot, k))) pivot = i
      end do
      if (.not. ieee_is_finite(a(pivot, k)) .or. abs(a(pivot, k)) <= 1.0e-14_dp) return
      if (pivot /= k) then
        do j = k, n
          tmp = a(k, j)
          a(k, j) = a(pivot, j)
          a(pivot, j) = tmp
        end do
        tmp = b(k)
        b(k) = b(pivot)
        b(pivot) = tmp
      end if
      pivot_value = a(k, k)
      do i = k + 1, n
        factor = a(i, k)/pivot_value
        a(i, k:n) = a(i, k:n) - factor*a(k, k:n)
        b(i) = b(i) - factor*b(k)
      end do
    end do
    do i = n, 1, -1
      x(i) = b(i)
      do j = i + 1, n
        x(i) = x(i) - a(i, j)*x(j)
      end do
      x(i) = x(i)/a(i, i)
    end do
    success = all(ieee_is_finite(x(1:n)))
  end subroutine solve_field_small_system

end submodule sheath_model_field_numerics
