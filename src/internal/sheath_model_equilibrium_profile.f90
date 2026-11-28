! SPDX-License-Identifier: MIT
!> Poisson profile reconstruction and physical density evaluation on an accepted root.
submodule(sheath_model_equilibrium) sheath_model_equilibrium_profile
  use sheath_model_constants, only: qe
  use sheath_model_core, only: evaluate_zhao_density_hat
  use sheath_model_ions, only: ion_density_ratio
  implicit none
contains

  module procedure evaluate_density
    type(zhao_params_type) :: p
    call prepare_params(input, p, status, message)
    if (status /= SHEATH_OK) return
    call evaluate_prepared_density(p, solution, potential_v, output, status, message, side)
  end procedure evaluate_density

  !> Use already validated physical inputs when evaluating many points on one profile.
  subroutine evaluate_prepared_density(p, solution, potential_v, output, status, message, side)
    type(zhao_params_type), intent(in) :: p
    type(sheath_equilibrium_result), intent(in) :: solution
    real(dp), intent(in) :: potential_v
    type(sheath_density_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    character(len=*), intent(in), optional :: side
    real(dp) :: d(5), upper
    character(len=9) :: region

    output = sheath_density_result()
    status = SHEATH_INVALID_ARGUMENT
    message = 'A valid equilibrium solution and finite potential are required.'
    if (.not. solution%valid) return
    if (.not. all(ieee_is_finite([potential_v, solution%surface_potential_v, solution%minimum_potential_v, &
        solution%ambient_electron_density_m3]))) return
    if (solution%ambient_electron_density_m3 <= 0.0_dp) return

    region = 'monotonic'
    upper = max(solution%surface_potential_v, 0.0_dp)

    select case (solution%branch)
    case ('A')
      message = 'Type A density requires side=lower or side=upper.'
      if (.not. present(side)) return

      region = lower_ascii(side)
      if (region /= 'lower' .and. region /= 'upper') return

      upper = solution%surface_potential_v
      if (region == 'upper') then
        upper = 0.0_dp
      end if
    case ('B', 'C')
    case default
      return
    end select

    message = 'Potential is outside the selected branch interval or blocks the ion fluid branch.'
    if (potential_v < solution%minimum_potential_v .or. potential_v > upper) return
    if (.not. ieee_is_finite(ion_density_ratio(potential_v, 0.5_dp*p%t_swe_ev*p%mach**2, &
        p%ion_pressure_factor*p%t_swi_ev))) return

    call evaluate_zhao_density_hat(p, solution%branch, region, potential_v/p%potential_scale_v, &
        solution%surface_potential_v/p%potential_scale_v, solution%minimum_potential_v/p%potential_scale_v, &
        solution%ambient_electron_density_m3/p%density_scale_m3, d(1), d(2), d(3), d(4), d(5))
    d = d*p%density_scale_m3
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Density evaluation is non-finite or negative.'
    if (.not. all(ieee_is_finite(d)) .or. any(d < 0.0_dp)) return

    output = sheath_density_result(d(1), d(2), d(3), d(4), d(5), qe*(d(1) - sum(d(2:5))))
    status = SHEATH_OK
    message = ''
  end subroutine evaluate_prepared_density

  module procedure build_profile
    type(zhao_params_type) :: p
    real(dp) :: raw(3), minimum_e2, boundary_e2, band
    type(sheath_profile_result) :: trial
    real(dp), allocatable :: phi(:), distance(:), rho(:), e2(:), z(:), v(:), e(:)
    real(dp) :: phi0, phim, cutoff, reference, turn
    integer :: n, i, total, kept
    logical :: real_field
    character(len=9) :: side

    call prepare_params(input, p, status, message)
    if (status /= SHEATH_OK) return
    status = SHEATH_INVALID_ARGUMENT
    message = 'Profile residual tolerance must be finite and positive.'
    if (.not. ieee_is_finite(residual_tolerance) .or. residual_tolerance <= 0.0_dp) return
    p%search%residual_tolerance = residual_tolerance
    if (present(upstream_band_tolerance)) p%search%upstream_band_tolerance = upstream_band_tolerance
    if (present(upstream_field_floor_v_m)) p%search%upstream_field_floor_v_m = upstream_field_floor_v_m
    message = 'Profile upstream band tolerance must be finite, nonnegative and below 1, and the field floor nonnegative.'
    if (.not. valid_search_options(p%search)) return

    status = SHEATH_INVALID_ARGUMENT
    message = 'Profile requires at least 32 points per segment and finite positive distance/cutoff.'
    if (options%points_per_segment < 32) return
    if (.not. all(ieee_is_finite([options%max_distance_m, options%potential_cutoff_v]))) return
    if (min(options%max_distance_m, options%potential_cutoff_v) <= 0.0_dp) return
    message = 'Profile upstream band margin must be finite, positive and below 1.'
    if (.not. ieee_is_finite(options%upstream_band_margin)) return
    if (options%upstream_band_margin <= 0.0_dp .or. options%upstream_band_margin >= 1.0_dp) return

    status = SHEATH_INVALID_ARGUMENT
    message = 'Profile requires a converged root for the supplied physical conditions.'
    if (.not. root%valid .or. index('ABC', root%branch) == 0) return
    if (root%branch == 'A' .and. root%minimum_potential_v >= min(root%surface_potential_v, 0.0_dp)) return
    if (root%branch == 'B' .and. root%minimum_potential_v /= 0.0_dp) return
    if (root%branch == 'C' .and. root%minimum_potential_v /= root%surface_potential_v) return
    call equilibrium_residual(p, root%branch, &
        [root%surface_potential_v, root%minimum_potential_v, root%ambient_electron_density_m3], raw)
    if (.not. all(ieee_is_finite(raw)) .or. maxval(abs(raw)) > p%search%residual_tolerance) return
    call validate_zhao_profile(p, root%branch, root%surface_potential_v/p%potential_scale_v, &
        root%minimum_potential_v/p%potential_scale_v, root%ambient_electron_density_m3/p%density_scale_m3, &
        minimum_e2, boundary_e2, status, message, band)
    if (status /= SHEATH_OK) return

    n = options%points_per_segment
    allocate (phi(n), distance(n), rho(n), e2(n))
    allocate (z(2*n), v(2*n), e(2*n))
    phi0 = root%surface_potential_v/p%potential_scale_v
    phim = root%minimum_potential_v/p%potential_scale_v
    reference = abs(merge(phim, phi0, root%branch == 'A'))
    cutoff = min(options%potential_cutoff_v/p%potential_scale_v, 0.5_dp*reference)
    ! Stop the upstream end before an accepted band of negative E^2.
    if (band > 0.0_dp) cutoff = max(cutoff, band + options%upstream_band_margin*(reference - band))
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Profile first integral is non-finite or does not support a real electric field.'
    turn = 0.0_dp
    if (root%branch == 'A') then
      call turning_segment('lower', phi0 - phim, real_field)
      if (.not. real_field) return
      turn = distance(n)
      z(:n) = turn - distance(n:1:-1)
      v(:n) = phi(n:1:-1)
      e(:n) = sqrt(e2(n:1:-1))

      call turning_segment('upper', -cutoff - phim, real_field)
      if (.not. real_field) return
      total = 2*n - 1
      z(n + 1:total) = turn + distance(2:n)
      v(n + 1:total) = phi(2:n)
      e(n + 1:total) = -sqrt(e2(2:n))
    else
      call monotonic_segment(real_field)
      if (.not. real_field) return
    end if
    z(1:total) = z(1:total)*p%length_scale_m
    v(1:total) = v(1:total)*p%potential_scale_v
    ! Preserve exact physical endpoints through the normalized-coordinate round trip.
    v(1) = root%surface_potential_v
    if (root%branch == 'A') then
      v(n) = root%minimum_potential_v
    end if
    e(1:total) = e(1:total)*p%potential_scale_v/p%length_scale_m
    if (.not. all(ieee_is_finite(z(1:total))) .or. .not. all(ieee_is_finite(e(1:total)))) return

    kept = count(z(1:total) <= options%max_distance_m)
    if (kept < 2) then
      status = SHEATH_INVALID_ARGUMENT
      message = 'max_distance_m contains fewer than two nodes; increase points_per_segment or distance.'
      return
    end if

    trial%equilibrium = root
    if (root%branch == 'A') then
      trial%turning_height_m = turn*p%length_scale_m
    end if
    trial%z_m = z(1:kept)
    trial%potential_v = v(1:kept)
    trial%electric_field_v_m = e(1:kept)
    allocate (trial%density(kept))
    do i = 1, kept
      side = 'monotonic'
      if (root%branch == 'A') then
        side = 'lower'
        if (i > n) then
          side = 'upper'
        end if
      end if
      call evaluate_prepared_density(p, root, v(i), trial%density(i), status, message, side)
      if (status /= SHEATH_OK) return
    end do
    output = trial
    status = SHEATH_OK
    message = ''

  contains

    !> Integrate away from a zero-field internal minimum on one side.
    subroutine turning_segment(region, delta, success)
      character(len=*), intent(in) :: region
      real(dp), intent(in) :: delta
      logical, intent(out) :: success
      integer :: i
      real(dp) :: t, part
      success = .false.
      do i = 1, n
        t = real(i - 1, dp)/real(n - 1, dp)
        phi(i) = phim + delta*t*t
        rho(i) = charge_at(phi(i), region)
      end do
      e2(1) = 0.0_dp
      distance(1) = 0.0_dp
      do i = 2, n
        part = phi(i) - phi(i - 1)
        e2(i) = e2(i - 1) - (rho(i) + rho(i - 1))*part
        if (.not. ieee_is_finite(e2(i)) .or. e2(i) <= 0.0_dp) return

        if (i == 2) then
          ! Integrate the 1/sqrt(phi-phi_min) endpoint analytically.
          distance(i) = 2.0_dp*part/sqrt(e2(i))
        else
          distance(i) = distance(i - 1) + 0.5_dp*part*(1.0_dp/sqrt(e2(i - 1)) + 1.0_dp/sqrt(e2(i)))
        end if
      end do
      success = .true.
    end subroutine turning_segment

    !> Integrate from the neutral endpoint, then omit the upstream cutoff.
    subroutine monotonic_segment(success)
      logical, intent(out) :: success
      integer :: i
      real(dp) :: t
      success = .false.
      total = 0
      do i = 1, n
        t = real(i - 1, dp)/real(n - 1, dp)
        phi(i) = phi0*(1.0_dp - t)**2
        rho(i) = charge_at(phi(i), 'monotonic')
      end do
      e2(n) = 0.0_dp
      do i = n - 1, 1, -1
        e2(i) = e2(i + 1) + (rho(i + 1) + rho(i))*(phi(i + 1) - phi(i))
      end do
      do i = 1, n - 1
        if (abs(phi(i)) < cutoff) exit
        if (.not. ieee_is_finite(e2(i)) .or. e2(i) <= 0.0_dp) return

        z(i) = 0.0_dp
        if (i > 1) then
          z(i) = z(i - 1) + 0.5_dp*abs(phi(i) - phi(i - 1))* &
              (1.0_dp/sqrt(e2(i)) + 1.0_dp/sqrt(e2(i - 1)))
        end if
        v(i) = phi(i)
        e(i) = sign(sqrt(e2(i)), phi0)
        total = i
      end do
      success = .true.
    end subroutine monotonic_segment

    real(dp) function charge_at(phi_hat, region) result(charge)
      real(dp), intent(in) :: phi_hat
      character(len=*), intent(in) :: region
      real(dp) :: d(5)
      call evaluate_zhao_density_hat(p, root%branch, region, phi_hat, phi0, phim, &
          root%ambient_electron_density_m3/p%density_scale_m3, d(1), d(2), d(3), d(4), d(5))
      charge = d(1) - sum(d(2:5))
    end function charge_at

  end procedure build_profile

end submodule sheath_model_equilibrium_profile
