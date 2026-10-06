! SPDX-License-Identifier: MIT
module sheath_model_atlas_physics
  use sheath_model_constants, only: dp, i32, pi, qe
  use sheath_model_core, only: zhao_params_type
  use sheath_model_atlas, only: sheath_equilibrium_atlas, sheath_atlas_point
  use sheath_model_photoelectrons, only: photoelectron_fluxes
  use sheath_model_equilibrium_physics, only: encoded_equilibrium_residual
  use sheath_model_coordinates, only: decode_unknowns
  use sheath_model_continuation, only: continue_guarded_system
  use sheath_model_search, only: sheath_search_diagnostics
  use sheath_model_admissibility, only: validate_zhao_profile
  use sheath_model_status, only: SHEATH_OK
  use sheath_model_ions, only: ion_critical_potential
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: equilibrium_key, atlas_equilibrium_seeds, continue_equilibrium_from_atlas
contains
  subroutine equilibrium_key(p, key, shape)
    type(zhao_params_type), intent(in) :: p
    real(dp), intent(out) :: key(6)
    real(dp), allocatable, intent(out) :: shape(:)
    real(dp) :: outward, escape, returning
    call photoelectron_fluxes(p%photoelectrons, p%m_e_kg, 0.0_dp, outward, escape, returning)
    key = [log(p%tau), asinh(p%u), log(p%v_d_ion_mps/p%velocity_scale_mps), log(p%m_i_kg/p%m_e_kg), &
        log(1.0_dp + p%ion_pressure_factor*p%t_swi_ev/p%potential_scale_v), &
        log(1.0_dp + 2.0_dp*sqrt(pi)*outward/(p%n_swi_inf_m3*p%velocity_scale_mps))]
    shape = p%photoelectrons%atlas_shape(p%potential_scale_v)
  end subroutine

  subroutine params_from_key(key, reference, p, valid)
    real(dp), intent(in) :: key(6)
    type(zhao_params_type), intent(in) :: reference
    type(zhao_params_type), intent(out) :: p
    logical, intent(out) :: valid
    real(dp) :: flux
    valid = all(ieee_is_finite(key)) .and. maxval(abs(key)) <= 100.0_dp .and. minval(key(5:6)) >= 0.0_dp
    p = reference
    if (.not. valid) return
    p%t_swe_ev = p%potential_scale_v*exp(key(1))
    p%v_swe_th_mps = p%velocity_scale_mps*sqrt(exp(key(1)))
    p%v_d_electron_mps = sinh(key(2))*p%v_swe_th_mps
    p%v_d_ion_mps = p%velocity_scale_mps*exp(key(3))
    p%m_i_kg = p%m_e_kg*exp(key(4))
    p%t_swi_ev = p%potential_scale_v*(exp(key(5)) - 1.0_dp)
    p%ion_pressure_factor = 1.0_dp
    flux = p%n_swi_inf_m3*p%velocity_scale_mps*(exp(key(6)) - 1.0_dp)/(2.0_dp*sqrt(pi))
    p%photoelectrons = reference%photoelectrons%with_outward_flux(flux, p%m_e_kg)
    p%cs_mps = sqrt(qe*p%t_swe_ev/p%m_i_kg)
    p%mach = p%v_d_ion_mps/p%cs_mps
    p%u = p%v_d_electron_mps/p%v_swe_th_mps
    p%tau = exp(key(1))
    valid = ieee_is_finite(ion_critical_potential(0.5_dp*p%t_swe_ev*p%mach**2, p%t_swi_ev))
  end subroutine

  subroutine atlas_equilibrium_seeds(p, branch, atlas, seeds)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    type(sheath_equilibrium_atlas), intent(in) :: atlas
    real(dp), allocatable, intent(out) :: seeds(:, :)
    real(dp), allocatable :: shape(:), predictions(:, :), trials(:, :)
    real(dp) :: key(6), physical(3)
    integer :: i, count
    logical :: valid
    call equilibrium_key(p, key, shape)
    call atlas%predictions(key, shape, branch, predictions)
    allocate (trials(3, size(predictions, 2)))
    count = 0
    do i = 1, size(predictions, 2)
      call decode_unknowns(p, branch, predictions(:, i), physical(1), physical(2), physical(3), valid)
      if (.not. valid) cycle
      count = count + 1
      trials(:, count) = physical
    end do
    seeds = trials(:, :count)
  end subroutine

  subroutine continue_equilibrium_from_atlas(p, branch, atlas, diagnostics, physical, success)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    type(sheath_equilibrium_atlas), intent(in) :: atlas
    type(sheath_search_diagnostics), intent(inout) :: diagnostics
    real(dp), intent(out) :: physical(3)
    logical, intent(out) :: success
    type(sheath_atlas_point) :: point
    real(dp), allocatable :: shape(:)
    integer, allocatable :: indices(:)
    real(dp) :: target(6), start(6), y(3)
    integer :: i, n, k
    logical :: valid
    call equilibrium_key(p, target, shape)
    call atlas%neighbors(target, shape, branch, indices)
    n = merge(3, 2, branch == 'A')
    k = index('ABC', branch)
    physical = 0.0_dp
    success = .false.
    do i = 1, size(indices)
      point = atlas%point(indices(i))
      start = point%key
      y = 0.0_dp
      call continue_guarded_system(n, path_residual, point%coordinates(:n), p%search, atlas%continuation, &
          diagnostics, k, y(:n), success, accept)
      if (.not. success) cycle
      call decode_unknowns(p, branch, y, physical(1), physical(2), physical(3), valid)
      if (valid .and. accept(y(:n), 1.0_dp)) then
        diagnostics%atlas_hits(k) = diagnostics%atlas_hits(k) + 1
        diagnostics%roots_found(k) = diagnostics%roots_found(k) + 1
        return
      end if
      success = .false.
    end do
  contains
    subroutine path_residual(value, t, f, ok)
      real(dp), intent(in) :: value(:), t
      real(dp), intent(out) :: f(:)
      logical, intent(out) :: ok
      type(zhao_params_type) :: current
      if (t == 1.0_dp) then
        current = p
        ok = .true.
      else
        call params_from_key(start + t*(target - start), p, current, ok)
      end if
      f = 0.0_dp
      if (ok) call encoded_equilibrium_residual(current, branch, value, f, ok)
    end subroutine
    logical function accept(value, t) result(ok)
      real(dp), intent(in) :: value(:), t
      type(zhao_params_type) :: current
      real(dp) :: encoded(3), trial(3), minimum_e2, boundary_e2
      integer(i32) :: status
      character(len=256) :: message
      if (t == 1.0_dp) then
        current = p
        ok = .true.
      else
        call params_from_key(start + t*(target - start), p, current, ok)
      end if
      if (.not. ok) return
      encoded = 0.0_dp
      encoded(:n) = value
      call decode_unknowns(current, branch, encoded, trial(1), trial(2), trial(3), ok)
      if (.not. ok) return
      call validate_zhao_profile(current, branch, trial(1)/current%potential_scale_v, trial(2)/current%potential_scale_v, &
          trial(3)/current%density_scale_m3, minimum_e2, boundary_e2, status, message)
      ok = status == SHEATH_OK
    end function
  end subroutine
end module sheath_model_atlas_physics
