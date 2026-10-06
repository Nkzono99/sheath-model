! SPDX-License-Identifier: MIT
module sheath_model_equilibrium_physics
  use sheath_model_constants, only: dp
  use sheath_model_core, only: zhao_params_type, zhao_residuals_type_a, zhao_residuals_type_b, zhao_residuals_type_c, &
      type_a_connection_residual
  use sheath_model_coordinates, only: decode_unknowns
  use sheath_model_ions, only: ion_density_ratio
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: equilibrium_residual, encoded_equilibrium_residual
contains
  !> Normalize all equations to ion density; keep the A connection nondegenerate.
  subroutine equilibrium_residual(p, branch, physical, raw)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: physical(3)
    real(dp), intent(out) :: raw(3)
    raw = 0.0_dp
    select case (branch)
    case ('A')
      call zhao_residuals_type_a(p, physical, raw)
    case ('B')
      call zhao_residuals_type_b(p, physical([1, 3]), raw(1:2))
    case ('C')
      call zhao_residuals_type_c(p, physical([1, 3]), raw(1:2))
    end select
    raw(1:2) = raw(1:2)/p%n_swi_inf_m3
    if (branch == 'A') raw(3) = type_a_connection_residual(p, physical(1)/p%potential_scale_v, &
        physical(2)/p%potential_scale_v, physical(3)/p%density_scale_m3)
  end subroutine

  subroutine encoded_equilibrium_residual(p, branch, value, f, valid)
    type(zhao_params_type), intent(in) :: p
    character(len=1), intent(in) :: branch
    real(dp), intent(in) :: value(:)
    real(dp), intent(out) :: f(:)
    logical, intent(out) :: valid
    real(dp) :: encoded(3), physical(3), raw(3), ion
    encoded = 0.0_dp
    encoded(:size(value)) = value
    f = 0.0_dp
    call decode_unknowns(p, branch, encoded, physical(1), physical(2), physical(3), valid)
    if (.not. valid) return
    ion = ion_density_ratio(max(physical(1), 0.0_dp), 0.5_dp*p%t_swe_ev*p%mach**2, p%ion_pressure_factor*p%t_swi_ev)
    valid = ieee_is_finite(ion)
    if (.not. valid) return
    call equilibrium_residual(p, branch, physical, raw)
    f = raw(:size(f))
    valid = all(ieee_is_finite(f))
  end subroutine
end module
