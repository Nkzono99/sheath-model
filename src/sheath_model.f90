! SPDX-License-Identifier: MIT
!> Standalone Zhao sheath models: J=0, prescribed E_H, and static potential evaluation with spectral sources.
!! Public quantities use SI units except temperatures/normal energies [eV] and solar elevation [degrees].
!! The potential reference is zero at infinity; +z points from the boundary toward upstream.
!! Check status against SHEATH_OK before using results; message provides diagnostic details.
module sheath_model
  use sheath_model_photoelectrons, only: photoelectron_source, maxwellian_photoelectrons, binned_photoelectrons
  use sheath_model_state, only: zhao_plasma_input, zhao_state_result, evaluate_sheath_state
  use sheath_model_constants, only: dp, i32
  use sheath_model_status, only: SHEATH_OK, SHEATH_INVALID_ARGUMENT, &
      SHEATH_NO_PHYSICAL_SOLUTION, SHEATH_NUMERICAL_FAILURE, SHEATH_AMBIGUOUS_SOLUTION
  use sheath_model_equilibrium, only: zhao_equilibrium_input, zhao_equilibrium_result, &
      zhao_density_result, solve_equilibrium, evaluate_density, &
      zhao_profile_options, zhao_profile_result, solve_profile
  use sheath_model_field, only: zhao_field_input, zhao_field_result, solve_prescribed_field, solve_prescribed_field_candidates
  use sheath_model_field, only: zhao_field_search_diagnostics
  implicit none

  private

  public :: photoelectron_source, maxwellian_photoelectrons, binned_photoelectrons
  public :: zhao_plasma_input, zhao_state_result, evaluate_sheath_state
  public :: dp, i32, SHEATH_OK, SHEATH_INVALID_ARGUMENT, SHEATH_NO_PHYSICAL_SOLUTION
  public :: SHEATH_NUMERICAL_FAILURE, SHEATH_AMBIGUOUS_SOLUTION
  public :: zhao_equilibrium_input, zhao_equilibrium_result, zhao_density_result
  public :: solve_equilibrium, evaluate_density, solve_profile
  public :: zhao_profile_options, zhao_profile_result
  public :: zhao_field_input, zhao_field_result, solve_prescribed_field, solve_prescribed_field_candidates
  public :: zhao_field_search_diagnostics
end module sheath_model
