! SPDX-License-Identifier: MIT
!> Standalone photoelectron sheaths with fixed-entry or solar-illumination inputs.
!! J=0, prescribed E_H, and static potential evaluation support spectral sources.
!! Public quantities use SI units except temperatures/normal energies [eV] and solar elevation [degrees].
!! The potential reference is zero at infinity; +z points from the boundary toward upstream.
!! Check status against SHEATH_OK before using results; message provides diagnostic details.
module sheath_model
  use sheath_model_photoelectrons, only: photoelectron_source, maxwellian_photoelectrons, binned_photoelectrons
  use sheath_model_ions, only: ion_density_ratio, ion_critical_potential
  use sheath_model_state, only: plasma_input, sheath_state_result, evaluate_sheath_state
  use sheath_model_constants, only: dp, i32
  use sheath_model_status, only: SHEATH_OK, SHEATH_INVALID_ARGUMENT, &
      SHEATH_NO_PHYSICAL_SOLUTION, SHEATH_NUMERICAL_FAILURE, SHEATH_AMBIGUOUS_SOLUTION
  use sheath_model_equilibrium, only: sheath_equilibrium_input, zhao_equilibrium_input, fixed_entry_equilibrium_input, &
      sheath_equilibrium_result, &
      sheath_density_result, evaluate_density, &
      sheath_profile_options, sheath_profile_result
  use sheath_model_field, only: prescribed_field_input, prescribed_field_result
  use sheath_model_solver, only: sheath_solver
  use sheath_model_search, only: sheath_search_options, sheath_search_diagnostics, sheath_continuation_options
  use sheath_model_atlas, only: sheath_equilibrium_atlas, sheath_field_atlas, sheath_atlas_options, sheath_atlas_point
  implicit none

  private

  public :: photoelectron_source, maxwellian_photoelectrons, binned_photoelectrons
  public :: ion_density_ratio, ion_critical_potential
  public :: plasma_input, sheath_state_result, evaluate_sheath_state
  public :: dp, i32, SHEATH_OK, SHEATH_INVALID_ARGUMENT, SHEATH_NO_PHYSICAL_SOLUTION
  public :: SHEATH_NUMERICAL_FAILURE, SHEATH_AMBIGUOUS_SOLUTION
  public :: zhao_equilibrium_input, fixed_entry_equilibrium_input, sheath_equilibrium_result, sheath_density_result
  public :: sheath_solver, evaluate_density, sheath_equilibrium_input
  public :: sheath_profile_options, sheath_profile_result
  public :: prescribed_field_input, prescribed_field_result
  public :: sheath_search_options, sheath_search_diagnostics
  public :: sheath_equilibrium_atlas, sheath_atlas_options, sheath_atlas_point, sheath_continuation_options
  public :: sheath_field_atlas
end module sheath_model
