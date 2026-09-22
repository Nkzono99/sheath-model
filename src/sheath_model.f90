! SPDX-License-Identifier: MIT
!> Standalone Zhao sheath models: J=0 equilibrium and prescribed E_H.
module sheath_model
  use sheath_model_constants, only: dp, i32, sheath_ok, sheath_invalid_argument, &
                                    sheath_no_physical_solution, sheath_numerical_failure, sheath_ambiguous_solution
  use sheath_model_equilibrium, only: zhao_equilibrium_input, zhao_equilibrium_result, &
                                      zhao_density_result, solve_equilibrium, evaluate_density, &
                                      zhao_profile_options, zhao_profile_result, solve_profile
  use sheath_model_field, only: zhao_field_input, zhao_field_result, solve_prescribed_field
  implicit none
  private
  public :: dp, i32, sheath_ok, sheath_invalid_argument, sheath_no_physical_solution
  public :: sheath_numerical_failure, sheath_ambiguous_solution
  public :: zhao_equilibrium_input, zhao_equilibrium_result, zhao_density_result
  public :: solve_equilibrium, evaluate_density, solve_profile
  public :: zhao_profile_options, zhao_profile_result
  public :: zhao_field_input, zhao_field_result, solve_prescribed_field
end module sheath_model
