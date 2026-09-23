! SPDX-License-Identifier: MIT
!> Status codes returned by the public sheath-model API.
module sheath_model_status
  use sheath_model_constants, only: i32
  implicit none

  private

  !> The requested operation succeeded; message may still contain search diagnostics.
  integer(i32), parameter, public :: SHEATH_OK = 0_i32
  !> An input value, option, or evaluation point is outside the supported domain.
  integer(i32), parameter, public :: SHEATH_INVALID_ARGUMENT = 1_i32
  !> Physical conditions excluded the searched branches or roots; not a proof of global nonexistence.
  integer(i32), parameter, public :: SHEATH_NO_PHYSICAL_SOLUTION = 2_i32
  !> Convergence, finite-value checks, or numerical profile evaluation failed.
  integer(i32), parameter, public :: SHEATH_NUMERICAL_FAILURE = 3_i32
  !> A single result was requested, but multiple admissible candidates were found.
  integer(i32), parameter, public :: SHEATH_AMBIGUOUS_SOLUTION = 4_i32
end module sheath_model_status
