! SPDX-License-Identifier: MIT
!> Status codes returned by the public sheath-model API.
module sheath_model_status
  use sheath_model_constants, only: i32
  implicit none
  private
  integer(i32), parameter, public :: SHEATH_OK = 0_i32
  integer(i32), parameter, public :: SHEATH_INVALID_ARGUMENT = 1_i32
  integer(i32), parameter, public :: SHEATH_NO_PHYSICAL_SOLUTION = 2_i32
  integer(i32), parameter, public :: SHEATH_NUMERICAL_FAILURE = 3_i32
  integer(i32), parameter, public :: SHEATH_AMBIGUOUS_SOLUTION = 4_i32
end module sheath_model_status
