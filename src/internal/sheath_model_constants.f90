! SPDX-License-Identifier: MIT
module sheath_model_constants
  use, intrinsic :: iso_fortran_env, only: real64, int32
  implicit none
  private
  integer, parameter, public :: dp = real64, i32 = int32
  real(dp), parameter, public :: pi = acos(-1.0_dp)
  real(dp), parameter, public :: eps0 = 8.8541878128e-12_dp
  real(dp), parameter, public :: qe = 1.602176634e-19_dp
  real(dp), parameter, public :: electron_mass = 9.1093837015e-31_dp
  real(dp), parameter, public :: proton_mass = 1.67262192369e-27_dp
  public :: lower_ascii
contains
  pure function lower_ascii(value) result(lower)
    character(len=*), intent(in) :: value
    character(len=len(value)) :: lower
    integer :: i, code
    lower = value
    do i = 1, len(value)
      code = iachar(value(i:i))
      if (code >= iachar('A') .and. code <= iachar('Z')) lower(i:i) = achar(code + 32)
    end do
  end function lower_ascii
end module sheath_model_constants
