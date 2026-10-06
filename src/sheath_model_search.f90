! SPDX-License-Identifier: MIT
!> Stateless search controls shared by equilibrium and prescribed-field models.
module sheath_model_search
  use sheath_model_constants, only: dp, lower_ascii
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: sheath_search_options, sheath_search_diagnostics, valid_search_options

  type :: sheath_search_options
    character(len=9) :: method = 'auto' ! auto, newton, lm, bracket (J=0 B/C only)
    real(dp) :: residual_tolerance = 1e-10_dp
    integer :: max_iterations = 100
    integer :: max_backtracks = 24
    integer :: max_starts = 32
    logical :: use_default_guesses = .true.
    integer :: bracket_points = 96
    real(dp) :: potential_extent = 200.0_dp ! finite search limit / potential_scale_v
  end type sheath_search_options

  !> Arrays are ordered A/B/C. A finite search does not prove root completeness.
  type :: sheath_search_diagnostics
    logical :: searched(3) = .false., excluded(3) = .false.
    integer :: starts(3) = 0, unconverged(3) = 0, rejected(3) = 0, profile_failures(3) = 0
    integer :: roots_found(3) = 0, evaluations(3) = 0, iterations(3) = 0, lm_steps(3) = 0, brackets(3) = 0
    real(dp) :: best_residual(3) = huge(1.0_dp)
  end type sheath_search_diagnostics

contains

  logical function valid_search_options(options) result(valid)
    type(sheath_search_options), intent(in) :: options
    valid = .false.
    select case (trim(lower_ascii(options%method)))
    case ('auto', 'newton', 'lm', 'bracket')
    case default
      return
    end select
    if (.not. all(ieee_is_finite([options%residual_tolerance, options%potential_extent]))) return
    valid = options%residual_tolerance > 0.0_dp .and. options%potential_extent > 0.0_dp .and. &
        options%max_iterations >= 0 .and. options%max_backtracks > 0 .and. options%max_starts > 0 .and. &
        options%bracket_points >= 2
  end function valid_search_options
end module sheath_model_search
