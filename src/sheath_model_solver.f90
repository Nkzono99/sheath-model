! SPDX-License-Identifier: MIT
!> Numerical policy and read-only solution maps, independent of physical inputs.
module sheath_model_solver
  use sheath_model_constants, only: i32
  use sheath_model_search, only: sheath_search_options, sheath_search_diagnostics, sheath_continuation_options
  use sheath_model_atlas, only: sheath_equilibrium_atlas, sheath_field_atlas
  use sheath_model_equilibrium, only: sheath_equilibrium_input, sheath_equilibrium_result, &
      sheath_profile_options, sheath_profile_result, solve_equilibrium, solve_equilibrium_candidates, &
      build_profile, build_equilibrium_atlas, add_equilibrium_to_atlas
  use sheath_model_field, only: prescribed_field_input, prescribed_field_result, &
      solve_prescribed_field, solve_prescribed_field_candidates, build_field_atlas, add_field_to_atlas
  use sheath_model_status, only: SHEATH_OK
  implicit none
  private
  public :: sheath_solver

  !> Reusable configuration; all methods are read-only with respect to this object.
  !! Assign a map explicitly to attach a snapshot. No query registers roots or
  !! retains a previous solution. Atlas construction and registration are explicit.
  type :: sheath_solver
    type(sheath_search_options) :: search
    type(sheath_profile_options) :: profile
    type(sheath_continuation_options) :: continuation
    type(sheath_equilibrium_atlas), allocatable :: equilibrium_atlas
    type(sheath_field_atlas), allocatable :: field_atlas
  contains
    procedure :: solve_equilibrium => solver_equilibrium
    procedure :: solve_equilibrium_candidates => solver_equilibrium_candidates
    procedure :: solve_prescribed_field => solver_field
    procedure :: solve_prescribed_field_candidates => solver_field_candidates
    procedure :: solve_profile => solver_profile
    procedure :: build_profile => solver_build_profile
    procedure :: build_equilibrium_atlas => solver_build_equilibrium_atlas
    procedure :: build_field_atlas => solver_build_field_atlas
    procedure :: add_equilibrium_to_atlas => solver_add_equilibrium
    procedure :: add_field_to_atlas => solver_add_field
  end type
contains

  subroutine solver_equilibrium(self, input, output, status, message, diagnostics, initial_guesses)
    class(sheath_solver), intent(in) :: self
    class(sheath_equilibrium_input), intent(in) :: input
    type(sheath_equilibrium_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(sheath_search_diagnostics), intent(out), optional :: diagnostics
    type(sheath_equilibrium_result), intent(in), optional :: initial_guesses(:)
    call solve_equilibrium(input, output, status, message, diagnostics, initial_guesses, &
        self%equilibrium_atlas, self%search, self%continuation)
  end subroutine

  subroutine solver_equilibrium_candidates(self, input, candidates, status, message, diagnostics, &
      initial_guesses, deflation, max_roots)
    class(sheath_solver), intent(in) :: self
    class(sheath_equilibrium_input), intent(in) :: input
    type(sheath_equilibrium_result), allocatable, intent(out) :: candidates(:)
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(sheath_search_diagnostics), intent(out), optional :: diagnostics
    type(sheath_equilibrium_result), intent(in), optional :: initial_guesses(:)
    logical, intent(in), optional :: deflation
    integer, intent(in), optional :: max_roots
    call solve_equilibrium_candidates(input, candidates, status, message, diagnostics, &
        initial_guesses, self%equilibrium_atlas, deflation, max_roots, self%search, self%continuation)
  end subroutine

  subroutine solver_field(self, input, output, status, message, diagnostics, initial_guesses, deflation, max_roots)
    class(sheath_solver), intent(in) :: self
    type(prescribed_field_input), intent(in) :: input
    type(prescribed_field_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(sheath_search_diagnostics), intent(out), optional :: diagnostics
    type(prescribed_field_result), intent(in), optional :: initial_guesses(:)
    logical, intent(in), optional :: deflation
    integer, intent(in), optional :: max_roots
    call solve_prescribed_field(input, output, status, message, diagnostics, &
        initial_guesses, self%field_atlas, deflation, max_roots, self%search, self%continuation)
  end subroutine

  subroutine solver_field_candidates(self, input, candidates, status, message, diagnostics, &
      initial_guesses, deflation, max_roots)
    class(sheath_solver), intent(in) :: self
    type(prescribed_field_input), intent(in) :: input
    type(prescribed_field_result), allocatable, intent(out) :: candidates(:)
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(sheath_search_diagnostics), intent(out), optional :: diagnostics
    type(prescribed_field_result), intent(in), optional :: initial_guesses(:)
    logical, intent(in), optional :: deflation
    integer, intent(in), optional :: max_roots
    call solve_prescribed_field_candidates(input, candidates, status, message, diagnostics, &
        initial_guesses, self%field_atlas, deflation, max_roots, self%search, self%continuation)
  end subroutine

  subroutine solver_profile(self, input, output, status, message, diagnostics, initial_guesses)
    class(sheath_solver), intent(in) :: self
    class(sheath_equilibrium_input), intent(in) :: input
    type(sheath_profile_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(sheath_search_diagnostics), intent(out), optional :: diagnostics
    type(sheath_equilibrium_result), intent(in), optional :: initial_guesses(:)
    type(sheath_equilibrium_result) :: root
    call self%solve_equilibrium(input, root, status, message, diagnostics, initial_guesses)
    if (status /= SHEATH_OK) return
    call self%build_profile(input, root, output, status, message)
  end subroutine

  !> Reconstruct an existing root without repeating its search.
  subroutine solver_build_profile(self, input, root, output, status, message)
    class(sheath_solver), intent(in) :: self
    class(sheath_equilibrium_input), intent(in) :: input
    type(sheath_equilibrium_result), intent(in) :: root
    type(sheath_profile_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    call build_profile(input, root, self%profile, output, status, message, self%search%residual_tolerance)
  end subroutine

  subroutine solver_build_equilibrium_atlas(self, inputs, atlas, status, message, report, deflation)
    class(sheath_solver), intent(in) :: self
    class(sheath_equilibrium_input), intent(in) :: inputs(:)
    type(sheath_equilibrium_atlas), intent(inout) :: atlas
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    integer(i32), allocatable, intent(out), optional :: report(:, :)
    logical, intent(in), optional :: deflation
    call build_equilibrium_atlas(inputs, atlas, status, message, report, deflation, self%search, self%continuation)
  end subroutine

  subroutine solver_build_field_atlas(self, inputs, atlas, status, message, report, deflation, max_roots)
    class(sheath_solver), intent(in) :: self
    type(prescribed_field_input), intent(in) :: inputs(:)
    type(sheath_field_atlas), intent(inout) :: atlas
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    integer(i32), allocatable, intent(out), optional :: report(:, :)
    logical, intent(in), optional :: deflation
    integer, intent(in), optional :: max_roots
    call build_field_atlas(inputs, atlas, status, message, report, deflation, max_roots, self%search, self%continuation)
  end subroutine

  subroutine solver_add_equilibrium(self, input, root, atlas, status, message, component)
    class(sheath_solver), intent(in) :: self
    class(sheath_equilibrium_input), intent(in) :: input
    type(sheath_equilibrium_result), intent(in) :: root
    type(sheath_equilibrium_atlas), intent(inout) :: atlas
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    integer, intent(in), optional :: component
    call add_equilibrium_to_atlas(input, root, atlas, status, message, component, self%search, self%continuation)
  end subroutine

  subroutine solver_add_field(self, input, root, atlas, status, message, component)
    class(sheath_solver), intent(in) :: self
    type(prescribed_field_input), intent(in) :: input
    type(prescribed_field_result), intent(in) :: root
    type(sheath_field_atlas), intent(inout) :: atlas
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    integer, intent(in), optional :: component
    call add_field_to_atlas(input, root, atlas, status, message, component, self%search, self%continuation)
  end subroutine
end module sheath_model_solver
