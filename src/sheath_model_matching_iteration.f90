! SPDX-License-Identifier: MIT
!> Outer J=0 root that a simulator re-solves from the electrons it observes leaving its matching plane.
!! The plasma upstream of the plane and the solver settings are configuration: the caller supplies them
!! on initialize and on read_state. Only the emitted source, the accepted root and counters form the state.
module sheath_model_matching_iteration
  use sheath_model_constants, only: dp, i32, lower_ascii
  use sheath_model_status, only: SHEATH_OK, SHEATH_INVALID_ARGUMENT, SHEATH_NUMERICAL_FAILURE
  use sheath_model_emission, only: emission_source, validate_emission, &
      write_emission_source, read_emission_source
  use sheath_model_state, only: plasma_input
  use sheath_model_equilibrium, only: fixed_entry_equilibrium_input, sheath_equilibrium_result
  use sheath_model_solver, only: sheath_solver
  use sheath_model_matching, only: sheath_matching_conditions, evaluate_matching_conditions
  implicit none
  private
  public :: sheath_matching_iteration

  integer, parameter :: state_version = 1
  character(len=3), parameter :: branches = 'ABC'

  !> J=0 outer root that follows the observed emitted source.
  !! An explicit branch is the only one tried. With auto, initialize tries initial_order and
  !! update tries the current branch first, then the rest in A, B, C order, seeded by the current root.
  type :: sheath_matching_iteration
    private
    logical :: initialized = .false.
    character(len=4) :: branch = 'auto'
    type(fixed_entry_equilibrium_input) :: input
    type(sheath_solver) :: solver
    type(sheath_equilibrium_result) :: root
    type(sheath_matching_conditions) :: matching
    integer(i32) :: solves = 0_i32
    real(dp) :: change = 0.0_dp
  contains
    procedure :: initialize
    procedure :: update
    procedure :: conditions => iteration_conditions
    procedure :: source => iteration_source
    procedure :: accepted_solves => iteration_accepted_solves
    procedure :: relative_change => iteration_relative_change
    procedure :: write_state
    procedure :: read_state
  end type sheath_matching_iteration

contains

  !> Solve the first root with plasma%emission as the emitted source.
  !! branch is auto (default), A, B or C; initial_order is a permutation of ABC used by auto (default ABC).
  subroutine initialize(self, plasma, solver, status, message, branch, initial_order)
    class(sheath_matching_iteration), intent(out) :: self
    type(plasma_input), intent(in) :: plasma
    type(sheath_solver), intent(in) :: solver
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    character(len=*), intent(in), optional :: branch, initial_order
    character(len=3) :: order
    type(sheath_equilibrium_result) :: no_guesses(0)

    order = 'ABC'
    if (present(initial_order)) order = upper(initial_order)
    if (present(branch)) self%branch = lower_ascii(branch)
    status = SHEATH_INVALID_ARGUMENT
    message = 'branch must be auto, A, B, or C.'
    select case (trim(self%branch))
    case ('auto', 'a', 'b', 'c')
    case default
      return
    end select
    message = 'initial_order must be a permutation of ABC.'
    if (present(initial_order)) then
      if (len_trim(initial_order) /= 3 .or. index(order, 'A') == 0 .or. index(order, 'B') == 0 .or. &
          index(order, 'C') == 0) return
    end if
    ! An explicit branch is the only one tried; initial_order applies to auto.
    if (trim(self%branch) /= 'auto') order = upper(self%branch(1:1))
    self%input%plasma = plasma
    self%solver = solver
    call solve_in_order(self, order, no_guesses, self%root, self%matching, status, message)
    if (status /= SHEATH_OK) return
    self%solves = 1_i32
    self%change = 0.0_dp
    self%initialized = .true.
  end subroutine initialize

  !> Re-solve with a new emitted source. On failure the previous source, root and conditions are kept
  !! and status reports why; relative_change is updated only by accepted solves.
  subroutine update(self, source, status, message)
    class(sheath_matching_iteration), intent(inout) :: self
    type(emission_source), intent(in) :: source
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(fixed_entry_equilibrium_input) :: previous_input
    type(sheath_equilibrium_result) :: root
    type(sheath_matching_conditions) :: matching
    character(len=3) :: order
    integer :: i

    status = SHEATH_INVALID_ARGUMENT
    message = 'Matching iteration must be initialized before update.'
    if (.not. self%initialized) return
    call validate_emission(source, status, message)
    if (status /= SHEATH_OK) return
    order = upper(self%branch(1:1))
    if (trim(self%branch) == 'auto') then
      order = self%root%branch
      do i = 1, 3
        if (index(order, branches(i:i)) == 0) order = trim(order)//branches(i:i)
      end do
    end if
    previous_input = self%input
    self%input%plasma%emission = source
    call solve_in_order(self, order, [self%root], root, matching, status, message)
    if (status /= SHEATH_OK) then
      self%input = previous_input
      return
    end if
    self%change = max(relative_difference(matching%emitted_outward_flux_m2_s, self%matching%emitted_outward_flux_m2_s), &
        relative_difference(source%mean_normal_energy(), previous_input%plasma%emission%mean_normal_energy()))
    self%root = root
    self%matching = matching
    self%solves = self%solves + 1_i32
  end subroutine update

  !> Conditions of the accepted root; valid is false before initialization.
  type(sheath_matching_conditions) function iteration_conditions(self) result(conditions)
    class(sheath_matching_iteration), intent(in) :: self
    conditions = self%matching
  end function iteration_conditions

  !> Emitted source of the accepted root.
  type(emission_source) function iteration_source(self) result(source)
    class(sheath_matching_iteration), intent(in) :: self
    source = self%input%plasma%emission
  end function iteration_source

  !> Accepted solves including the initial one; restored by read_state.
  integer(i32) function iteration_accepted_solves(self) result(solves)
    class(sheath_matching_iteration), intent(in) :: self
    solves = self%solves
  end function iteration_accepted_solves

  !> Larger relative change of emitted outward flux and mean normal energy at the last accepted update.
  real(dp) function iteration_relative_change(self) result(change)
    class(sheath_matching_iteration), intent(in) :: self
    change = self%change
  end function iteration_relative_change

  !> Write the state to an unformatted unit owned by the caller. Plasma and solver settings are not written.
  subroutine write_state(self, unit, status, message)
    class(sheath_matching_iteration), intent(in) :: self
    integer, intent(in) :: unit
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    integer :: iostat
    status = SHEATH_INVALID_ARGUMENT
    message = 'Matching iteration must be initialized before write.'
    if (.not. self%initialized) return
    status = SHEATH_NUMERICAL_FAILURE
    message = 'Could not write the matching iteration state.'
    write (unit, iostat=iostat) state_version, self%branch, self%root%branch, self%root%surface_potential_v, &
        self%root%minimum_potential_v, self%root%ambient_electron_density_m3, self%solves, self%change
    if (iostat /= 0) return
    call write_emission_source(self%input%plasma%emission, unit, iostat)
    if (iostat /= 0) return
    status = SHEATH_OK
    message = ''
  end subroutine write_state

  !> Restore a state written by write_state for the supplied plasma and solver. The saved root seeds a solve on its
  !! own branch with the saved source; a different branch or no root is an error, not a fallback.
  subroutine read_state(self, unit, plasma, solver, status, message)
    class(sheath_matching_iteration), intent(out) :: self
    integer, intent(in) :: unit
    type(plasma_input), intent(in) :: plasma
    type(sheath_solver), intent(in) :: solver
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(sheath_equilibrium_result) :: saved(1)
    type(emission_source) :: saved_source
    integer :: version, iostat
    integer(i32) :: solves
    real(dp) :: change

    status = SHEATH_INVALID_ARGUMENT
    message = 'Could not read a matching iteration state of a supported version.'
    read (unit, iostat=iostat) version, self%branch, saved(1)%branch, saved(1)%surface_potential_v, &
        saved(1)%minimum_potential_v, saved(1)%ambient_electron_density_m3, solves, change
    if (iostat /= 0 .or. version /= state_version) return
    saved(1)%valid = .true.
    call read_emission_source(saved_source, unit, iostat)
    if (iostat /= 0) return
    call validate_emission(saved_source, status, message)
    if (status /= SHEATH_OK) return
    status = SHEATH_INVALID_ARGUMENT
    message = 'Saved matching state has an invalid branch.'
    if (index('ABC', saved(1)%branch) == 0 .or. saved(1)%branch == ' ') return
    select case (trim(self%branch))
    case ('auto', 'a', 'b', 'c')
    case default
      return
    end select
    self%input%plasma = plasma
    self%input%plasma%emission = saved_source
    self%solver = solver
    call solve_in_order(self, saved(1)%branch, saved, self%root, self%matching, status, message)
    if (status /= SHEATH_OK) then
      message = 'Saved matching state does not reproduce a root for this plasma: '//trim(message)
      return
    end if
    self%solves = solves
    self%change = change
    self%initialized = .true.
  end subroutine read_state

  !> Try branches in order and return the first root with valid matching conditions.
  subroutine solve_in_order(self, order, guesses, root, matching, status, message)
    type(sheath_matching_iteration), intent(inout) :: self
    character(len=*), intent(in) :: order
    type(sheath_equilibrium_result), intent(in) :: guesses(:)
    type(sheath_equilibrium_result), intent(out) :: root
    type(sheath_matching_conditions), intent(out) :: matching
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    character(len=256) :: attempt
    integer :: i

    message = ''
    status = SHEATH_NUMERICAL_FAILURE
    do i = 1, len_trim(order)
      self%input%branch = order(i:i)
      call self%solver%solve_equilibrium(self%input, root, status, attempt, initial_guesses=guesses)
      if (status == SHEATH_OK) call evaluate_matching_conditions(self%input, root, matching, status, attempt)
      if (status == SHEATH_OK) then
        message = ''
        return
      end if
      if (len_trim(message) > 0) message = trim(message)//' | '
      message = trim(message)//order(i:i)//': '//trim(attempt)
    end do
  end subroutine solve_in_order

  pure real(dp) function relative_difference(new, old) result(change)
    real(dp), intent(in) :: new, old
    change = 0.0_dp
    if (old > 0.0_dp) then
      change = abs(new - old)/old
    else if (new > 0.0_dp) then
      change = 1.0_dp
    end if
  end function relative_difference

  pure function upper(text) result(converted)
    character(len=*), intent(in) :: text
    character(len=len(text)) :: converted
    integer :: i
    converted = text
    do i = 1, len(text)
      if (text(i:i) >= 'a' .and. text(i:i) <= 'z') converted(i:i) = achar(iachar(text(i:i)) - 32)
    end do
  end function upper

end module sheath_model_matching_iteration
