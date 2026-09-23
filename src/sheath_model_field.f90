! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone physical input/result; removed application coupling interfaces.
module sheath_model_field
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sheath_model_constants, only: dp, i32, qe, lower_ascii
  use sheath_model_status, only: SHEATH_OK, SHEATH_INVALID_ARGUMENT, SHEATH_NO_PHYSICAL_SOLUTION, SHEATH_NUMERICAL_FAILURE, &
      SHEATH_AMBIGUOUS_SOLUTION
  use sheath_model_core, only: zhao_params_type, neutral_electron_density, evaluate_zhao_fluxes
  use sheath_model_state, only: zhao_plasma_input, prepare_plasma_params

  implicit none

  private

  public :: zhao_field_input, zhao_field_result, solve_prescribed_field, solve_prescribed_field_candidates
  public :: zhao_field_search_diagnostics

  integer, parameter :: default_field_starts = 16

  !> Search diagnostics for branches A, B and C, stored in that array order.
  !! Counters except roots_found describe initial guesses, not distinct roots; success need not resolve every start.
  type :: zhao_field_search_diagnostics
    logical :: searched(3) = .false.
    logical :: excluded(3) = .false.
    integer(i32) :: starts(3) = 0
    integer(i32) :: unconverged(3) = 0
    integer(i32) :: rejected(3) = 0
    integer(i32) :: profile_failures(3) = 0
    integer(i32) :: roots_found(3) = 0
  end type zhao_field_search_diagnostics

  !> Plasma inputs and prescribed normal field E_H [V/m]; branch selects A/B/C/auto.
  !! Other quantities use SI units except temperatures/normal energies [eV]; field is positive outward, drift positive inward.
  type, extends(zhao_plasma_input) :: zhao_field_input
    character(len=9) :: branch = 'auto'
    real(dp) :: electric_field_v_m = 0.0_dp
  end type zhao_field_input

  !> Accepted prescribed-field root with potentials, densities, particle fluxes and current in SI units.
  !! valid marks success; residual_norm and minimum_field_squared_hat are dimensionless.
  !! ambient_electron_density_m3 is the Maxwellian normalization, not total upstream density.
  type :: zhao_field_result
    logical :: valid = .false.
    character(len=1) :: branch = ' '
    real(dp) :: boundary_potential_v = 0.0_dp
    real(dp) :: minimum_potential_v = 0.0_dp
    real(dp) :: ambient_electron_density_m3 = 0.0_dp
    real(dp) :: electron_inward_flux_m2_s = 0.0_dp
    real(dp) :: ion_inward_flux_m2_s = 0.0_dp
    real(dp) :: photoelectron_outward_flux_m2_s = 0.0_dp
    real(dp) :: photoelectron_return_flux_m2_s = 0.0_dp
    real(dp) :: photoelectron_escape_flux_m2_s = 0.0_dp
    real(dp) :: net_current_a_m2 = 0.0_dp
    real(dp) :: residual_norm = huge(1.0_dp)
    real(dp) :: minimum_field_squared_hat = huge(1.0_dp)
    integer(i32) :: nonlinear_iterations = 0_i32
  end type zhao_field_result

  type :: zhao_field_root
    character(len=1) :: branch = ' '
    real(dp) :: phi0_v = 0.0_dp
    real(dp) :: phi_m_v = 0.0_dp
    real(dp) :: ambient_electron_density_m3 = 0.0_dp
    real(dp) :: residual_norm = huge(1.0_dp)
    real(dp) :: minimum_field_squared_hat = huge(1.0_dp)
    integer(i32) :: nonlinear_iterations = 0_i32
  end type zhao_field_root

  interface
    module subroutine solve_field_root( &
        model, params, interface_field_v_m, &
        root, status, message, &
        diagnostics, initial_guesses &
        )
      character(len=*), intent(in) :: model
      type(zhao_params_type), intent(in) :: params
      real(dp), intent(in) :: interface_field_v_m
      type(zhao_field_root), intent(out) :: root
      integer(i32), intent(out) :: status
      character(len=*), intent(out) :: message
      type(zhao_field_search_diagnostics), intent(out) :: diagnostics
      type(zhao_field_result), intent(in), optional :: initial_guesses(:)
    end subroutine solve_field_root

    module subroutine find_field_roots( &
        model, params, interface_field_v_m, &
        roots, status, message, &
        diagnostics, initial_guesses &
        )
      character(len=*), intent(in) :: model
      type(zhao_params_type), intent(in) :: params
      real(dp), intent(in) :: interface_field_v_m
      type(zhao_field_root), allocatable, intent(out) :: roots(:)
      integer(i32), intent(out) :: status
      character(len=*), intent(out) :: message
      type(zhao_field_search_diagnostics), intent(out) :: diagnostics
      type(zhao_field_result), intent(in), optional :: initial_guesses(:)
    end subroutine find_field_roots

    module subroutine make_field_branch_guesses( &
        params, branch, target_field_hat, &
        guesses, count &
        )
      type(zhao_params_type), intent(in) :: params
      character(len=1), intent(in) :: branch
      real(dp), intent(in) :: target_field_hat
      real(dp), intent(out) :: guesses(3, default_field_starts)
      integer, intent(out) :: count
    end subroutine make_field_branch_guesses

    module subroutine newton_field_branch( &
        params, branch, target_field_hat, &
        y0, y_out, &
        final_norm, iterations, &
        success &
        )
      type(zhao_params_type), intent(in) :: params
      character(len=1), intent(in) :: branch
      real(dp), intent(in) :: target_field_hat, y0(3)
      real(dp), intent(out) :: y_out(3), final_norm
      integer, intent(out) :: iterations
      logical, intent(out) :: success
    end subroutine newton_field_branch

    module subroutine encode_field_unknowns( &
        params, branch, &
        phi0_v, phi_m_v, density_m3, &
        y, valid &
        )
      type(zhao_params_type), intent(in) :: params
      character(len=1), intent(in) :: branch
      real(dp), intent(in) :: phi0_v, phi_m_v, density_m3
      real(dp), intent(out) :: y(3)
      logical, intent(out) :: valid
    end subroutine encode_field_unknowns

    module subroutine decode_field_unknowns( &
        params, branch, y, &
        phi0_v, phi_m_v, density_m3, &
        valid &
        )
      type(zhao_params_type), intent(in) :: params
      character(len=1), intent(in) :: branch
      real(dp), intent(in) :: y(3)
      real(dp), intent(out) :: phi0_v, phi_m_v, density_m3
      logical, intent(out) :: valid
    end subroutine decode_field_unknowns

    module subroutine evaluate_charge_residual( &
        params, branch, target_field_hat, &
        y, residual, valid &
        )
      type(zhao_params_type), intent(in) :: params
      character(len=1), intent(in) :: branch
      real(dp), intent(in) :: target_field_hat, y(3)
      real(dp), intent(out) :: residual(3)
      logical, intent(out) :: valid
    end subroutine evaluate_charge_residual

    module subroutine validate_field_root_profile( &
        params, root, target_field_hat, &
        status, message &
        )
      type(zhao_params_type), intent(in) :: params
      type(zhao_field_root), intent(inout) :: root
      real(dp), intent(in) :: target_field_hat
      integer(i32), intent(out) :: status
      character(len=*), intent(out) :: message
    end subroutine validate_field_root_profile
  end interface

contains

  !> Solve neutrality/Sagdeev conditions at input's prescribed E_H and return the sole admissible candidate found.
  !! output includes the resulting current; multiple candidates return SHEATH_AMBIGUOUS_SOLUTION.
  !! Optional diagnostics reports search outcomes; initial_guesses supplements the default starts with nearby solutions.
  !! status/message describe success or failure; a successful finite search does not prove global uniqueness.
  subroutine solve_prescribed_field( &
      input, output, &
      status, message, &
      diagnostics, initial_guesses &
      )
    type(zhao_field_input), intent(in) :: input
    type(zhao_field_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(zhao_field_search_diagnostics), intent(out), optional :: diagnostics
    type(zhao_field_result), intent(in), optional :: initial_guesses(:)

    type(zhao_params_type) :: params
    type(zhao_field_root) :: root
    type(zhao_field_search_diagnostics) :: search
    character(len=256) :: search_message

    output = zhao_field_result()
    if (present(diagnostics)) then
      diagnostics = zhao_field_search_diagnostics()
    end if
    call prepare_field_params(input, params, status, message)
    if (status /= SHEATH_OK) return

    call solve_field_root( &
        trim(lower_ascii(input%branch)), params, input%electric_field_v_m, &
        root, status, message, &
        search, initial_guesses &
        )
    if (present(diagnostics)) then
      diagnostics = search
    end if
    if (status /= SHEATH_OK) return

    search_message = message
    call compose_result(params, root, output, status, message)
    if (status == SHEATH_OK) then
      message = search_message
    end if
  end subroutine solve_prescribed_field

  !> Find admissible roots at input's prescribed E_H and allocate outputs with the candidates found.
  !! status/message describe the outcome; on failure outputs is unallocated. Candidate order has no ranking meaning.
  !! Optional diagnostics reports unresolved starts even on success; the search need not find every root.
  !! initial_guesses supplements default starts and must be a different variable from outputs.
  subroutine solve_prescribed_field_candidates( &
      input, outputs, &
      status, message, &
      diagnostics, initial_guesses &
      )
    type(zhao_field_input), intent(in) :: input
    type(zhao_field_result), allocatable, intent(out) :: outputs(:)
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(zhao_field_search_diagnostics), intent(out), optional :: diagnostics
    type(zhao_field_result), intent(in), optional :: initial_guesses(:)

    type(zhao_params_type) :: params
    type(zhao_field_root), allocatable :: roots(:)
    type(zhao_field_search_diagnostics) :: search
    character(len=256) :: search_message
    integer :: i

    if (present(diagnostics)) then
      diagnostics = zhao_field_search_diagnostics()
    end if
    call prepare_field_params(input, params, status, message)
    if (status /= SHEATH_OK) return

    call find_field_roots( &
        trim(lower_ascii(input%branch)), params, input%electric_field_v_m, &
        roots, status, message, &
        search, initial_guesses &
        )
    if (present(diagnostics)) then
      diagnostics = search
    end if
    if (status /= SHEATH_OK) return

    search_message = message
    allocate (outputs(size(roots)))
    do i = 1, size(roots)
      call compose_result(params, roots(i), outputs(i), status, message)
      if (status /= SHEATH_OK) then
        deallocate (outputs)
        return
      end if
    end do
    message = search_message
  end subroutine solve_prescribed_field_candidates

  subroutine compose_result(params, root, output, status, message)
    type(zhao_params_type), intent(in) :: params
    type(zhao_field_root), intent(in) :: root
    type(zhao_field_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    type(zhao_field_result) :: trial

    output = zhao_field_result()
    status = SHEATH_OK
    message = ''

    trial%branch = root%branch
    trial%boundary_potential_v = root%phi0_v
    trial%minimum_potential_v = min(0.0_dp, root%phi0_v)
    if (root%branch == 'A') then
      trial%minimum_potential_v = root%phi_m_v
    end if
    trial%ambient_electron_density_m3 = root%ambient_electron_density_m3

    call evaluate_zhao_fluxes(params, root%phi0_v, trial%minimum_potential_v, root%ambient_electron_density_m3, &
        trial%electron_inward_flux_m2_s, trial%ion_inward_flux_m2_s, trial%photoelectron_outward_flux_m2_s, &
        trial%photoelectron_escape_flux_m2_s, trial%photoelectron_return_flux_m2_s)
    trial%net_current_a_m2 = qe*(trial%electron_inward_flux_m2_s - trial%ion_inward_flux_m2_s - &
        trial%photoelectron_escape_flux_m2_s)
    if (.not. all(ieee_is_finite([trial%electron_inward_flux_m2_s, trial%ion_inward_flux_m2_s, &
        trial%photoelectron_outward_flux_m2_s, trial%photoelectron_return_flux_m2_s, &
        trial%photoelectron_escape_flux_m2_s, trial%net_current_a_m2]))) then
      status = SHEATH_NUMERICAL_FAILURE
      message = 'Prescribed-field flux or current evaluation is non-finite.'
      return
    end if

    trial%residual_norm = root%residual_norm
    trial%minimum_field_squared_hat = root%minimum_field_squared_hat
    trial%nonlinear_iterations = root%nonlinear_iterations
    trial%valid = .true.
    output = trial
  end subroutine compose_result

  subroutine prepare_field_params(input, params, status, message)
    type(zhao_field_input), intent(in) :: input
    type(zhao_params_type), intent(out) :: params
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    params = zhao_params_type()
    status = SHEATH_INVALID_ARGUMENT
    message = 'branch must be auto, A, B, or C.'

    select case (trim(lower_ascii(input%branch)))
    case ('auto', 'a', 'b', 'c')
    case default
      return
    end select

    message = 'The prescribed field must be finite.'
    if (.not. ieee_is_finite(input%electric_field_v_m)) return
    call prepare_plasma_params(input, params, status, message)
  end subroutine prepare_field_params

end module sheath_model_field
