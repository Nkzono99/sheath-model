! SPDX-License-Identifier: Apache-2.0
! Adapted from BEACH (Jin Nakazono); see NOTICE and LICENSES/Apache-2.0.txt.
! Modified: standalone physical input/result; removed application coupling interfaces.
module sheath_model_field
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use sheath_model_constants, only: dp, i32, eps0, pi, qe, electron_mass, proton_mass, lower_ascii, &
                                    sheath_ok, sheath_invalid_argument, sheath_no_physical_solution, sheath_numerical_failure, &
                                    sheath_ambiguous_solution
  use sheath_model_core, only: zhao_params_type, swe_free_current_term
  implicit none
  private
  public :: zhao_field_input, zhao_field_result, solve_prescribed_field

  type :: zhao_field_input
    character(len=9) :: branch = 'auto'
    character(len=16) :: root_selection = 'require_unique'
    real(dp) :: electric_field_v_m = 0.0_dp
    real(dp) :: ion_density_m3 = 8.7e6_dp
    real(dp) :: photoelectron_source_density_m3 = 0.0_dp
    real(dp) :: electron_temperature_ev = 12.0_dp
    real(dp) :: photoelectron_temperature_ev = 2.2_dp
    real(dp) :: electron_drift_mps = 4.0529988897111727e5_dp
    real(dp) :: ion_drift_mps = 4.0529988897111727e5_dp
    real(dp) :: ion_mass_kg = proton_mass
    real(dp) :: electron_mass_kg = electron_mass
  end type zhao_field_input

  type :: zhao_field_result
    logical :: valid = .false.
    character(len=1) :: branch = ' '
    real(dp) :: boundary_potential_v = 0.0_dp
    real(dp) :: minimum_potential_v = 0.0_dp
    real(dp) :: ambient_electron_density_m3 = 0.0_dp
    real(dp) :: electron_inward_flux_m2_s = 0.0_dp
    real(dp) :: ion_inward_flux_m2_s = 0.0_dp
    real(dp) :: photoelectron_escape_flux_m2_s = 0.0_dp
    real(dp) :: net_current_a_m2 = 0.0_dp
    real(dp) :: residual_norm = huge(1.0_dp)
    real(dp) :: minimum_field_squared_hat = huge(1.0_dp)
    real(dp) :: potential_energy_j_m2 = huge(1.0_dp)
    integer(i32) :: nonlinear_iterations = 0_i32
  end type zhao_field_result

  type :: zhao_field_root
    character(len=1) :: branch = ' '
    real(dp) :: phi0_v = 0.0_dp
    real(dp) :: phi_m_v = 0.0_dp
    real(dp) :: ambient_electron_density_m3 = 0.0_dp
    real(dp) :: residual_norm = huge(1.0_dp)
    real(dp) :: minimum_field_squared_hat = huge(1.0_dp)
    real(dp) :: potential_energy_j_m2 = huge(1.0_dp)
    integer(i32) :: nonlinear_iterations = 0_i32
  end type zhao_field_root

  interface
    module subroutine solve_field_root(model, root_selection, params, interface_field_v_m, root, status, message)
      character(len=*), intent(in) :: model, root_selection
      type(zhao_params_type), intent(in) :: params
      real(dp), intent(in) :: interface_field_v_m
      type(zhao_field_root), intent(out) :: root
      integer(i32), intent(out) :: status
      character(len=*), intent(out) :: message
    end subroutine solve_field_root

    module subroutine make_field_branch_guesses(params, branch, guesses, count)
      type(zhao_params_type), intent(in) :: params
      character(len=1), intent(in) :: branch
      real(dp), intent(out) :: guesses(3, 8)
      integer, intent(out) :: count
    end subroutine make_field_branch_guesses

    module subroutine newton_field_branch( &
      params, branch, target_field_hat, y0, y_out, final_norm, iterations, success &
      )
      type(zhao_params_type), intent(in) :: params
      character(len=1), intent(in) :: branch
      real(dp), intent(in) :: target_field_hat, y0(3)
      real(dp), intent(out) :: y_out(3), final_norm
      integer, intent(out) :: iterations
      logical, intent(out) :: success
    end subroutine newton_field_branch

    module subroutine encode_field_unknowns(params, branch, phi0_v, phi_m_v, density_m3, y, valid)
      type(zhao_params_type), intent(in) :: params
      character(len=1), intent(in) :: branch
      real(dp), intent(in) :: phi0_v, phi_m_v, density_m3
      real(dp), intent(out) :: y(3)
      logical, intent(out) :: valid
    end subroutine encode_field_unknowns

    module subroutine decode_field_unknowns(params, branch, y, phi0_v, phi_m_v, density_m3, valid)
      type(zhao_params_type), intent(in) :: params
      character(len=1), intent(in) :: branch
      real(dp), intent(in) :: y(3)
      real(dp), intent(out) :: phi0_v, phi_m_v, density_m3
      logical, intent(out) :: valid
    end subroutine decode_field_unknowns

    module subroutine evaluate_charge_residual(params, branch, target_field_hat, y, residual, valid)
      type(zhao_params_type), intent(in) :: params
      character(len=1), intent(in) :: branch
      real(dp), intent(in) :: target_field_hat, y(3)
      real(dp), intent(out) :: residual(3)
      logical, intent(out) :: valid
    end subroutine evaluate_charge_residual

    module subroutine validate_field_root_profile(params, root, target_field_hat, status, message)
      type(zhao_params_type), intent(in) :: params
      type(zhao_field_root), intent(inout) :: root
      real(dp), intent(in) :: target_field_hat
      integer(i32), intent(out) :: status
      character(len=*), intent(out) :: message
    end subroutine validate_field_root_profile

    module subroutine evaluate_root_potential_energy(params, root, status, message)
      type(zhao_params_type), intent(in) :: params
      type(zhao_field_root), intent(inout) :: root
      integer(i32), intent(out) :: status
      character(len=*), intent(out) :: message
    end subroutine evaluate_root_potential_energy

  end interface
contains

  !> Prescribe E_H and solve neutrality/Sagdeev conditions; current is an output.
  subroutine solve_prescribed_field(input, output, status, message)
    type(zhao_field_input), intent(in) :: input
    type(zhao_field_result), intent(out) :: output
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message
    type(zhao_params_type) :: params
    type(zhao_field_root) :: root
    type(zhao_field_result) :: trial
    real(dp) :: cutoff, flux_scale

    output = zhao_field_result()
    call prepare_field_params(input, params, status, message)
    if (status /= sheath_ok) return
    call solve_field_root(trim(lower_ascii(input%branch)), trim(lower_ascii(input%root_selection)), &
                          params, input%electric_field_v_m, root, status, message)
    if (status /= sheath_ok) return

    trial%branch = root%branch
    trial%boundary_potential_v = root%phi0_v
    trial%minimum_potential_v = min(0.0_dp, root%phi0_v)
    if (root%branch == 'A') trial%minimum_potential_v = root%phi_m_v
    trial%ambient_electron_density_m3 = root%ambient_electron_density_m3
    cutoff = sqrt(max(0.0_dp, -trial%minimum_potential_v/params%t_swe_ev)) - params%u
    flux_scale = params%v_phe_th_mps/(2.0_dp*sqrt(pi))
    trial%electron_inward_flux_m2_s = flux_scale*swe_free_current_term(params, root%ambient_electron_density_m3, cutoff)
    trial%ion_inward_flux_m2_s = params%n_swi_inf_m3*params%v_d_ion_mps
    trial%photoelectron_escape_flux_m2_s = params%n_phe0_m3*flux_scale* &
                                           exp((trial%minimum_potential_v - root%phi0_v)/params%t_phe_ev)
    trial%net_current_a_m2 = qe*(trial%electron_inward_flux_m2_s - trial%ion_inward_flux_m2_s - &
                                 trial%photoelectron_escape_flux_m2_s)
    if (.not. all(ieee_is_finite([trial%electron_inward_flux_m2_s, trial%ion_inward_flux_m2_s, &
                                  trial%photoelectron_escape_flux_m2_s, trial%net_current_a_m2]))) then
      status = sheath_numerical_failure
      message = 'Prescribed-field flux or current evaluation is non-finite.'
      return
    end if
    trial%residual_norm = root%residual_norm
    trial%minimum_field_squared_hat = root%minimum_field_squared_hat
    trial%potential_energy_j_m2 = root%potential_energy_j_m2
    trial%nonlinear_iterations = root%nonlinear_iterations
    trial%valid = .true.
    output = trial
  end subroutine solve_prescribed_field

  subroutine prepare_field_params(input, params, status, message)
    type(zhao_field_input), intent(in) :: input
    type(zhao_params_type), intent(out) :: params
    integer(i32), intent(out) :: status
    character(len=*), intent(out) :: message

    params = zhao_params_type()
    status = sheath_invalid_argument
    message = 'branch must be auto, A, B, or C.'
    select case (trim(lower_ascii(input%branch)))
    case ('auto', 'a', 'b', 'c')
    case default
      return
    end select
    message = 'root_selection must be require_unique or minimum_energy.'
    select case (trim(lower_ascii(input%root_selection)))
    case ('require_unique', 'minimum_energy')
    case default
      return
    end select
    message = 'Prescribed-field inputs must be finite.'
    if (.not. all(ieee_is_finite([input%electric_field_v_m, input%ion_density_m3, &
                         input%photoelectron_source_density_m3, input%electron_temperature_ev, input%photoelectron_temperature_ev, &
                                 input%electron_drift_mps, input%ion_drift_mps, input%ion_mass_kg, input%electron_mass_kg]))) return
    message = 'Ion density, temperatures, ion speed, and masses must be positive; photoelectron density nonnegative.'
    if (min(input%ion_density_m3, input%electron_temperature_ev, input%photoelectron_temperature_ev, &
            input%ion_drift_mps, input%ion_mass_kg, input%electron_mass_kg) <= 0.0_dp) return
    if (input%photoelectron_source_density_m3 < 0.0_dp) return
    params%n_swi_inf_m3 = input%ion_density_m3
    params%n_phe_ref_m3 = input%ion_density_m3
    params%n_phe0_m3 = input%photoelectron_source_density_m3
    params%t_swe_ev = input%electron_temperature_ev
    params%t_phe_ev = input%photoelectron_temperature_ev
    params%v_d_electron_mps = input%electron_drift_mps
    params%v_d_ion_mps = input%ion_drift_mps
    params%m_i_kg = input%ion_mass_kg
    params%m_e_kg = input%electron_mass_kg
    params%v_swe_th_mps = sqrt(2.0_dp*qe*params%t_swe_ev/params%m_e_kg)
    params%v_phe_th_mps = sqrt(2.0_dp*qe*params%t_phe_ev/params%m_e_kg)
    params%cs_mps = sqrt(qe*params%t_swe_ev/params%m_i_kg)
    params%mach = params%v_d_ion_mps/params%cs_mps
    params%u = params%v_d_electron_mps/params%v_swe_th_mps
    params%tau = params%t_swe_ev/params%t_phe_ev
    params%lambda_d_phe_ref_m = sqrt(eps0*params%t_phe_ev/(params%n_phe_ref_m3*qe))
    status = sheath_numerical_failure
    message = 'Prescribed-field normalization produced non-finite or underflowed parameters.'
    if (.not. all(ieee_is_finite([params%v_swe_th_mps, params%v_phe_th_mps, params%cs_mps, &
                                  params%mach, params%u, params%tau, params%lambda_d_phe_ref_m]))) return
    if (min(params%v_swe_th_mps, params%v_phe_th_mps, params%cs_mps, params%mach, &
            params%tau, params%lambda_d_phe_ref_m) <= 0.0_dp) return
    status = sheath_ok
    message = ''
  end subroutine prepare_field_params
end module sheath_model_field
