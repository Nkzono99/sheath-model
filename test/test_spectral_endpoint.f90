! SPDX-License-Identifier: Apache-2.0
! Static regression adapted from BEACH's batch-154 precision case; see NOTICE.
program test_spectral_endpoint
  use sheath_model
  use sheath_model_constants, only: qe, eps0
  implicit none
  type(zhao_plasma_input) :: input
  type(zhao_state_result) :: state, left, right
  real(dp) :: edges(145), flux(144), lo, phi, density
  integer :: unit, ios, i
  integer(i32) :: status
  character(len=256) :: message

  open (newunit=unit, file='test/fixtures/beach_batch154_spectrum.csv', status='old', iostat=ios)
  call check(ios == 0, 'fixture open')
  read (unit, '(a)') message
  do i = 1, 144
    read (unit, *, iostat=ios) lo, edges(i + 1), flux(i)
    call check(ios == 0, 'fixture row')
    if (i == 1) edges(1) = lo
    call check(lo == edges(i), 'contiguous spectrum')
  end do
  close (unit)
  input%ion_density_m3 = 5e6_dp
  input%electron_temperature_ev = 10.0_dp*1.160451812e4_dp*1.380649e-23_dp/qe
  input%electron_drift_mps = 4e5_dp
  input%ion_drift_mps = 4e5_dp
  input%electron_mass_kg = 9.1093837139e-31_dp
  input%photoelectrons = binned_photoelectrons(edges, flux)
  phi = 6.43321741907773337_dp
  density = 4167907.56298632319959109_dp
  call evaluate_sheath_state(input, 'B', nearest(phi, -1.0_dp), left, status, message)
  call check(status == SHEATH_OK, 'left adjacent potential')
  call evaluate_sheath_state(input, 'B', nearest(phi, 1.0_dp), right, status, message)
  call check(status == SHEATH_OK, 'right adjacent potential')
  call check(density >= min(left%ambient_electron_density_m3, right%ambient_electron_density_m3) .and. &
      density <= max(left%ambient_electron_density_m3, right%ambient_electron_density_m3), 'oracle density inside rounding cell')
  call evaluate_sheath_state(input, 'B', phi, state, status, message, electron_normalization_m3=density)
  call check(status == SHEATH_OK .and. state%admissible, 'physical spectral endpoint')
  ! Independent 60-digit velocity/Poisson oracle recorded in BEACH's regression.
  call check(abs(eps0*state%electric_field_v_m - 1.9063843016813198782561e-11_dp) < 5e-24_dp, &
      'field agrees with high-precision oracle')
  call check(abs(state%neutrality_residual_m3)/input%ion_density_m3 < 2e-12_dp, 'neutrality within one potential ulp')
  call check(abs(state%photoelectron_outward_flux_m2_s - state%photoelectron_escape_flux_m2_s &
      - state%photoelectron_return_flux_m2_s)/sum(flux) < 2e-15_dp, 'source conservation')
  print *, 'BEACH spectral endpoint agrees with independent high-precision reference.'
contains
  subroutine check(condition, label)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: label
    if (.not. condition) then
      print *, 'FAIL: ', label, ' ', trim(message), ' ', trim(state%physical_message)
      print *, 'D, rho: ', eps0*state%electric_field_v_m, state%neutrality_residual_m3
      error stop 1
    end if
  end subroutine
end program
