! SPDX-License-Identifier: MIT
!> Couple a simulator's top plane to the outer J=0 sheath: plane conditions, an outflow source,
!! an outer update, and a checkpoint round trip. The outflow histogram is synthetic here; a simulator
!! accumulates it from the electrons that cross its matching plane outward during a window.
program matching_plane
  use sheath_model
  implicit none
  integer, parameter :: samples = 20000
  real(dp), parameter :: area_m2 = 1.0_dp, duration_s = 1.0e-6_dp
  type(sheath_solver) :: solver
  type(plasma_input) :: plasma
  type(sheath_matching_iteration) :: outer, restored
  type(sheath_matching_conditions) :: plane, restart
  type(emission_source) :: source
  real(dp), allocatable :: edges(:), weight(:)
  real(dp) :: energy_ev, particles, observed_flux
  integer(i32) :: status
  integer :: j, unit
  character(len=512) :: message

  ! Upstream plasma above the plane; plasma%emission is the first outer source.
  plasma%electron_drift_mps = 0.0_dp
  plasma%emission = maxwellian_emission(55.42562584220407e6_dp, 2.2_dp)
  call outer%initialize(plasma, solver, status, message)
  call require_ok('initialize')
  plane = outer%conditions()
  call report('initial')

  ! One window: bin the normal kinetic energy K [eV] of each outward crossing, weighted by physical particles.
  ! Here the crossings sample a 2.0 eV half-Maxwellian carrying 90% of the initial outward flux.
  edges = log_energy_edges()
  allocate (weight(size(edges) - 1), source=0.0_dp)
  observed_flux = 0.9_dp*plane%emitted_outward_flux_m2_s
  particles = observed_flux*area_m2*duration_s/real(samples, dp)
  do j = 1, samples
    energy_ev = -2.0_dp*log((real(j, dp) - 0.5_dp)/real(samples, dp))
    weight(energy_bin(edges, energy_ev)) = weight(energy_bin(edges, energy_ev)) + particles
  end do
  ! A parallel simulator sums weight over its ranks here, then divides by area and window length.
  call observed_outflow_source(edges, weight/(area_m2*duration_s), source, status, message)
  call require_ok('observed_outflow_source')
  print '(a,f8.4)', 'Observed mean normal energy [eV]: ', source%mean_normal_energy()

  call outer%update(source, status, message)
  call require_ok('update')
  plane = outer%conditions()
  call report('updated')
  print '(a,es12.4,a,i0)', 'Relative change: ', outer%relative_change(), ', accepted solves: ', outer%accepted_solves()

  ! Checkpoint: the caller owns the unit; plasma and solver are configuration and are passed again on restart.
  open (newunit=unit, status='scratch', form='unformatted', access='stream')
  call outer%write_state(unit, status, message)
  call require_ok('write_state')
  rewind (unit)
  call restored%read_state(unit, plasma, solver, status, message)
  call require_ok('read_state')
  close (unit)
  restart = restored%conditions()
  print '(a,es12.4)', 'Restart potential difference [V]: ', &
      abs(restart%plane_potential_v - plane%plane_potential_v)

contains

  subroutine require_ok(step)
    character(len=*), intent(in) :: step
    if (status /= SHEATH_OK) then
      print '(a,": ",a)', step, trim(message)
      error stop 1
    end if
  end subroutine require_ok

  subroutine report(label)
    character(len=*), intent(in) :: label
    print '(a," Type ",a,": phi_H=",f9.5," V, E(H+)=",es11.4," V/m, path minimum=",f9.5," V")', &
        label, plane%branch, plane%plane_potential_v, plane%plane_field_v_m, plane%electron_path_minimum_v
    print '(a,3es12.4,a,es10.2)', '  inward e/i, emitted escape [m^-2 s^-1]:', plane%electron_inward_flux_m2_s, &
        plane%ion_inward_flux_m2_s, plane%emitted_escape_flux_m2_s, '; J [A/m^2]:', plane%net_current_a_m2
  end subroutine report
end program matching_plane
