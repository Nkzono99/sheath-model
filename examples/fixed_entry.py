"""Evaluate a warm-ion sheath with the entrance speed and emitted source fixed."""

from sheath_model import FixedEntryParams, FixedEntrySheathSolver, SearchOptions


def main():
    params = FixedEntryParams(
        ion_density_m3=8.7e6, ion_entry_speed_mps=405299.88897111727,
        ion_temperature_ev=12., ion_pressure_factor=3.,
        electron_temperature_ev=12., electron_drift_mps=0.,
        photoelectron_density_m3=55.42562584220407e6, photoelectron_temperature_ev=2.2,
    )
    # Select auto, newton or lm here; bracket is available for B/C.
    solver = FixedEntrySheathSolver(params, search=SearchOptions(method="newton"))
    profile = solver.solve_profile("A")
    fluxes = solver.fluxes_at_z(profile, 0., unit="m")
    print(f"Surface potential [V]: {profile['phi0_V']:.8f}")
    print(f"Minimum potential [V]: {profile['phi_m_V']:.8f}")
    print(f"Surface field [V/m]: {profile['E_Vpm'][0]:.6e}")
    print(f"Inward ion flux [m^-2 s^-1]: {-fluxes['Gamma_swi_signed_m2s']:.6e}")


if __name__ == "__main__":
    main()
