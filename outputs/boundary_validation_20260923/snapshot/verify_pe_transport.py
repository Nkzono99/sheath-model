#!/usr/bin/env python3
"""Independent stationary PE transport checks; run only on a compute node.

No sheath-model implementation is imported. Numerical quadrature is over
dimensionless normal velocity, using Liouville transport and kinetic-energy
conservation. Archive comparisons hold the recorded H spectrum and barrier
fixed; they do not rerun a sheath solve or establish statistical significance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.integrate import quad


TEMPERATURE_EV = 2.2
SOURCE_FLUX = 2.808679e13
DEFAULT_ARCHIVE = Path(
    "/LARGE0/gr20001/b36291/Github/BEACH/outputs/"
    "sheath_pe_spectrum_implementation_20260922/raw"
)


def integrate(function, lower, upper=math.inf):
    if upper <= lower:
        return 0.0
    value, error = quad(function, lower, upper, epsabs=2e-13, epsrel=2e-12, limit=200)
    assert math.isfinite(value) and error < max(2e-11, abs(value) * 2e-10)
    return value


def source_velocity_flux(a):
    # Surface half-Maxwell, weighted by normal speed and normalized to unit flux.
    return 2.0 * a * math.exp(-a * a)


def check_close(actual, expected, scale=1.0, tolerance=2e-10):
    error = abs(actual - expected) / max(abs(expected), abs(scale), 1e-300)
    assert error < tolerance, (actual, expected, error)
    return error


def transported_state(surface_phi, h_phi, inner_minimum):
    """Return integrals of f_H(v) from the source-connected branch.

    K_s - phi_s = K_H - phi_H; the inner minimum is no higher than
    either endpoint. The flux distribution has K_H >= phi_H - phi_min,inner.
    """
    assert inner_minimum <= min(surface_phi, h_phi)
    delta = surface_phi - h_phi
    source_threshold = surface_phi - inner_minimum
    cutoff = h_phi - inner_minimum
    w_min = math.sqrt(cutoff / TEMPERATURE_EV)
    a_min = math.sqrt(source_threshold / TEMPERATURE_EV)

    def local_velocity_flux(w):
        a_squared = w * w + delta / TEMPERATURE_EV
        assert a_squared >= -1e-12
        return 2.0 * w * math.exp(-a_squared)

    total = integrate(local_velocity_flux, w_min)
    source_total = integrate(source_velocity_flux, a_min)
    mean = integrate(
        lambda w: TEMPERATURE_EV * w * w * local_velocity_flux(w), w_min
    ) / total
    flux_error = check_close(total, source_total)
    mean_error = check_close(mean, TEMPERATURE_EV + cutoff)
    return dict(
        delta=delta, cutoff=cutoff, w_min=w_min, source_threshold=source_threshold,
        flux=total, mean=mean, distribution=local_velocity_flux,
        integral_error=max(flux_error, mean_error),
    )


def escape_and_return(state, barrier):
    cutoff_w = math.sqrt(max(state["cutoff"], barrier) / TEMPERATURE_EV)
    escape = integrate(state["distribution"], cutoff_w)
    returned = integrate(state["distribution"], state["w_min"], cutoff_w)
    source_threshold = max(state["source_threshold"], state["delta"] + barrier)
    direct_source_escape = integrate(
        source_velocity_flux, math.sqrt(max(0.0, source_threshold) / TEMPERATURE_EV)
    )
    err = max(
        check_close(escape + returned, state["flux"]),
        check_close(escape, direct_source_escape),
    )
    return escape, returned, err


def verify_controlled():
    rows = []
    max_error = check_close(integrate(source_velocity_flux, 0.0), 1.0)
    # Four gauge-shifted surface potentials; each sees the same total barrier.
    for surface_phi in (-2.0, 0.0, 5.0, 8.0):
        outer_minimum = surface_phi - 8.0
        for drop in (0.0, 0.1, 1.0, 3.0, 7.0):
            h_phi = surface_phi - drop
            state = transported_state(surface_phi, h_phi, h_phi)
            barrier = h_phi - outer_minimum
            escape, returned, error = escape_and_return(state, barrier)
            predicted_escape = state["flux"] * math.exp(-barrier / TEMPERATURE_EV)
            max_error = max(max_error, state["integral_error"], error,
                            check_close(escape, predicted_escape),
                            check_close(escape, math.exp(-8.0 / TEMPERATURE_EV)))
            # Check several cumulative probabilities, not just the two moments.
            for energy in (0.1, 0.5, 1.0, 2.0, 5.0):
                energy *= TEMPERATURE_EV
                cumulative = integrate(state["distribution"], 0.0,
                                       math.sqrt(energy / TEMPERATURE_EV))
                max_error = max(max_error, check_close(
                    cumulative / state["flux"], 1.0 - math.exp(-energy / TEMPERATURE_EV)
                ))
            rows.append(make_row("monotonic_retardation", surface_phi, h_phi,
                                 h_phi, barrier, state, escape, returned))

    sign_checks = []
    for name, surface_phi, h_phi, inner_minimum in (
        ("acceleration_small", 0.0, 0.5, 0.0),
        ("acceleration_large", 0.0, 4.0, 0.0),
        ("inner_minimum", 4.0, 2.0, 0.0),
        ("inner_minimum_negative_potentials", -1.0, -2.0, -4.0),
    ):
        state = transported_state(surface_phi, h_phi, inner_minimum)
        cutoff = state["cutoff"]
        barriers = (0.0, 0.25 * cutoff, 0.9 * cutoff, cutoff,
                    cutoff + 0.5 * TEMPERATURE_EV, cutoff + TEMPERATURE_EV,
                    cutoff + 2.0 * TEMPERATURE_EV, cutoff + 5.0 * TEMPERATURE_EV)
        errors = []
        for barrier in barriers:
            escape, returned, error = escape_and_return(state, barrier)
            row = make_row(name, surface_phi, h_phi, inner_minimum, barrier,
                           state, escape, returned)
            rows.append(row)
            max_error = max(max_error, state["integral_error"], error)
            fit_escape = row["moment_fit_escape_m2_s"] / SOURCE_FLUX
            if 0.0 < barrier <= cutoff:
                check_close(returned, 0.0)
                assert fit_escape < escape
            at_crossover = abs(barrier - state["mean"]) <= 1e-10
            if at_crossover:
                max_error = max(max_error, check_close(fit_escape, escape))
            elif 0.0 < barrier < state["mean"]:
                assert fit_escape < escape
            elif barrier > state["mean"]:
                assert fit_escape > escape
            errors.append(row["moment_fit_minus_exact_escape_over_H_flux"])
        assert min(errors) < -0.05 and max(errors) > 0.001
        sign_checks.append(dict(
            case=name, cutoff_ev=cutoff, crossover_barrier_ev=state["mean"],
            minimum_escape_error_over_H_flux=min(errors),
            maximum_escape_error_over_H_flux=max(errors),
        ))
    return rows, dict(
        status="PASS", quadrature_max_scaled_conservation_or_mapping_error=max_error,
        monotonic_retardation_cases=20, shifted_truncated_cases=32,
        source_temperature_ev=TEMPERATURE_EV, source_flux_m2_s=SOURCE_FLUX,
        sign_checks=sign_checks,
        conclusions=[
            "Static planar monotonic retardation preserves a same-temperature half-Maxwellian.",
            "Escaping flux is invariant under the location of H on the same monotonic potential segment.",
            "Acceleration or an earlier inner potential minimum can create a positive H energy cutoff.",
            "Moment matching underestimates escape below the mean-energy barrier and overestimates it above, for these shifted exponentials.",
        ],
    )


def make_row(case, surface_phi, h_phi, inner_minimum, barrier, state, escape, returned):
    fit_escape = state["flux"] * math.exp(-barrier / state["mean"])
    return dict(
        case=case, source_phi_v=surface_phi, H_phi_v=h_phi,
        inner_minimum_phi_v=inner_minimum, outer_barrier_ev=barrier,
        H_energy_cutoff_ev=state["cutoff"], H_mean_energy_ev=state["mean"],
        H_flux_m2_s=SOURCE_FLUX * state["flux"],
        exact_escape_m2_s=SOURCE_FLUX * escape,
        exact_return_m2_s=SOURCE_FLUX * returned,
        moment_fit_escape_m2_s=SOURCE_FLUX * fit_escape,
        moment_fit_return_m2_s=SOURCE_FLUX * (state["flux"] - fit_escape),
        moment_fit_minus_exact_escape_over_H_flux=(fit_escape - escape) / state["flux"],
        moment_fit_relative_escape_error=fit_escape / escape - 1.0,
    )


def verify_archive(archive):
    rows = []
    provenance = []
    for name in ("moments_128", "spectrum_128", "spectrum_32"):
        spectrum_path = archive / name / "matching_plane_spectrum_history.csv"
        history_path = archive / name / "matching_plane_history.csv"
        if not spectrum_path.exists() or not history_path.exists():
            raise FileNotFoundError(f"Missing expected archive input for {name}: {archive}")
        for path in (spectrum_path, history_path):
            provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        with history_path.open(newline="") as stream:
            history = {int(r["batch"]): r for r in csv.DictReader(stream)}
        groups = defaultdict(list)
        with spectrum_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                groups[int(row["batch"])].append(row)
        for batch, bins in sorted(groups.items()):
            first = bins[0]
            barrier = max(0.0, float(first["phi_H_V"]) - float(first["phi_min_V"]))
            input_flux = float(first["input_PE_flux_m2_s"])
            input_mean = float(first["input_mean_energy_eV"])
            modeled = float(first["modeled_escape_flux_m2_s"])
            spectral = first["spectral_closure"].strip().upper() == "T"
            exact_binned = total = first_moment = lower_bound = upper_bound = 0.0
            for b in bins:
                low = float(b["energy_low_eV"])
                high = float(b["energy_high_eV"])
                flux = float(b["input_flux_m2_s"])
                assert high > low >= 0.0 and flux >= 0.0
                fraction = max(0.0, min(1.0, (high - barrier) / (high - low)))
                exact_binned += flux * fraction
                total += flux
                first_moment += flux * (low + high) / 2.0
                if low >= barrier:
                    lower_bound += flux
                if high > barrier:
                    upper_bound += flux
            assert total > 0.0 and input_mean > 0.0
            # The stored input bins have the exact accepted response input flux.
            check_close(total, input_flux, scale=input_flux, tolerance=2e-9)
            binned_mean = first_moment / total
            moment_fit = input_flux * math.exp(-barrier / input_mean)
            histogram_moment_fit = total * math.exp(-barrier / binned_mean)
            expected_model = exact_binned if spectral else moment_fit
            model_error = check_close(modeled, expected_model, scale=input_flux, tolerance=2e-9)
            assert lower_bound <= exact_binned <= upper_bound
            # Uniform-in-bin is exact for the archived representation, not for unknown sub-bin particles.
            actual_observed_escape = float(history[batch]["photoelectron_escape_flux_m2_s"])
            rows.append(dict(
                archive_case=name, batch=batch, spectral_closure=spectral,
                H_phi_v=float(first["phi_H_V"]), phi_min_v=float(first["phi_min_V"]),
                fixed_barrier_ev=barrier, input_flux_m2_s=input_flux,
                input_mean_energy_ev=input_mean, histogram_midpoint_mean_ev=binned_mean,
                exact_piecewise_constant_bin_escape_m2_s=exact_binned,
                moment_fit_at_same_saved_input_escape_m2_s=moment_fit,
                fit_to_histogram_moments_escape_m2_s=histogram_moment_fit,
                moment_fit_relative_to_bin_escape=moment_fit / exact_binned - 1.0,
                histogram_fit_relative_to_bin_escape=histogram_moment_fit / exact_binned - 1.0,
                bin_escape_lower_bound_m2_s=lower_bound,
                bin_escape_upper_bound_m2_s=upper_bound,
                saved_model_escape_m2_s=modeled,
                saved_model_reproduction_error_over_H_flux=model_error,
                trajectory_observed_escape_m2_s=actual_observed_escape,
            ))
    assert len(rows) == 9
    return rows, dict(
        status="PASS", archive_root=str(archive), samples=len(rows), inputs=provenance,
        maximum_absolute_same_input_moment_vs_histogram_escape_fraction=max(
            abs(r["moment_fit_relative_to_bin_escape"]) for r in rows
        ),
        limitations=[
            "Exact refers to the archived piecewise-constant energy-bin representation, not an exact kinetic solution.",
            "Each comparison uses one saved distribution and the same recorded barrier; it does not compare separate evolved runs as if their states were identical.",
            "Observed trajectory escape additionally uses local crossing potentials and individual energies, so it is not identified with an exact mean-plane-bin escape.",
            "The small pilot has Monte Carlo sampling and bin uncertainty. Differences do not establish systematic transport distortion or statistical significance.",
            "These calculations do not reproduce the historical no-root event or solve any modified outer boundary model.",
        ],
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation"))
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--skip-archive", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    controlled_rows, controlled = verify_controlled()
    write_csv(args.output / "pe_transport_controlled.csv", controlled_rows)
    if args.skip_archive:
        archive = dict(status="SKIPPED")
    else:
        archive_rows, archive = verify_archive(args.archive)
        write_csv(args.output / "pe_transport_saved_spectra.csv", archive_rows)
    result = dict(status="PASS", controlled=controlled, archive=archive)
    output_json = args.output / "pe_transport_validation.json"
    output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], report=str(output_json),
                          controlled_max_error=controlled["quadrature_max_scaled_conservation_or_mapping_error"],
                          archive_status=archive["status"]), indent=2))


if __name__ == "__main__":
    main()
