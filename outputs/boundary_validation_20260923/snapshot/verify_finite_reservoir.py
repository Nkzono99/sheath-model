#!/usr/bin/env python3
"""Construct and verify a finite-reservoir sheath; compute-node execution only.

This is an independent validation prototype, not a production boundary model.
The outer length is derived from a constructed profile, then held fixed for a
perturbed-seed inverse solve. It is not fitted to BEACH or sent to production.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root

QE, ME, MI, EPS0 = 1.602176634e-19, 9.1093837015e-31, 1.67262192369e-27, 8.8541878128e-12
SQRT_PI = math.sqrt(math.pi)
NREF, TPE, TE = 64e6, 2.2, 12.0
NI = 8.7e6/NREF
NPE = math.sin(math.pi/3.0)
VI = 468e3*math.sin(math.pi/3.0)
U = VI/math.sqrt(2*QE*TE/ME)
TAU = TE/TPE
LAMBDA = math.sqrt(EPS0*TPE/(QE*NREF))
FIELD_SCALE = TPE/LAMBDA


def quadrature(fun, lo, hi, tol=2e-10):
    return quad(fun, lo, hi, epsabs=tol, epsrel=tol, limit=100)[0]


def rho_hat(phi, phi0, phim, electron_density, side):
    """Dimensionless charge from reservoir-connected trajectories only."""
    psi, barrier = phi/TAU, phim/TAU
    cutoff = math.sqrt(max(0.0, psi-barrier))
    def local_g(w):
        a = math.sqrt(max(0.0, w*w-psi))
        return math.exp(-(a-U)**2)/SQRT_PI
    free = quadrature(local_g, cutoff, math.inf)
    reflected = 0.0
    if side == "upper" and cutoff > 0.0:
        reflected = 2.0*quadrature(local_g, 0.0, cutoff)
    ions = NI/math.sqrt(1.0-2.0*QE*TPE*phi/(MI*VI*VI))
    s = math.sqrt(max(0.0, phi-phim))
    factor = 1.0+math.erf(s) if side == "lower" else math.erfc(s)
    photo = 0.5*NPE*math.exp(phi-phi0)*factor
    return ions-electron_density*(free+reflected)-photo


def cumulative_simpson_even(values, dx):
    """Quadratic integration with even panel count and values at every endpoint."""
    result = np.zeros_like(values)
    for i in range(0, len(values)-2, 2):
        result[i+1] = result[i]+dx*(5*values[i]+8*values[i+1]-values[i+2])/12.0
        result[i+2] = result[i]+dx*(values[i]+4*values[i+1]+values[i+2])/3.0
    return result


def simpson(values, dx):
    return dx*(values[0]+values[-1]+4*np.sum(values[1:-1:2])+2*np.sum(values[2:-1:2]))/3.0


def profile(phi0, phim, density, panels, keep=False):
    if not phim < min(phi0, 0.0):
        return {"valid": False, "reason": "invalid topology"}
    t = np.linspace(0.0, 1.0, panels+1)
    output = {"valid": True}
    for side, endpoint in [("lower", phi0), ("upper", 0.0)]:
        width = endpoint-phim
        potentials = phim+width*t*t
        charges = np.array([rho_hat(float(phi), phi0, phim, density, side) for phi in potentials])
        e2 = -2.0*cumulative_simpson_even(charges*2.0*width*t, 1.0/panels)
        if charges[0] >= 0.0 or np.any(e2[1:] <= 0.0):
            return {"valid": False, "reason": "nonpositive real field", "side": side,
                    "minimum_e2": float(np.min(e2[1:])), "rho_at_minimum": float(charges[0])}
        distance_integrand = np.empty_like(t)
        distance_integrand[0] = math.sqrt(2.0*width/(-charges[0]))
        distance_integrand[1:] = 2.0*width*t[1:]/np.sqrt(e2[1:])
        output[side+"_length_hat"] = float(simpson(distance_integrand, 1.0/panels))
        output[side+"_endpoint_e2_hat"] = float(e2[-1])
        output[side+"_minimum_e2_over_t2"] = float(np.min(e2[1:]/(t[1:]**2)))
        output[side+"_endpoint_charge_hat"] = float(charges[-1])
        if keep:
            output[side+"_arrays"] = dict(t=t, potential_hat=potentials, charge_hat=charges, field_squared_hat=e2,
                distance_from_minimum_m=LAMBDA*cumulative_simpson_even(distance_integrand, 1.0/panels))
    output["length_m"] = LAMBDA*(output["lower_length_hat"]+output["upper_length_hat"])
    output["electric_field_h_v_m"] = FIELD_SCALE*math.sqrt(output["lower_endpoint_e2_hat"])
    output["electric_field_l_v_m"] = -FIELD_SCALE*math.sqrt(output["upper_endpoint_e2_hat"])
    return output


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("validation"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    phi0_v, phim_v, n_e_reference = 2.6544268139403324, -1.1897238115913789, 7923268.55824609
    upstream_path = args.out/"upstream_validation.json"
    if upstream_path.exists():
        upstream = json.loads(upstream_path.read_text())["independent_algebraic_root"]
        phi0_v, phim_v, n_e_reference = upstream["phi0_v"], upstream["phim_v"], upstream["electron_normalization_m3"]
    phi0, phim = phi0_v/TPE, phim_v/TPE
    trials = []
    chosen = None
    for increase in [0.01, 0.05, 0.1]:
        density = n_e_reference*(1.0+increase)/NREF
        candidate = profile(phi0, phim, density, 128)
        trials.append(dict(increase_fraction=increase, electron_normalization_m3=density*NREF, **candidate))
        if candidate["valid"]:
            chosen = (increase, density)
            break
    if chosen is None:
        report = dict(passed=False, trials=trials, explanation="No constructed finite profile found in the bounded three-density sample.")
        (args.out/"finite_reservoir_validation.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        return 1
    increase, density = chosen
    grids = []
    selected = None
    for panels in [128, 256, 512, 1024]:
        selected = profile(phi0, phim, density, panels, keep=panels==1024)
        grids.append(dict(panels=panels, **{k:v for k,v in selected.items() if not k.endswith("_arrays")}))
    write_csv(args.out/"finite_reservoir_convergence.csv", grids)
    profile_rows = []
    for side in ["lower", "upper"]:
        data = selected[side+"_arrays"]
        for i in range(len(data["t"])):
            direction = -1.0 if side == "lower" else 1.0
            z = selected["lower_length_hat"]*LAMBDA+direction*data["distance_from_minimum_m"][i]
            field = -direction*FIELD_SCALE*math.sqrt(max(0.0, data["field_squared_hat"][i]))
            profile_rows.append(dict(side=side, z_m=float(z), potential_v=float(data["potential_hat"][i])*TPE,
                electric_field_v_m=field, **{key:float(val[i]) for key,val in data.items()}))
    write_csv(args.out/"finite_reservoir_profile.csv", profile_rows)

    # Adaptive endpoint integrals independently check the tabulated field integrals.
    adaptive_e2 = {}
    for side, endpoint in [("lower", phi0), ("upper", 0.0)]:
        width = endpoint-phim
        adaptive_e2[side] = -2.0*quadrature(
            lambda t: rho_hat(phim+width*t*t, phi0, phim, density, side)*2.0*width*t,
            0.0, 1.0, tol=1e-11)
    near_upstream = []
    for p in [0.0, -1e-8, -1e-6, -1e-4, -1e-3, -1e-2]:
        e2 = adaptive_e2["upper"]+2.0*quadrature(
            lambda phi: rho_hat(phi, phi0, phim, density, "upper"), p, 0.0, tol=1e-12)
        near_upstream.append(dict(potential_hat=p, potential_v=p*TPE, field_squared_hat=e2))
    write_csv(args.out/"finite_reservoir_near_upstream.csv", near_upstream)

    # Hold derived E_H, L, N_e, phi(L)=0 fixed and recover both unknown potentials.
    target_e2 = selected["lower_endpoint_e2_hat"]
    target_length = selected["length_m"]
    cache = {}
    def residual(y):
        key = tuple(map(float, y))
        if key not in cache:
            p0, pm = float(y[0]), -math.exp(float(y[1]))
            value = profile(p0, pm, density, 256)
            if value["valid"]:
                cache[key] = np.array([value["lower_endpoint_e2_hat"]/target_e2-1.0,
                                       value["length_m"]/target_length-1.0])
            else:
                cache[key] = np.array([1e2, 1e2])
        return cache[key]
    # Lowering phi_H at fixed minimum strengthens the PE density, keeping the
    # initial finite profile on the real-field side of the nearby obstruction.
    guess = np.array([phi0*0.99, math.log(-phim)])
    inverse = root(residual, guess, method="hybr", options={"xtol":1e-8, "maxfev":28})
    recovered_phi0, recovered_phim = float(inverse.x[0]), -math.exp(float(inverse.x[1]))
    recovered = profile(recovered_phi0, recovered_phim, density, 1024)
    recovered_residual = [recovered["lower_endpoint_e2_hat"]/target_e2-1.0,
                          recovered["length_m"]/target_length-1.0] if recovered["valid"] else [1e2, 1e2]
    fine, coarse = grids[-1], grids[-2]
    convergence = {key:abs(fine[key]-coarse[key])/max(abs(fine[key]), 1e-30)
                   for key in ["length_m", "electric_field_h_v_m", "electric_field_l_v_m"]}
    endpoint_errors = {side:abs(selected[side+"_endpoint_e2_hat"]-adaptive_e2[side])
                           /max(abs(adaptive_e2[side]), 1e-30) for side in ["lower", "upper"]}
    checks = dict(
        positive_drift=U>0.0,
        nonmonotonic_finite_profile=all(g["valid"] for g in grids),
        field_positive_in_near_upstream_samples=min(r["field_squared_hat"] for r in near_upstream)>0.0,
        independently_checked_field_integrals=max(endpoint_errors.values())<2e-6,
        grid_doubling_converged=max(convergence.values())<2e-5,
        inverse_fixed_boundary_problem_recovered=max(map(abs, recovered_residual))<2e-5,
        recovered_boundary_potentials=max(abs(recovered_phi0-phi0),abs(recovered_phim-phim))*TPE<1e-3)
    report = dict(
        passed=all(checks.values()), checks=checks, elapsed_seconds=time.monotonic()-started,
        assumptions=dict(domain="finite H <= z <= L", potential_l_v=0.0,
            electron_incident_vdf="Ne exp(-(a-u)^2)/(sqrt(pi) vth), a>=0 at L",
            outgoing_boundary="absorbed by external reservoir; no prescribed E_L or neutrality",
            ions="cold inward beam prescribed at L", photoelectrons="half Maxwell source at H",
            internal_electrons="excluded", electron_drift_ratio=U),
        construction=dict(increase_fraction=increase, fixed_electron_normalization_m3=density*NREF,
            phi_h_v=phi0*TPE, minimum_potential_v=phim*TPE,
            derived_fixed_electric_field_h_v_m=selected["electric_field_h_v_m"],
            derived_fixed_length_m=selected["length_m"],
            output_electric_field_l_v_m=selected["electric_field_l_v_m"],
            rho_l_c_m3=QE*NREF*selected["upper_endpoint_charge_hat"],
            ion_minus_electron_density_l_m3=NREF*selected["upper_endpoint_charge_hat"]),
        trial_profiles=trials, convergence_relative_change_512_to_1024=convergence,
        adaptive_endpoint_relative_errors=endpoint_errors,
        inverse_solve=dict(seed_phi_h_v=float(guess[0])*TPE, seed_phim_v=-math.exp(float(guess[1]))*TPE,
            recovered_phi_h_v=recovered_phi0*TPE, recovered_phim_v=recovered_phim*TPE,
            normalized_EH_squared_and_L_residual_at_1024=recovered_residual,
            solver_reported_success=bool(inverse.success), solver_message=str(inverse.message),
            unique_profile_evaluations=len(cache)),
        limitations=[
            "This is a manufactured finite-domain example; its L and incoming Ne were chosen for a bounded existence check.",
            "No measured BEACH matching-plane state or physical location of the outer reservoir was fitted.",
            "Positive E^2 was checked numerically with grid refinement and independent endpoint/near-boundary quadrature, not proved for all parameters.",
            "No neutrality or zero-field limit at infinity, infinite-domain convergence, or dynamical stability is claimed.",
            "This demonstrates a different boundary problem can admit a positive-drift A-shaped profile; it does not select that model for production."])
    (args.out/"finite_reservoir_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
