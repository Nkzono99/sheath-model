#!/usr/bin/env python3
"""Independent bounded checks of the upstream collisionless orbit model.

Run on a compute node: python verify_upstream.py --out validation
No production rejection guard is used to demonstrate the asymptotic obstruction.
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

from sheath_model._orbits import electron_density as production_density

SQRT_PI = math.sqrt(math.pi)
QE = 1.602176634e-19
ME = 9.1093837015e-31
MI = 1.67262192369e-27
EPS0 = 8.8541878128e-12


def integrate(fun, a, b, tol=2e-11):
    return quad(fun, a, b, epsabs=tol, epsrel=tol, limit=120)[0]


def g(a, drift):
    return math.exp(-(a - drift)**2) / SQRT_PI


def density_upstream(psi, barrier, drift):
    """Density from an independent adaptive integral in upstream velocity a."""
    cutoff = math.sqrt(-barrier)
    minimum = math.sqrt(max(0.0, -psi))
    jacobian_integrand = lambda a: a * g(a, drift) / math.sqrt(a*a + psi)
    free = integrate(jacobian_integrand, cutoff, math.inf)
    reflected = 0.0
    if cutoff > minimum:
        reflected = 2.0 * integrate(jacobian_integrand, minimum, cutoff)
    return free, reflected


def density_local(psi, barrier, drift):
    """Adaptive local-velocity integration, independent of production quadrature."""
    cutoff = math.sqrt(max(0.0, psi - barrier))
    minimum = math.sqrt(max(0.0, psi))
    local_g = lambda w: g(math.sqrt(max(0.0, w*w - psi)), drift)
    free = integrate(local_g, cutoff, math.inf)
    reflected = 2.0 * integrate(local_g, minimum, cutoff) if cutoff > minimum else 0.0
    return free, reflected


def density_difference_over_h(h, barrier, drift):
    """Stable [n(-h)-n(0)]/h without subtracting nearly equal densities.

    n(-h)=2 int_0^c g(sqrt(w*w+h)) dw + int_c^inf g(sqrt(w*w+h)) dw,
    c=sqrt(-barrier-h). The moving-cutoff correction is evaluated on [0,1].
    """
    b = math.sqrt(-barrier)
    c = math.sqrt(-barrier - h)

    def difference(w):
        shift = h / (math.sqrt(w*w + h) + w)
        return g(w, drift) * math.expm1(-h + 2.0*drift*shift) / h

    split = min(math.sqrt(h), c)
    inner = integrate(difference, 0.0, split)
    if split < c:
        inner += integrate(difference, split, c)
    correction = integrate(lambda t: g(c + (b-c)*t, drift), 0.0, 1.0) / (b+c)
    return 2.0*inner + integrate(difference, c, math.inf) - correction


def save_csv(path, records):
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

    orbit_rows = []
    states = [(0.25, -0.08), (-0.03, -0.08), (0.25, 0.0), (-0.3, -0.5)]
    drift_values = [0.0, 0.05, 0.2, 0.213272]
    for drift in drift_values:
        for psi, barrier in states:
            direct_free, direct_reflected = density_upstream(psi, barrier, drift)
            source_free, source_reflected = map(float, production_density(psi, barrier, drift))
            cutoff = math.sqrt(psi - barrier)
            local_flux = integrate(
                lambda w: w*g(math.sqrt(max(0.0, w*w-psi)), drift), cutoff, math.inf)
            upstream_flux = integrate(lambda a: a*g(a, drift), math.sqrt(-barrier), math.inf)
            orbit_rows.append(dict(
                drift=drift, psi=psi, barrier=barrier,
                density_free_upstream=direct_free, density_free_source=source_free,
                density_reflected_upstream=direct_reflected, density_reflected_source=source_reflected,
                density_free_abs_error=abs(direct_free-source_free),
                density_reflected_abs_error=abs(direct_reflected-source_reflected),
                local_flux=local_flux, upstream_flux=upstream_flux,
                flux_abs_error=abs(local_flux-upstream_flux)))
    save_csv(args.out / "upstream_orbit_quadrature.csv", orbit_rows)

    asymptotic_rows = []
    asymptotic_summary = []
    barrier = -0.1
    h_values = [10.0**(-power) for power in range(3, 13)]
    for drift in drift_values:
        coefficient = drift*math.exp(-drift*drift)/SQRT_PI
        previous = None
        local_rows = []
        for h in h_values:
            quotient = density_difference_over_h(h, barrier, drift)
            slope = None if previous is None else (quotient-previous[1])/math.log(previous[0]/h)
            row = dict(drift=drift, barrier=barrier, h=h, log_inverse_h=math.log(1.0/h),
                       density_difference_over_h=quotient, predicted_log_coefficient=coefficient,
                       observed_log_slope="" if slope is None else slope)
            asymptotic_rows.append(row)
            local_rows.append(row)
            previous = (h, quotient)
        observed = local_rows[-1]["observed_log_slope"]
        asymptotic_summary.append(dict(
            drift=drift, predicted_log_coefficient=coefficient, observed_log_coefficient=observed,
            absolute_error=abs(observed-coefficient),
            explanation="A upper and C share this reflected upstream population; normalization N_e=1."))
    save_csv(args.out / "upstream_hlog_asymptotic.csv", asymptotic_rows)

    # Independently refine a former A algebraic root, without consulting any source solver.
    nref = 64.0e6
    ni = 8.7e6/nref
    npe = math.sin(math.pi/3.0)
    te, tpe = 12.0, 2.2
    vi = 468.0e3*math.sin(math.pi/3.0)
    ve = math.sqrt(2.0*QE*te/ME)
    vp = math.sqrt(2.0*QE*tpe/ME)
    drift = vi/ve

    def rho(phi_v, phi0_v, phim_v, density):
        free, reflected = density_local(phi_v/te, phim_v/te, drift)
        ions = ni/math.sqrt(1.0 - 2.0*QE*phi_v/(MI*vi*vi))
        photo = 0.5*npe*math.exp((phi_v-phi0_v)/tpe)*math.erfc(math.sqrt(max(0.0, (phi_v-phim_v)/tpe)))
        return ions-density*(free+reflected)-photo

    def equations(values):
        phi0, phim, density = values
        if phim >= min(phi0, 0.0) or density <= 0.0:
            return np.array([1e3, 1e3, 1e3])
        neutral = rho(0.0, phi0, phim, density)
        upper_integral = integrate(
            lambda t: rho(phim + (-phim)*t*t, phi0, phim, density)*2.0*(-phim)*t/tpe,
            0.0, 1.0, tol=2e-10)
        b = math.sqrt(-phim/te)
        electron_flux = density*ve/(2.0*SQRT_PI)*(
            math.exp(-(b-drift)**2) + SQRT_PI*drift*math.erfc(b-drift))
        ion_flux = ni*vi
        photo_flux = npe*vp/(2.0*SQRT_PI)*math.exp((phim-phi0)/tpe)
        return np.array([neutral, upper_integral, (electron_flux-ion_flux-photo_flux)/vp])

    solution = root(equations, [2.6544268139403324, -1.1897238115913789, 7923268.55824609/nref],
                    method="hybr", options={"xtol":1e-9, "maxfev":45})
    phi0, phim, density = map(float, solution.x)
    residuals = equations(solution.x)
    field_rows = []
    field_scale_squared = QE*nref*tpe/EPS0
    for lower_phi_hat in [-0.1, -0.03, -0.01, -0.003, -0.001, -0.0003]:
        squared = 2.0*integrate(
            lambda phi_hat: rho(phi_hat*tpe, phi0, phim, density), lower_phi_hat, 0.0, tol=1e-12)
        field_rows.append(dict(
            lower_phi_hat=lower_phi_hat, lower_phi_v=lower_phi_hat*tpe,
            independently_integrated_field_squared_hat=squared,
            independently_integrated_field_squared_v2_m2=squared*field_scale_squared))
    save_csv(args.out / "upstream_algebraic_root_field_integral.csv", field_rows)

    checks = dict(
        source_density_matches_independent_upstream_quadrature=max(
            max(r["density_free_abs_error"], r["density_reflected_abs_error"]) for r in orbit_rows) < 3e-9,
        flux_conserved_between_local_and_upstream_velocity=max(r["flux_abs_error"] for r in orbit_rows) < 3e-10,
        hlog_coefficients_match_independent_asymptotics=max(r["absolute_error"] for r in asymptotic_summary) < 2e-5,
        independent_A_algebraic_root_refined=float(np.max(np.abs(residuals))) < 2e-8,
        independently_negative_upstream_field_integral=min(
            r["independently_integrated_field_squared_hat"] for r in field_rows) < -1e-7)
    summary = dict(
        checks=checks, passed=all(checks.values()),
        elapsed_seconds=time.monotonic()-started,
        max_density_abs_error=max(max(r["density_free_abs_error"], r["density_reflected_abs_error"]) for r in orbit_rows),
        max_flux_abs_error=max(r["flux_abs_error"] for r in orbit_rows),
        asymptotic=asymptotic_summary,
        independent_algebraic_root=dict(
            phi0_v=phi0, phim_v=phim, electron_normalization_m3=density*nref,
            electron_drift_ratio=drift, root_solver_reported_success=bool(solution.success),
            root_solver_message=str(solution.message), residuals=residuals.tolist(),
            most_negative_field_squared_hat=min(r["independently_integrated_field_squared_hat"] for r in field_rows)),
        interpretation=[
            "The algebraic A root satisfies neutrality, J=0, and the upper Sagdeev endpoint condition.",
            "Its negative E^2 near upstream infinity independently disproves a real half-infinite profile for that root.",
            "The positive-u h log(1/h) coefficient was measured without calling the production admissibility guard.",
            "No internal-domain electrons or their space charge were added.",
            "A finite upstream reservoir, collisions, and alternative low-speed VDFs were not solved or validated here.",
            "The finite parameter sample is a numerical check of the stated asymptotic argument, not an existence theorem."])
    (args.out / "upstream_validation.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
