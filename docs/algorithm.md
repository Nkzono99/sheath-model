# Algorithm notes

The current model transports the upstream electron distribution along collisionless
orbits. It retains Zhao's cold-ion, photoelectron and A/B/C population topology,
but replaces the locally shifted Maxwellian expressions used in v0.1.0.
The derivation, closure and limits are detailed in [kinetic-model.md](kinetic-model.md).
Fortran inputs and units are listed in [fortran-api.md](fortran-api.md).

## Upstream distribution and local moments

Let `psi = phi/T_e`, `u = v_d/v_th`, and `v_th = sqrt(2 e T_e/m_e)`.
The incoming upstream distribution is proportional to `exp(-(a-u)^2)` for
`a = |v_infinity|/v_th >= 0`. Energy conservation gives

```text
w^2 = a^2 + psi
f_local(w) = N_e/(sqrt(pi) v_th) exp(-(sqrt(w^2-psi)-u)^2)
```

Only accessible velocities are populated. Passing electrons have
`w >= sqrt((phi-phi_min)/T_e)`; reflected electrons occupy both directions below
that threshold on A's upper side and on C. B uses `phi_min=0`, so its incoming
velocity cutoff is nonzero for positive potentials. At zero drift,
`n_e/N_e = exp(psi) erfc(sqrt(psi))/2` on B.

Both implementations integrate the same local distribution for density. At zero
drift they use the exact expressions, evaluating `exp(psi)*erfc(...)` with the
scaled complementary error function to avoid overflow and endpoint quadrature error.
At nonzero drift they use an endpoint transformation `w=lower+(upper-lower)t^2` and composite Gauss quadrature.
Velocity tails are truncated ten thermal widths beyond the drift or cutoff.
The passing flux is evaluated by the equivalent upstream integral, using
`w dw = a da`, and is constant along each orbit population. Python VDF diagnostics
sample this distribution directly; flux diagnostics use its analytic moments.

## Root equations

J=0 imposes upstream neutrality and zero net current. A additionally imposes a
zero field at infinity when integrating from its internal minimum. Prescribed-field
solves replace the zero-current equation with the prescribed boundary field.
The upstream electron normalization N_e is a solved unknown in both cases.
The static potential API instead obtains N_e directly from upstream neutrality
at the supplied potentials. It returns signed boundary E squared and A's upper
connection residual without imposing either the current or boundary-field closure.

## Photoelectron energy spectra

The Fortran field and static APIs accept either an analytic Maxwellian source or
arbitrary normal-energy bins. A bin stores its **integrated outward number flux**;
`g=dGamma/dK=bin_flux/bin_width` is constant within it. Let `v*=sqrt(2e/m_e)`,
`q=phi_H-phi`, and `B=phi_H-phi_min`. For each bin `[a,b]`, the passing density is

```text
n_free = 2g/v* [sqrt(b-q) - sqrt(max(a,B,q)-q)]
```

when `b>max(a,B,q)`, and zero otherwise. On the lower side, the returning density
uses `[max(a,q), min(b,B)]` and twice this prefactor for the two orbit directions.
Fluxes integrate g over the escaping or returning energy interval, counting return
only once at the boundary. A's upper side and C have no captured photoelectrons.

PE density integrals in potential are analytic differences of 3/2 powers, split
at turning points. Square-root differences are rationalized; mixed 3/2-power
differences factor both bin width and potential interval. This avoids subtracting
nearly equal primitives for narrow bins or adjacent floating-point endpoints.
Only the ion/background-electron contribution uses potential quadrature for a bin
source. No Maxwellian moment fit enters either density or current.

Spectral Type A root searches and acceptance divide the upper connection residual by
`(-phi_min_hat)^(3/2)`, so a shallow minimum alone cannot make that residual appear
converged. Numerical potential scales are T_PE for a Maxwellian source and T_e for
a bin source, rounded down to a power of two. Binary scaling preserves spectral
bin edges through normalization round trips; it is not an inferred source temperature.

Type A permits `phi_min < min(phi_H,0)`; it is not restricted to positive surface
potentials. Unknown transformations for the field solve use a positive gap
`phi_H-phi_min` and a negative minimum. E_H=0 is searched normally; a flat solution
is added as one candidate when its density is positive. The A/C endpoint is
coalesced into C when the minimum reaches the boundary.

## Acceptance and profiles

Roots must have accessible cold ions and a real connecting first integral of
Poisson's equation. Fortran shares the same acceptance routine between J=0 and
prescribed-field solves and static evaluation. Python also checks physical admissibility before returning
algebraic unknowns. Auto selection skips inadmissible roots.

An additional asymptotic obstruction applies to A/C with `u>0`: reflected slow
electrons create a positive density correction proportional to
`u |psi| log(1/|psi|)` near neutral infinity. The resulting E squared becomes
negative arbitrarily near that endpoint. Such roots are rejected even if a finite
sampling grid misses the negative interval. The nondrifting model is selected
explicitly with `electron_drift_mode='zero'` in J=0 or `electron_drift_mps=0` in the
field API; there is no silent drift substitution.

For Type B, `rho/e=C sqrt(phi)+...` at upstream infinity. C>0 produces negative
E squared arbitrarily near that endpoint. Both implementations reject the Maxwellian
case with

```text
C = N_e exp(-u^2)/sqrt(pi T_e) - N_PE exp(-phi_H/T_PE)/sqrt(pi T_PE).
```

For bins, the second term is `2(2g_left-g_right)/v*`. At a bin edge, returning and
escaping populations sample different one-sided values of g. A 128-epsilon
relative margin on the two coefficients allows for roundoff without replacing
the physical check by a finite profile grid. C=0 still needs the remaining checks.

Both implementations reconstruct all profiles from the semi-infinite first integral.
The old Python B/C finite-interval BVP has been removed: its trial iterates can leave
the accessible potential interval after introducing the kinetic cutoffs. The
Python parameter `n_profile_grid` controls the potential grid for B/C, and
`profile_phi_tol_hat` controls the upstream cutoff on all branches. Returned positions
are nonuniform and may stop before zmax_hat; no artificial zero-potential tail is added.

## Multiple solutions and numerical limits

The prescribed-field solve returns a solution only when the finite multistart
search finds one admissible root. Multiple roots produce an ambiguity status;
candidate enumeration lets callers inspect their potentials, densities, and fluxes.
No stability ranking or automatic change of the input drift is performed.

The former Type A integral with a removable `1/u` singularity has been replaced by
quadrature of the same charge density as Poisson's equation. It is regular at u=0.
All root existence statements are limited by finite numerical search and integration
tolerances. Numerical failure is not proof that no physical solution exists.

For a very steep spectral endpoint, binary64 potentials can straddle an external
closure root without any representable midpoint satisfying its residual tolerance.
The static API supports reevaluating a B/C state at a supplied N_e restricted to
the neutral densities of the immediate neighboring potentials. All moments and
integrals then use that same N_e, and existing neutrality/profile tolerances remain
in force. The external equation, its bracket refinement, residual normalization,
and time integration belong to the caller. See [spectral-api.md](spectral-api.md).

## Validation

Tests compare velocity moments against independent adaptive or Simpson integration,
check the zero-drift B expression and A-to-B limit, conserve passing flux at multiple
positions, reject inadmissible algebraic roots, verify negative-surface A profiles,
and follow the field response through a nonflat E_H=0 transition. Profile checks also
compare E with minus the potential gradient and its derivative with charge density.
Spectral tests add independent local-velocity and potential quadrature, escape/return
conservation, Maxwellian refinement, equal-moment distributions with different shapes,
bin-edge one-sided limits, and narrow-bin/adjacent-potential precision. The BEACH
batch-154 spectrum is checked against its independent high-precision static oracle.
