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

Both implementations integrate the same local distribution for density. They use
an endpoint transformation `w=lower+(upper-lower)t^2` and composite Gauss quadrature.
Velocity tails are truncated ten thermal widths beyond the drift or cutoff.
The passing flux is evaluated by the equivalent upstream integral, using
`w dw = a da`, and is constant along each orbit population. Python VDF diagnostics
sample this distribution directly; flux diagnostics use its analytic moments.

## Root equations

J=0 imposes upstream neutrality and zero net current. A additionally imposes a
zero field at infinity when integrating from its internal minimum. Prescribed-field
solves replace the zero-current equation with the prescribed boundary field.
The upstream electron normalization N_e is a solved unknown in both cases.

Type A permits `phi_min < min(phi_H,0)`; it is not restricted to positive surface
potentials. Unknown transformations for the field solve use a positive gap
`phi_H-phi_min` and a negative minimum. E_H=0 is searched normally; a flat solution
is added as one candidate when its density is positive. The A/C endpoint is
coalesced into C when the minimum reaches the boundary.

## Acceptance and profiles

Roots must have accessible cold ions and a real connecting first integral of
Poisson's equation. Fortran shares the same acceptance routine between J=0 and
prescribed-field solves. Python also checks physical admissibility before returning
algebraic unknowns. Auto selection skips inadmissible roots.

An additional asymptotic obstruction applies to A/C with `u>0`: reflected slow
electrons create a positive density correction proportional to
`u |psi| log(1/|psi|)` near neutral infinity. The resulting E squared becomes
negative arbitrarily near that endpoint. Such roots are rejected even if a finite
sampling grid misses the negative interval. The nondrifting model is selected
explicitly with `electron_drift_mode='zero'` in J=0 or `electron_drift_mps=0` in the
field API; there is no silent drift substitution.

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

## Validation

Tests compare velocity moments against independent adaptive or Simpson integration,
check the zero-drift B expression and A-to-B limit, conserve passing flux at multiple
positions, reject inadmissible algebraic roots, verify negative-surface A profiles,
and follow the field response through a nonflat E_H=0 transition. Profile checks also
compare E with minus the potential gradient and its derivative with charge density.
