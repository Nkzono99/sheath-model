# Algorithm notes

This document summarizes the numerical formulation implemented in `zhao_sheath/solver.py` for the Zhao et al. semianalytic lunar photoelectron sheath model.

## Scope

The implementation solves the three sheath branches discussed in the Zhao model:

- **Type A**: non-monotonic potential with an internal minimum
- **Type B**: monotonic positive-potential branch
- **Type C**: monotonic negative-potential branch

The code assumes a 1D sheath normal to the surface and includes three particle populations:

- cold solar-wind ions
- drifting Maxwellian solar-wind electrons
- Maxwellian photoelectrons

## Common definitions

The implementation uses the photoelectron temperature as the potential scale.

- `phi_hat = phi / T_phe_eV`
- `z_hat = z / lambda_D_phe_ref`
- `tau = T_swe / T_phe`
- `u = v_d / v_th_swe`
- `M = v_i_inf / c_s`

The normalized Poisson equation is written as

```math
\frac{d^2 \hat\phi}{d\hat z^2} = -\hat\rho(\hat\phi)
```

with charge density

```math
\hat\rho = \hat n_{\mathrm{swi}}
          - \hat n_{\mathrm{swe,f}}
          - \hat n_{\mathrm{swe,r}}
          - \hat n_{\mathrm{phe,f}}
          - \hat n_{\mathrm{phe,c}}.
```

The exact branch-dependent expressions for the density components are implemented in branch-specific helper functions.

## Solver structure

The numerical algorithm has two stages.

1. **Solve branch unknowns**
   - Type A: solve for `(phi0, phi_m, n_swe_inf)`
   - Type B/C: solve for `(phi0, n_swe_inf)`

2. **Reconstruct the profile**
   - Type A: reconstruct `phi(z)` using the first integral of Poisson's equation
   - Type B/C: solve a two-point boundary value problem for `phi(z)` and `E(z)`

The nonlinear unknowns are solved with `scipy.optimize.root(method="hybr")`.

## Type A

## Physical picture

Type A has a non-monotonic potential profile.

- The potential starts at the surface value `phi0`
- It decreases to an internal minimum `phi_m`
- It then recovers toward `0 V` as `z -> infinity`

Because of this turning point, the particle populations differ below and above the minimum.

## Unknowns

The Type A branch solves three unknowns:

- `phi0`
- `phi_m`
- `n_swe_inf`

These are obtained from three nonlinear conditions:

1. far-field charge neutrality
2. zero net current at infinity
3. vanishing electric field at infinity

## Density treatment

The implementation treats the two subdomains separately.

### Lower branch: surface -> `z_m`

Included populations:

- solar-wind ions
- free solar-wind electrons
- free photoelectrons
- captured photoelectrons

Excluded population:

- reflected solar-wind electrons

### Upper branch: `z_m` -> infinity

Included populations:

- solar-wind ions
- free solar-wind electrons
- reflected solar-wind electrons
- free photoelectrons

Excluded population:

- captured photoelectrons

This split is essential. If the same populations are used on both sides of the minimum, the Type A profile becomes unphysical.

## Profile reconstruction

Type A is not reconstructed with a simple finite-domain Dirichlet BVP. Instead, the code uses the first integral of Poisson's equation:

```math
\hat E^2(\hat\phi) = -2 \int_{\hat\phi_m}^{\hat\phi} \hat\rho(\psi)\, d\psi
```

and then maps potential to distance through

```math
\hat z(\hat\phi) = \int \frac{d\hat\phi}{|\hat E(\hat\phi)|}.
```

This is done separately on the lower and upper branches and then concatenated.

## Turning-point handling

A naive discretization around `phi = phi_m` produces an artificial plateau because `E(phi_m) = 0` and `1 / |E|` becomes singular at the first upper-branch grid point.

To avoid this, the implementation uses:

- an **asymptotic launch** near `phi_m`
- **midpoint integration** away from the minimum

Near the minimum,

```math
z - z_m \approx \sqrt{\frac{2(\phi - \phi_m)}{-\rho(\phi_m)}}
```

is used to start the upper branch smoothly.

This was added specifically to remove the artificial flat segment that appears if the first interval is integrated by a direct endpoint-based trapezoidal rule.

## Type B

## Physical picture

Type B is a monotonic positive-potential branch.

- `phi(0) = phi0 > 0`
- `phi(z)` decays monotonically toward `0`

No internal turning point is present.

## Unknowns

Type B solves two unknowns:

- `phi0`
- `n_swe_inf`

These are obtained from:

1. far-field charge neutrality
2. zero net current at infinity

## Profile reconstruction

Once the branch unknowns are known, the code solves the first-order system

```math
\frac{d\hat\phi}{d\hat z} = \hat E,
\qquad
\frac{d\hat E}{d\hat z} = -\hat\rho(\hat\phi)
```

with the boundary conditions

```math
\hat\phi(0) = \hat\phi_0,
\qquad
\hat\phi(\hat z_{\max}) = 0.
```

This is solved using `scipy.integrate.solve_bvp`.

## Type C

## Physical picture

Type C is a monotonic negative-potential branch.

- `phi(0) = phi0 < 0`
- `phi(z)` recovers monotonically toward `0`

Like Type B, it has no internal minimum.

## Unknowns

Type C also solves two unknowns:

- `phi0`
- `n_swe_inf`

from:

1. far-field charge neutrality
2. zero net current at infinity

## Profile reconstruction

The profile is computed by the same BVP formulation used for Type B:

```math
\frac{d\hat\phi}{d\hat z} = \hat E,
\qquad
\frac{d\hat E}{d\hat z} = -\hat\rho(\hat\phi)
```

with

```math
\hat\phi(0) = \hat\phi_0,
\qquad
\hat\phi(\hat z_{\max}) = 0.
```

## Numerical notes

### 1. Type A is treated differently on purpose

Type A should not be handled with the same finite-domain Dirichlet strategy used for Type B/C. A forced condition like `phi(z_max) = 0` on a non-monotonic branch tends to create a spurious flat profile followed by a sharp return to zero near the outer boundary.

### 2. No artificial zero-potential tail

The current implementation does not append a constant `phi = 0` tail to Type A after the physically integrated profile ends. Returning such an artificial tail makes the profile look flat even when the true solution is still recovering.

### 3. `zmax_hat` has different roles

- For **Type B/C**, `zmax_hat` is the outer boundary of the BVP.
- For **Type A**, the reconstructed profile is returned up to the range reached by the first-integral integration, rather than padded to `zmax_hat` with a fake tail.

### 4. Tightening the Type A outer tolerance

The parameter `type_a_phi_tol_hat` controls how close the Type A reconstruction approaches `phi = 0` before stopping. Smaller values produce a longer tail and allow the recovery toward `0 V` to be followed more closely.

## Recommended outputs

The solver returns dictionaries containing branch-dependent fields such as:

- `branch`
- `z_hat`
- `phi_hat`
- `z_m_hat` and `phi_m_hat` for Type A
- dimensional quantities such as `phi0_V`, `phi_m_V`

This is intended to support both plotting and regression testing.

## Suggested future extensions

- add regression tests against digitized figures from the paper
- expose branch equations in a separate `equations.py`
- add a notebook comparing Type A/B/C across solar zenith angle
- document the exact correspondence between code expressions and equation numbers in the paper
