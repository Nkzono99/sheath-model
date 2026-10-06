"""Predictor/corrector continuation in dimensionless solution coordinates."""
from dataclasses import dataclass, replace
import math
import numpy as np

from .search import solve_guarded_system


@dataclass(frozen=True)
class ContinuationOptions:
    """parameter follows the input path; arclength can follow its simple folds.

    Steps use a path parameter running from 0 to 1 (arclength also includes
    solution coordinates). A root jump exceeding max_root_distance is retried
    with a smaller step. These finite controls do not prove branch completeness.
    """
    method: str = "parameter"
    initial_step: float = .25
    min_step: float = 1e-4
    max_step: float = .5
    max_steps: int = 128
    max_root_distance: float = .75

    def __post_init__(self):
        if self.method not in {"parameter", "arclength"}:
            raise ValueError("continuation method must be parameter or arclength")
        values = (self.min_step, self.initial_step, self.max_step, self.max_root_distance)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("continuation step sizes and root distance must be finite and positive")
        if not self.min_step <= self.initial_step <= self.max_step:
            raise ValueError("require min_step <= initial_step <= max_step")
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")


def continue_guarded_system(residual, start, search, options, diagnostics, k, *, accept=None):
    """Follow F(y,t)=0 from t=0 to t=1, returning an exactly corrected end root.

    The callback returns None outside its domain. accept(y,t) checks each
    corrected root; intermediate states are not evidence for a final root.
    """
    local = replace(search, method="newton" if search.method in {"auto", "bracket"} else search.method,
                    use_default_guesses=False)
    y = np.asarray(start, dtype=float).copy()
    t, step = 0., options.initial_step
    previous = None
    tangent = np.r_[np.zeros(len(y)), 1.]

    def valid_root(value, parameter):
        diagnostics.evaluations[k] += 1
        f = residual(value, parameter)
        valid = f is not None and np.all(np.isfinite(f))
        if valid and parameter == 1.:
            diagnostics.best_residual[k] = min(diagnostics.best_residual[k], float(np.max(np.abs(f))))
        return (valid and np.max(np.abs(f)) <= local.residual_tolerance and
                (accept is None or accept(value, parameter)))

    def correct(callback, prediction, at_target=False):
        best = diagnostics.best_residual[k]
        result = solve_guarded_system(callback, prediction, local, diagnostics, k)
        if not at_target:
            diagnostics.best_residual[k] = best
        return result

    if not valid_root(y, t):
        return y, False
    for _ in range(options.max_steps):
        if options.method == "parameter":
            trial_t = min(1., t+step)
            prediction = y.copy()
            if previous is not None and t != previous[1]:
                prediction += (trial_t-t)/(t-previous[1])*(y-previous[0])
            trial, _, success = correct(lambda v: residual(v, trial_t), prediction, at_target=trial_t == 1.)
        else:
            z = np.r_[y, t]
            f0 = residual(y, t)
            jac = np.zeros((len(y), len(z)))
            success = f0 is not None
            for column in range(len(z)):
                h = np.finfo(float).eps**(1/3)*max(1., abs(z[column]))
                zp, zm = z.copy(), z.copy()
                zp[column] += h
                zm[column] -= h
                fp, fm = residual(zp[:-1], zp[-1]), residual(zm[:-1], zm[-1])
                diagnostics.evaluations[k] += 2
                if fp is not None and fm is not None:
                    jac[:, column] = (fp-fm)/(2*h)
                elif fp is not None:
                    jac[:, column] = (fp-f0)/h
                elif fm is not None:
                    jac[:, column] = (f0-fm)/h
                else:
                    success = False
                    break
            if not success or not np.all(np.isfinite(jac)):
                return y, False
            try:
                next_tangent = np.linalg.solve(np.vstack((jac, tangent)), np.r_[np.zeros(len(y)), 1.])
            except np.linalg.LinAlgError:
                return y, False
            next_tangent /= np.linalg.norm(next_tangent)
            prediction = z+step*next_tangent

            def augmented(value):
                f = residual(value[:-1], value[-1])
                return None if f is None else np.r_[f, np.dot(value-prediction, next_tangent)]

            corrected, _, success = correct(augmented, prediction)
            trial, trial_t = corrected[:-1], corrected[-1]
            success = success and np.linalg.norm(trial-y) <= options.max_root_distance
            if success and (t-1.)*(trial_t-1.) <= 0 and trial_t != t:
                end_prediction = y+(1.-t)/(trial_t-t)*(trial-y)
                end, _, end_ok = correct(lambda v: residual(v, 1.), end_prediction, at_target=True)
                if end_ok and np.linalg.norm(end-y) <= options.max_root_distance and valid_root(end, 1.):
                    diagnostics.continuation_steps[k] += 1
                    return end, True
                success = False
        success = (success and np.linalg.norm(trial-y) <= options.max_root_distance and
                   valid_root(trial, trial_t))
        if not success:
            diagnostics.continuation_retries[k] += 1
            step *= .5
            if step < options.min_step:
                return y, False
            continue
        previous = (y.copy(), t)
        y, t = trial, trial_t
        if options.method == "arclength":
            tangent = next_tangent
        diagnostics.continuation_steps[k] += 1
        if t == 1.:
            return y, True
        step = min(options.max_step, step*1.5)
    return y, False


def find_guarded_roots(residual, starts, search, diagnostics, k, *, max_roots=16, deflation=True, known_roots=(),
                       atlas_flags=None, origins=None):
    """Finite multistart/shifted-deflation search; verify the original equations.

    Coordinates must be dimensionless. Known roots are removed only from the
    search residual. No successful deflated solve bypasses the original check.
    """
    if not isinstance(max_roots, int) or isinstance(max_roots, bool) or max_roots < 1:
        raise ValueError("max_roots must be a positive integer")
    if not isinstance(deflation, bool):
        raise ValueError("deflation must be a bool")
    roots = [np.asarray(root).copy() for root in known_roots]
    initial_count = len(roots)

    def modified(value):
        raw = residual(value)
        if raw is None or not np.all(np.isfinite(raw)):
            return None
        if not deflation or not roots:
            return raw
        log_factor = 0.
        for root in roots:
            distance2 = float(np.dot(value-root, value-root))
            if distance2 < 1e-20:
                return None
            log_factor += math.log1p(1./distance2)
        if log_factor > 200.:
            return None
        return raw*math.exp(log_factor)

    for index, start in enumerate(starts):
        while diagnostics.starts[k] < search.max_starts and len(roots) < max_roots:
            diagnostics.starts[k] += 1
            if atlas_flags is not None and atlas_flags[index]:
                diagnostics.atlas_starts[k] += 1
            value, _, success = solve_guarded_system(modified, start, search, diagnostics, k)
            diagnostics.evaluations[k] += 1
            raw = residual(value)
            if not success or raw is None or not np.all(np.isfinite(raw)):
                diagnostics.unconverged[k] += 1
                break
            norm = float(np.max(np.abs(raw)))
            diagnostics.best_residual[k] = min(diagnostics.best_residual[k], norm)
            if norm > search.residual_tolerance or any(np.linalg.norm(value-root) < 1e-6 for root in roots):
                break
            roots.append(value.copy())
            if origins is not None:
                origins.append(index)
            if not deflation:
                break
            diagnostics.deflations[k] += 1
    return roots[initial_count:]
