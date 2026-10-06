"""Dimensionless, stateless nonlinear search controls and diagnostics."""
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass(frozen=True)
class SearchOptions:
    """auto: scalar B/C then Newton with LM recovery; bracket: explicit B/C only.

    Tolerance bounds every normalized equation. potential_extent is a finite
    search limit in units of the photoelectron temperature, not an absence proof.
    max_starts limits nonlinear starts; bracket_points controls the scalar grid.
    """
    method: str = "auto"
    residual_tolerance: float = 1e-10
    max_iterations: int = 100
    max_backtracks: int = 24
    max_starts: int = 32
    use_default_guesses: bool = True
    bracket_points: int = 96
    potential_extent: float = 200.

    def __post_init__(self):
        if self.method not in {"auto", "newton", "lm", "bracket"}:
            raise ValueError("method must be auto, newton, lm, or bracket")
        for name in ("residual_tolerance", "potential_extent"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, minimum in (("max_iterations", 0), ("max_backtracks", 1),
                              ("max_starts", 1), ("bracket_points", 2)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.use_default_guesses, bool):
            raise ValueError("use_default_guesses must be a bool")


def _zeros():
    return [0, 0, 0]


@dataclass
class SearchDiagnostics:
    """Counters ordered A/B/C. Located roots do not prove search completeness."""
    searched: list[bool] = field(default_factory=lambda: [False]*3)
    excluded: list[bool] = field(default_factory=lambda: [False]*3)
    starts: list[int] = field(default_factory=_zeros)
    unconverged: list[int] = field(default_factory=_zeros)
    rejected: list[int] = field(default_factory=_zeros)
    profile_failures: list[int] = field(default_factory=_zeros)
    roots_found: list[int] = field(default_factory=_zeros)
    evaluations: list[int] = field(default_factory=_zeros)
    iterations: list[int] = field(default_factory=_zeros)
    lm_steps: list[int] = field(default_factory=_zeros)
    brackets: list[int] = field(default_factory=_zeros)
    atlas_starts: list[int] = field(default_factory=_zeros)
    atlas_hits: list[int] = field(default_factory=_zeros)
    continuation_steps: list[int] = field(default_factory=_zeros)
    continuation_retries: list[int] = field(default_factory=_zeros)
    deflations: list[int] = field(default_factory=_zeros)
    best_residual: list[float] = field(default_factory=lambda: [math.inf]*3)

    def include(self, other):
        """Accumulate independently attempted branches for one auto search."""
        for k in range(3):
            self.searched[k] |= other.searched[k]
            self.excluded[k] |= other.excluded[k]
            self.best_residual[k] = min(self.best_residual[k], other.best_residual[k])
            for name in ("starts", "unconverged", "rejected", "profile_failures", "roots_found",
                         "evaluations", "iterations", "lm_steps", "brackets", "atlas_starts", "atlas_hits",
                         "continuation_steps", "continuation_retries", "deflations"):
                getattr(self, name)[k] += getattr(other, name)[k]


class SearchFailure(RuntimeError):
    """A failed finite search, with its diagnostics and physical exclusion count."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def solve_guarded_system(residual, start, options, diagnostics, k):
    """Newton/LM with central or domain-aware one-sided differences.

    Invalid callback states are None. Small steps and least-squares stationary
    points are never accepted without satisfying the equation residuals.
    """
    def evaluate(y):
        diagnostics.evaluations[k] += 1
        f = residual(y)
        if f is None or not np.all(np.isfinite(f)):
            return None
        return np.asarray(f, dtype=float)

    y = np.asarray(start, dtype=float).copy()
    f = evaluate(y)
    if f is None:
        return y, math.inf, False
    norm = float(np.max(np.abs(f)))

    def accept(direction):
        for backtrack in range(options.max_backtracks):
            trial = y + (0.5**backtrack)*direction
            ft = evaluate(trial)
            if ft is not None and np.max(np.abs(ft)) < norm:
                return trial, ft
        return None

    for iteration in range(options.max_iterations+1):
        diagnostics.best_residual[k] = min(diagnostics.best_residual[k], norm)
        if norm <= options.residual_tolerance:
            return y, norm, True
        if iteration == options.max_iterations:
            break
        jac = np.empty((len(y), len(y)))
        for column in range(len(y)):
            h = np.finfo(float).eps**(1/3)*max(1., abs(y[column]))
            yp, ym = y.copy(), y.copy()
            yp[column] += h
            ym[column] -= h
            fp, fm = evaluate(yp), evaluate(ym)
            if fp is not None and fm is not None:
                jac[:, column] = (fp-fm)/(2*h)
            elif fp is not None:
                jac[:, column] = (fp-f)/h
            elif fm is not None:
                jac[:, column] = (f-fm)/h
            else:
                return y, norm, False
        accepted = None
        if options.method != "lm":
            try:
                accepted = accept(np.linalg.solve(jac, -f))
            except np.linalg.LinAlgError:
                pass
        if options.method == "lm" or (options.method == "auto" and accepted is None):
            normal = jac.T@jac
            diagonal = np.maximum(np.diag(normal), 1e-12)
            damping = 1e-3
            for _ in range(options.max_backtracks):
                try:
                    direction = np.linalg.solve(normal+damping*np.diag(diagonal), -jac.T@f)
                    accepted = accept(direction)
                except np.linalg.LinAlgError:
                    pass
                if accepted is not None:
                    diagnostics.lm_steps[k] += 1
                    break
                damping *= 10
        if accepted is None:
            break
        y, f = accepted
        norm = float(np.max(np.abs(f)))
        diagnostics.iterations[k] += 1
    diagnostics.best_residual[k] = min(diagnostics.best_residual[k], norm)
    return y, norm, False
