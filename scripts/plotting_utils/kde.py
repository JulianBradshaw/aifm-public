"""
Standard Gaussian KDE helper.

The previous implementation tried to hand-tune the bandwidth with several
heuristics. That made the plots hard to reason about and diverged from textbook
KDE behaviour. This helper simply defers to SciPy's built-in `gaussian_kde`
using its default Scott's rule bandwidth, which is the canonical choice for
one-dimensional densities.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.special import gammaln
from scipy.stats import gaussian_kde, norm

SQRT_TWO_PI = float(np.sqrt(2.0 * np.pi))


def make_gaussian_kde(data: np.ndarray) -> gaussian_kde:
    """
    Construct a Gaussian KDE with Scott's rule bandwidth.

    Args:
        data: Array of samples (must have at least 2 elements)

    Returns:
        scipy.stats.gaussian_kde object

    Raises:
        ValueError: If data has fewer than 2 samples
    """
    data = np.asarray(data, dtype=float)
    if data.size < 2:
        raise ValueError("Need at least two samples for KDE.")
    return gaussian_kde(data, bw_method="scott")


@dataclass
class LowerBoundedKDE:
    """
    Density estimator with hard support [lower_bound, inf).

    The estimator trains a standard Gaussian KDE in the log-transformed space
    and exposes helper methods that operate on the original scale.
    """

    estimator: "KernelDensity"
    lower_bound: float
    tau: float

    def logpdf(self, t: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return log density on the original time axis."""
        t = np.asarray(t, dtype=float)
        log_density = np.full(t.shape, -np.inf)
        mask = t > self.lower_bound
        if not np.any(mask):
            return log_density

        shifted = t[mask] - self.lower_bound
        y = np.log(shifted / self.tau)
        log_density[mask] = (
            self.estimator.score_samples(y[:, None]) - np.log(shifted)
        )
        return log_density

    def pdf(self, t: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return density on the original time axis."""
        return np.exp(self.logpdf(t))

    def evaluate(self, t: Sequence[float] | np.ndarray) -> np.ndarray:
        """Alias kept for API parity with scipy's gaussian_kde."""
        return self.pdf(t)

    __call__ = pdf


@dataclass
class GammaKernelKDE:
    """
    Positive-support KDE using gamma kernels (Chen 2000).

    Each sample contributes a gamma pdf whose shape depends on the
    sample-to-boundary distance and a common bandwidth parameter,
    producing densities that taper to zero exactly at the boundary.
    """

    lower_bound: float
    shifted_samples: np.ndarray
    alphas: np.ndarray
    bandwidth: float

    def pdf(self, t: Sequence[float] | np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        density = np.zeros_like(t, dtype=float)
        mask = t > self.lower_bound
        if not np.any(mask):
            return density

        y = t[mask] - self.lower_bound
        # Avoid log(0); zeros stay zero density.
        positive = y > 0
        if not np.any(positive):
            return density

        y_pos = y[positive]
        log_y = np.log(y_pos)

        # Broadcasting across samples.
        alphas = self.alphas[:, None]
        log_kernel = (
            (alphas - 1.0) * log_y
            - y_pos / self.bandwidth
            - gammaln(alphas)
            - alphas * np.log(self.bandwidth)
        )
        contributions = np.exp(log_kernel)
        density_segment = contributions.mean(axis=0)

        tmp = np.zeros_like(y)
        tmp[positive] = density_segment
        density[mask] = tmp
        return density

    def logpdf(self, t: Sequence[float] | np.ndarray) -> np.ndarray:
        pdf_vals = self.pdf(t)
        with np.errstate(divide="ignore"):
            return np.log(pdf_vals)

    evaluate = pdf
    __call__ = pdf


@dataclass
class CutAndNormalizeKDE:
    """
    Boundary-corrected Gaussian KDE (Gasser–Müller cut-and-normalize).

    Each Gaussian kernel is truncated below the lower bound and renormalized so
    that the total mass above the boundary remains one, following Gasser &
    Müller's classic proposal for positive-support densities.
    """

    lower_bound: float
    samples: np.ndarray
    bandwidth: float
    column_weights: np.ndarray

    def _evaluate_positive(self, points: np.ndarray) -> np.ndarray:
        """Return the density evaluated at `points` (assumed >= lower_bound)."""
        shifted = (points[:, None] - self.samples[None, :]) / self.bandwidth
        kernels = np.exp(-0.5 * shifted**2)
        return kernels @ self.column_weights

    def pdf(self, t: Sequence[float] | np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        density = np.zeros_like(t, dtype=float)
        mask = t >= self.lower_bound
        if not np.any(mask):
            return density
        density[mask] = self._evaluate_positive(t[mask])
        return density

    def logpdf(self, t: Sequence[float] | np.ndarray) -> np.ndarray:
        pdf_vals = self.pdf(t)
        with np.errstate(divide="ignore"):
            return np.log(pdf_vals)

    evaluate = pdf
    __call__ = pdf


def make_lower_bounded_kde(
    data: Iterable[float],
    lower_bound: float,
    tau: float = 1.0,
    bandwidth: float | None = None,
    bandwidth_grid: Sequence[float] | None = None,
    cv: int | None = None,
    min_offset_days: float = 1.0,
) -> LowerBoundedKDE:
    """
    Build a KDE whose support is constrained to [lower_bound, inf).

    The samples are first transformed with y = log((t - lower_bound) / tau),
    a Gaussian KDE is fit in that unconstrained space using
    cross-validated bandwidth selection, and the estimator is then wrapped to
    evaluate densities on the original axis via the change-of-variables rule.

    Args:
        data: One-dimensional sample array.
        lower_bound: Time that represents "today" / minimum support.
        tau: Positive scaling constant used in the log transform.
        bandwidth: Optional fixed Gaussian bandwidth that skips the grid search.
        bandwidth_grid: Optional sequence of candidate bandwidths for CV.
        cv: Optional cross-validation fold count (defaults to min(10, n)).
        min_offset_days: Clip samples that fall on the boundary by this many
            days to avoid log(0).

    Returns:
        LowerBoundedKDE instance that can evaluate the density.

    Raises:
        ImportError: If scikit-learn is not available.
        ValueError: If validation fails (not enough samples, bad params, etc).
    """
    try:
        from sklearn.model_selection import GridSearchCV
        from sklearn.neighbors import KernelDensity
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "scikit-learn is required for make_lower_bounded_kde(). "
            "Install it with `pip install scikit-learn`."
        ) from exc

    data = np.asarray(list(data), dtype=float)
    if data.size < 2:
        raise ValueError("Need at least two samples for KDE.")

    if tau <= 0:
        raise ValueError("tau must be positive.")

    lower_bound = float(lower_bound)
    eps = max(min_offset_days / 365.25, np.finfo(float).eps)
    valid = data >= lower_bound
    data = data[valid]
    if data.size < 2:
        raise ValueError(
            "Need at least two samples on or after the lower bound to "
            "construct the KDE."
        )

    shifted = np.clip(data - lower_bound, eps, None)
    y = np.log(shifted / tau)

    if bandwidth is not None:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        best_kde = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth))
        best_kde.fit(y[:, None])
    else:
        if bandwidth_grid is None:
            bandwidth_grid = np.logspace(-1.5, 0.8, 40)
        else:
            bandwidth_grid = np.asarray(bandwidth_grid, dtype=float)
            if np.any(bandwidth_grid <= 0):
                raise ValueError("bandwidth_grid values must be positive.")

        if cv is None:
            cv = min(10, y.size)
        if cv < 2:
            raise ValueError("cv must be at least 2.")
        if cv > y.size:
            raise ValueError("cv cannot exceed the number of valid samples.")

        grid = {"bandwidth": bandwidth_grid}
        base_kde = KernelDensity(kernel="gaussian")
        search = GridSearchCV(base_kde, grid, cv=cv)
        search.fit(y[:, None])
        best_kde = search.best_estimator_

    return LowerBoundedKDE(
        estimator=best_kde,
        lower_bound=lower_bound,
        tau=float(tau),
    )


def _silverman_bandwidth(samples: np.ndarray) -> float:
    n = samples.size
    if n < 2:
        raise ValueError("Need at least two samples to estimate bandwidth.")
    std = float(np.std(samples, ddof=1))
    iqr = float(np.subtract(*np.percentile(samples, [75, 25])))
    sigma_candidates = [value for value in (std, iqr / 1.349 if iqr > 0 else 0.0) if value > 0]
    sigma = min(sigma_candidates) if sigma_candidates else std
    if sigma <= 0:
        sigma = max(np.mean(samples), 1.0)
    h = 0.9 * sigma * (n ** (-1 / 5))
    return float(max(h, np.finfo(float).eps))


def _build_gamma_kernel_estimator(
    *,
    lower_bound: float,
    shifted_samples: np.ndarray,
    bandwidth: float,
    alpha_epsilon: float,
) -> GammaKernelKDE:
    if shifted_samples.size < 2:
        raise ValueError("Need at least two samples on or after the lower bound.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    alphas = shifted_samples / bandwidth + 1.0 + float(alpha_epsilon)
    return GammaKernelKDE(
        lower_bound=lower_bound,
        shifted_samples=shifted_samples,
        alphas=alphas,
        bandwidth=float(bandwidth),
    )


def _default_gamma_bandwidth_grid(shifted: np.ndarray) -> np.ndarray:
    base = _silverman_bandwidth(shifted)
    coarse = np.logspace(-1.0, 1.0, 35)
    fine = np.linspace(0.7, 1.3, 25)
    factors = np.concatenate([coarse, fine])
    grid = base * factors
    grid = np.clip(grid, np.finfo(float).eps, None)
    return np.unique(np.sort(grid))


def _score_gamma_bandwidth(
    samples: np.ndarray,
    *,
    lower_bound: float,
    bandwidth: float,
    alpha_epsilon: float,
    cv: int,
) -> float:
    if bandwidth <= 0:
        return -np.inf
    n = samples.size
    if n < 3:
        return -np.inf
    try:
        from sklearn.model_selection import KFold, LeaveOneOut
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "scikit-learn is required for gamma kernel bandwidth search. "
            "Install it with `pip install scikit-learn`."
        ) from exc

    indices = np.arange(n)
    if cv >= n:
        splitter = LeaveOneOut()
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=0)

    log_scores: list[float] = []
    for train_idx, test_idx in splitter.split(indices):
        train = samples[train_idx]
        if train.size < 2:
            continue
        shifted_train = train - lower_bound
        shifted_train = shifted_train[shifted_train >= 0]
        if shifted_train.size < 2:
            continue
        estimator = _build_gamma_kernel_estimator(
            lower_bound=lower_bound,
            shifted_samples=shifted_train,
            bandwidth=bandwidth,
            alpha_epsilon=alpha_epsilon,
        )
        log_vals = estimator.logpdf(samples[test_idx])
        # If the estimator underflowed to zero anywhere, treat this candidate as invalid.
        if not np.all(np.isfinite(log_vals)):
            return -np.inf
        log_scores.extend(log_vals.tolist())

    if not log_scores:
        return -np.inf
    return float(np.mean(log_scores))


def _select_gamma_bandwidth(
    samples: np.ndarray,
    *,
    lower_bound: float,
    alpha_epsilon: float,
    bandwidth_grid: Sequence[float] | None,
    cv: int,
) -> float:
    shifted = samples - lower_bound
    if np.any(shifted < 0):
        raise ValueError("All samples must be on or after the lower bound.")

    if bandwidth_grid is None:
        grid = _default_gamma_bandwidth_grid(shifted)
    else:
        grid = np.asarray(list(bandwidth_grid), dtype=float)
    grid = grid[np.isfinite(grid)]
    grid = grid[grid > 0]
    if grid.size == 0:
        raise ValueError(
            "bandwidth_grid must contain at least one positive, finite value."
        )

    best_bandwidth = None
    best_score = -np.inf
    for candidate in grid:
        score = _score_gamma_bandwidth(
            samples,
            lower_bound=lower_bound,
            bandwidth=float(candidate),
            alpha_epsilon=alpha_epsilon,
            cv=cv,
        )
        if not np.isfinite(score):
            continue
        if score > best_score:
            best_score = score
            best_bandwidth = float(candidate)

    if best_bandwidth is None:
        raise RuntimeError(
            "Gamma kernel bandwidth search failed to find a valid candidate."
        )
    return best_bandwidth


def make_gamma_kernel_kde(
    data: Iterable[float],
    lower_bound: float,
    bandwidth: float | None = None,
    alpha_epsilon: float = 1e-3,
    bandwidth_grid: Sequence[float] | None = None,
    cv: int | None = None,
) -> GammaKernelKDE:
    """
    Build a gamma-kernel KDE on [lower_bound, inf).

    Args:
        data: Sample times.
        lower_bound: Minimum support (typically present_day).
        bandwidth: Optional smoothing bandwidth. When omitted, a grid search
            tuned via a proper scoring rule (mean log predictive density)
            selects the bandwidth.
        alpha_epsilon: Ensures gamma shape parameters stay > 1 for zero density at boundary.
        bandwidth_grid: Optional candidate bandwidths for the scoring-rule search.
        cv: Optional fold count for the scoring-rule search (defaults to min(10, n)).
    """
    data = np.asarray(list(data), dtype=float)
    if data.size < 2:
        raise ValueError("Need at least two samples for KDE.")

    lower_bound = float(lower_bound)
    samples = data[data >= lower_bound]
    if samples.size < 2:
        raise ValueError("Need at least two samples on or after the lower bound.")

    shifted = samples - lower_bound
    baseline_bandwidth = _silverman_bandwidth(shifted)

    selected_bandwidth = bandwidth
    if selected_bandwidth is None:
        if samples.size >= 3:
            search_cv = cv if cv is not None else min(10, samples.size)
            if search_cv < 2:
                raise ValueError("cv must be at least 2.")
            if search_cv > samples.size:
                raise ValueError("cv cannot exceed the number of samples.")
            try:
                selected_bandwidth = _select_gamma_bandwidth(
                    samples,
                    lower_bound=lower_bound,
                    alpha_epsilon=alpha_epsilon,
                    bandwidth_grid=bandwidth_grid,
                    cv=search_cv,
                )
            except Exception as exc:
                warnings.warn(
                    (
                        "Gamma kernel bandwidth search failed; falling back to "
                        f"Silverman's rule. Reason: {exc}"
                    ),
                    RuntimeWarning,
                )
                selected_bandwidth = baseline_bandwidth
        else:
            selected_bandwidth = baseline_bandwidth

    if selected_bandwidth is None or selected_bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    return _build_gamma_kernel_estimator(
        lower_bound=lower_bound,
        shifted_samples=shifted,
        bandwidth=float(selected_bandwidth),
        alpha_epsilon=alpha_epsilon,
    )


def _build_cut_and_normalize_estimator(
    *,
    lower_bound: float,
    samples: np.ndarray,
    bandwidth: float,
) -> CutAndNormalizeKDE:
    if samples.size < 2:
        raise ValueError("Need at least two samples on or after the lower bound.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    scaled_boundary = (lower_bound - samples) / bandwidth
    survival = norm.sf(scaled_boundary)
    epsilon = np.finfo(float).tiny
    survival = np.clip(survival, epsilon, None)

    denom = samples.size * bandwidth * SQRT_TWO_PI
    column_weights = (1.0 / survival) / denom
    return CutAndNormalizeKDE(
        lower_bound=lower_bound,
        samples=samples,
        bandwidth=float(bandwidth),
        column_weights=column_weights,
    )


def _default_cut_and_normalize_bandwidth_grid(shifted: np.ndarray) -> np.ndarray:
    base = _silverman_bandwidth(shifted)
    coarse = np.logspace(-1.0, 1.0, 35)
    fine = np.linspace(0.7, 1.3, 25)
    factors = np.concatenate([coarse, fine])
    grid = base * factors
    grid = np.clip(grid, np.finfo(float).eps, None)
    return np.unique(np.sort(grid))


def _score_cut_and_normalize_bandwidth(
    samples: np.ndarray,
    *,
    lower_bound: float,
    bandwidth: float,
    cv: int,
) -> float:
    if bandwidth <= 0:
        return -np.inf
    n = samples.size
    if n < 3:
        return -np.inf
    try:
        from sklearn.model_selection import KFold, LeaveOneOut
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "scikit-learn is required for Gasser–Müller bandwidth search. "
            "Install it with `pip install scikit-learn`."
        ) from exc

    indices = np.arange(n)
    if cv >= n:
        splitter = LeaveOneOut()
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=0)

    log_scores: list[float] = []
    for train_idx, test_idx in splitter.split(indices):
        train = samples[train_idx]
        if train.size < 2:
            continue
        estimator = _build_cut_and_normalize_estimator(
            lower_bound=lower_bound,
            samples=train,
            bandwidth=bandwidth,
        )
        log_vals = estimator.logpdf(samples[test_idx])
        finite = log_vals[np.isfinite(log_vals)]
        if finite.size:
            log_scores.extend(finite.tolist())

    if not log_scores:
        return -np.inf
    return float(np.mean(log_scores))


def _select_cut_and_normalize_bandwidth(
    samples: np.ndarray,
    *,
    lower_bound: float,
    bandwidth_grid: Sequence[float] | None,
    cv: int,
) -> float:
    shifted = samples - lower_bound
    if np.any(shifted < 0):
        raise ValueError("All samples must be on or after the lower bound.")

    if bandwidth_grid is None:
        grid = _default_cut_and_normalize_bandwidth_grid(shifted)
    else:
        grid = np.asarray(list(bandwidth_grid), dtype=float)
    grid = grid[np.isfinite(grid)]
    grid = grid[grid > 0]
    if grid.size == 0:
        raise ValueError(
            "bandwidth_grid must contain at least one positive, finite value."
        )

    best_bandwidth = None
    best_score = -np.inf
    for candidate in grid:
        score = _score_cut_and_normalize_bandwidth(
            samples,
            lower_bound=lower_bound,
            bandwidth=float(candidate),
            cv=cv,
        )
        if not np.isfinite(score):
            continue
        if score > best_score:
            best_score = score
            best_bandwidth = float(candidate)

    if best_bandwidth is None:
        raise RuntimeError(
            "Gasser–Müller bandwidth search failed to find a valid candidate."
        )
    return best_bandwidth


def make_gasser_muller_kde(
    data: Iterable[float],
    lower_bound: float,
    bandwidth: float | None = None,
    bandwidth_grid: Sequence[float] | None = None,
    cv: int | None = None,
) -> CutAndNormalizeKDE:
    """
    Build a Gaussian KDE with Gasser–Müller cut-and-normalize boundary correction.

    Args:
        data: Sample times (must contain at least two values on/after lower_bound).
        lower_bound: Support boundary; density is zero below this value.
        bandwidth: Optional Gaussian bandwidth override. When omitted, the
            bandwidth is picked via cross-validation over `bandwidth_grid`.
        bandwidth_grid: Optional candidate bandwidths for the scoring-rule search.
        cv: Optional fold count for the scoring-rule search (defaults to min(10, n)).

    Returns:
        CutAndNormalizeKDE instance that evaluates the corrected density.

    Raises:
        ValueError: If insufficient samples are available for CV or if CV params are invalid.
    """
    samples = np.asarray(list(data), dtype=float)
    if samples.size < 2:
        raise ValueError("Need at least two samples for KDE.")

    lower_bound = float(lower_bound)
    samples = samples[samples >= lower_bound]
    if samples.size < 2:
        raise ValueError("Need at least two samples on or after the lower bound.")

    selected_bandwidth = bandwidth
    if selected_bandwidth is None:
        if samples.size < 3:
            raise ValueError(
                "Need at least three samples to run cross-validation. "
                "Provide `bandwidth` explicitly if you want to bypass CV."
            )
        search_cv = cv if cv is not None else min(10, samples.size)
        if search_cv < 2:
            raise ValueError("cv must be at least 2.")
        if search_cv > samples.size:
            raise ValueError("cv cannot exceed the number of samples.")
        selected_bandwidth = _select_cut_and_normalize_bandwidth(
            samples,
            lower_bound=lower_bound,
            bandwidth_grid=bandwidth_grid,
            cv=search_cv,
        )

    if selected_bandwidth is None or selected_bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    return _build_cut_and_normalize_estimator(
        lower_bound=lower_bound,
        samples=samples,
        bandwidth=float(selected_bandwidth),
    )
