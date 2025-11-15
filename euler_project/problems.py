"""Right-hand side definitions for the ODE problems."""

from __future__ import annotations  # postpone evaluation of type hints (forward refs)

import numpy as np

Array = np.ndarray


def rhs_cubic(t: float, x: Array, q: float) -> Array:
    """Cubic scalar ODE ``x' = q x - x^3``.

    Parameters
    ----------
    t:
        Time variable (unused; included for API compatibility).
    x:
        State array of shape ``(1,)``.
    q:
        Scalar parameter multiplying the linear term.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(1,)`` with the derivative.
    """

    _ = t
    return q * x - x ** 3


def rhs_lorenz(t: float, X: Array, a: float = 10.0, b: float = 25.0, c: float = 8.0 / 3.0) -> Array:
    """Lorenz 63 system."""

    x1, x2, x3 = X
    return np.array(
        [
            a * (x2 - x1),
            b * x1 - x2 - x1 * x3,
            -c * x3 + x1 * x2,
        ],
        dtype=float,
    )


def rhs_logistic(t: float, x: Array, q: float) -> Array:
    """Logistic growth ODE ``x' = q x (1 - x)`` for scalar ``x``.

    Parameters
    ----------
    t
        Time (unused; present for compatibility).
    x
        State array of shape ``(1,)``.
    q
        Growth parameter.
    """

    _ = t
    return q * x * (1.0 - x)


def logistic_analytic(t: Array | float, x0: float, q: float) -> Array:
    """Closed-form solution of the logistic ODE.

    Returns an array with the same shape as ``t`` (or a scalar if ``t`` is
    scalar). The formula is

    ``x(t) = 1 / (1 + ((1 - x0)/x0) * exp(-q t))``.
    """

    tt = np.asarray(t, dtype=float)
    ratio = (1.0 - x0) / x0
    xt = 1.0 / (1.0 + ratio * np.exp(-q * tt))
    return xt


def rhs_forced_lorenz(
    t: float,
    X: Array,
    a: float = 10.0,
    b: float = 25.0,
    c: float = 8.0 / 3.0,
    forcing_amplitude: float = 100.0,
) -> Array:
    """Lorenz 63 system with sinusoidal forcing in the first equation.

    x1' = a (x2 - x1) + A sin(t)
    x2' = b x1 - x2 - x1 x3
    x3' = -c x3 + x1 x2
    """

    x1, x2, x3 = X
    return np.array(
        [
            a * (x2 - x1) + forcing_amplitude * np.sin(t),
            b * x1 - x2 - x1 * x3,
            -c * x3 + x1 * x2,
        ],
        dtype=float,
    )


def rhs_forced_lorenz_const(
    t: float,
    X: Array,
    a: float = 10.0,
    b: float = 25.0,
    c: float = 8.0 / 3.0,
    forcing_amplitude: float = 100.0,
) -> Array:
    """Lorenz 63 with constant forcing in x1' (epsilon(t) = A)."""

    x1, x2, x3 = X
    return np.array(
        [
            a * (x2 - x1) + forcing_amplitude,
            b * x1 - x2 - x1 * x3,
            -c * x3 + x1 * x2,
        ],
        dtype=float,
    )

def rhs_cos2_arctan_problem(t: float, x: Array) -> Array:
    """Nonlinear scalar ODE ``x'(t) = 30 cos^2(x(t))``.

    Designed for the exercise where the exact solution is
    ``x(t) = arctan(30 t - 120)`` with initial value ``x(0) = arctan(-120)``.
    Works element-wise for vector ``x`` as well.
    """

    _ = t
    return 30.0 * np.cos(x) ** 2


def arctan_analytic(t: Array | float) -> Array:
    """Exact solution ``x(t) = arctan(30 t - 120)``.

    Returns an array broadcast to the shape of ``t``.
    """

    tt = np.asarray(t, dtype=float)
    return np.arctan(30.0 * tt - 120.0)


__all__ = [
    "rhs_cubic",
    "rhs_lorenz",
    "rhs_logistic",
    "logistic_analytic",
    "rhs_forced_lorenz",
    "rhs_forced_lorenz_const",
    "rhs_cos2_arctan_problem",
    "arctan_analytic",
]


def rhs_harmonic_oscillator(t: float, x: Array, alpha: float = 0.0) -> Array:
    """2D (damped) harmonic oscillator.

    System: x1' = x2,  x2' = -x1 - alpha x2.  ``alpha=0`` is undamped.
    """

    _ = t
    x1, x2 = x
    return np.array([x2, -x1 - alpha * x2], dtype=float)


def oscillator_exact_undamped(t: Array | float) -> Array:
    """Exact solution for the undamped oscillator with x(0)=(1,0).

    Returns array with shape ``(..., 2)`` for array ``t`` or shape ``(2,)`` for
    scalar ``t``.  The solution is (x1, x2) = (cos t, -sin t).
    """

    tt = np.asarray(t, dtype=float)
    x1 = np.cos(tt)
    x2 = -np.sin(tt)
    if tt.ndim == 0:
        return np.array([float(x1), float(x2)], dtype=float)
    else:
        return np.vstack([x1, x2]).T


__all__ += ["rhs_harmonic_oscillator", "oscillator_exact_undamped"]
