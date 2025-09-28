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


__all__ = [
    "rhs_cubic",
    "rhs_lorenz",
    "rhs_logistic",
    "logistic_analytic",
    "rhs_forced_lorenz",
]
