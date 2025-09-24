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


__all__ = ["rhs_cubic", "rhs_lorenz"]
