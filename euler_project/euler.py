"""Explicit Euler integrator implementation."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

Array = np.ndarray


def explEuler(
    f: Callable[[float, Array, ...], Array],
    x0: Array,
    T: float,
    tau: float,
    *,
    t0: float = 0.0,
    args: Iterable = (),
) -> tuple[Array, Array]:
    """Integrate an ODE with the explicit Euler method.

    Parameters
    ----------
    f:
        Right-hand side of the ODE ``y'(t) = f(t, y, *args)``.
    x0:
        Initial value array of shape ``(d,)``.
    T:
        Final integration time ``T > t0``.
    tau:
        Time step, must be positive.
    t0:
        Optional initial time. Defaults to ``0.0`` to mirror the assignment
        interface where only the final time ``T`` is specified explicitly.
    args:
        Optional iterable of additional arguments forwarded to ``f``.

    Returns
    -------
    tuple[Array, Array]
        Time grid ``t`` of shape ``(N+1,)`` and solution array ``Y`` of
        shape ``(N+1, d)`` representing ``xSol`` on the computed time grid.
        ``N`` is ``floor((T - t0) / tau)``; the final time therefore does not
        exceed ``T`` and may be slightly smaller if ``T - t0`` is not an
        integer multiple of ``tau``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """

    if tau <= 0:
        raise ValueError("tau must be positive")
    if T <= t0:
        raise ValueError("T must be greater than t0")

    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be a one-dimensional array")

    duration = T - t0
    n_steps = int(np.floor(duration / tau))
    if n_steps < 1:
        raise ValueError("time span must accommodate at least one Euler step")

    t = t0 + tau * np.arange(n_steps + 1, dtype=float)
    y = np.empty((n_steps + 1, x0.size), dtype=float)
    y[0] = x0

    for k in range(n_steps):
        step = np.asarray(f(t[k], y[k], *args), dtype=float)
        if step.shape != x0.shape:
            raise ValueError("f must return an array with the same shape as x0")
        y[k + 1] = y[k] + tau * step

    return t, y


if __name__ == "__main__":
    def monotone_decay(_: float, value: Array) -> Array:
        return -value

    _, traj = explEuler(monotone_decay, np.array([1.0]), 1.0, 0.01)
    assert np.all(np.diff(traj[:, 0]) <= 0.0), "Scalar decay should be monotone"

    matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])

    def linear_rhs(_: float, vec: Array) -> Array:
        return matrix @ vec

    _, vec_traj = explEuler(linear_rhs, np.array([1.0, 0.0]), 0.1, 0.05)
    expected = np.array([1.0, 0.0]) + 0.05 * matrix @ np.array([1.0, 0.0])
    assert np.allclose(vec_traj[1], expected), "One Euler step should match explicit formula"

    print("Self-checks passed.")
