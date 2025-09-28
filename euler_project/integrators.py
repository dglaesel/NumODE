"""Explicit time integrators.

This module originally provided only the explicit Euler method. It now also
contains a generic explicit Runge–Kutta (ERK) integrator that is configured by
its Butcher tableau ``(A, b, c)``. Both autonomous ``f(x)`` and
non‑autonomous ``f(t, x)`` right‑hand sides are supported.
"""

from __future__ import annotations  # postpone evaluation of type hints (forward refs)

import inspect
from typing import Callable, Iterable, Tuple

import numpy as np

Array = np.ndarray


class ExplicitEuler:
    """Explicit Euler for autonomous IVPs."""

    def __init__(self, f: Callable, x0: Iterable[float], T: float, tau: float):
        if tau <= 0:
            raise ValueError("tau > 0 required")
        if T <= 0:
            raise ValueError("T > 0 required")
        x = np.atleast_1d(np.array(x0, dtype=float))
        if x.ndim != 1:
            raise ValueError("x0 must be 1-D")
        self.f = f
        self.x0 = x
        self.T = float(T)
        self.tau = float(tau)
        self._arity_is_unary = len(inspect.signature(f).parameters) == 1

    def run(self) -> Tuple[Array, Array]:
        d = self.x0.size
        n_steps = int(np.floor(self.T / self.tau))
        if n_steps < 1:
            raise ValueError("integration horizon too short")
        t = self.tau * np.arange(n_steps + 1, dtype=float)
        X = np.empty((n_steps + 1, d), dtype=float)
        X[0] = self.x0
        for k in range(n_steps):
            if self._arity_is_unary:
                rhs = self.f(X[k])
            else:
                rhs = self.f(t[k], X[k])
            rhs = np.asarray(rhs, dtype=float).reshape(-1)
            if rhs.shape != (d,):
                raise ValueError(f"f returned shape {rhs.shape}, expected {(d,)}")
            X[k + 1] = X[k] + self.tau * rhs
        return t, X


def explEuler(f: Callable, x0: Iterable[float], T: float, tau: float) -> Tuple[Array, Array]:
    """Assignment 2.3(a) wrapper."""

    return ExplicitEuler(f, x0, T, tau).run()

class ExplicitRungeKutta:
    """Generic explicit Runge–Kutta (ERK) method.

    Parameters
    ----------
    f
        Right‑hand side. Can be autonomous ``f(x)`` or non‑autonomous
        ``f(t, x)``; the arity is detected automatically.
    x0
        Initial value (1‑D array‑like; supports vector states).
    T
        Final time ``T > 0``.
    tau
        Step size ``tau > 0``.
    A, b, c
        Butcher coefficients. ``b`` and ``c`` must be 1‑D of equal length
        ``s``. ``A`` must be an ``(s, s)`` strictly lower‑triangular matrix
        (zeros on and above the diagonal).
    """

    def __init__(
        self,
        f: Callable,
        x0: Iterable[float],
        T: float,
        tau: float,
        A: Iterable[Iterable[float]],
        b: Iterable[float],
        c: Iterable[float],
    ) -> None:
        if tau <= 0:
            raise ValueError("tau > 0 required")
        if T <= 0:
            raise ValueError("T > 0 required")

        x = np.atleast_1d(np.array(x0, dtype=float))
        if x.ndim != 1:
            raise ValueError("x0 must be 1-D")

        b = np.asarray(b, dtype=float).reshape(-1)
        c = np.asarray(c, dtype=float).reshape(-1)
        if b.ndim != 1 or c.ndim != 1:
            raise ValueError("b and c must be 1-D arrays")
        if b.size != c.size:
            raise ValueError("b and c must have the same length")
        s = b.size
        A = np.asarray(A, dtype=float)
        if A.shape != (s, s):
            raise ValueError(f"A must have shape {(s, s)}, got {A.shape}")
        # Strictly lower triangular (zeros on diag and above)
        if not np.allclose(A, np.tril(A, k=-1), atol=1e-14, rtol=0.0):
            raise ValueError("A must be strictly lower triangular for an explicit RK method")

        self.f = f
        self.x0 = x
        self.T = float(T)
        self.tau = float(tau)
        self.A = A
        self.b = b
        self.c = c
        self.s = s
        self._arity_is_unary = len(inspect.signature(f).parameters) == 1

    def run(self) -> Tuple[Array, Array]:
        d = self.x0.size
        n_steps = int(np.floor(self.T / self.tau))
        if n_steps < 1:
            raise ValueError("integration horizon too short")
        t = self.tau * np.arange(n_steps + 1, dtype=float)
        X = np.empty((n_steps + 1, d), dtype=float)
        X[0] = self.x0

        A, b, c = self.A, self.b, self.c

        for n in range(n_steps):
            k_list: list[Array] = []
            x_n = X[n]
            t_n = t[n]
            for i in range(self.s):
                incr = np.zeros_like(x_n)
                if i > 0:
                    # Sum over j < i
                    for j in range(i):
                        if A[i, j] != 0.0:
                            incr = incr + A[i, j] * k_list[j]
                x_stage = x_n + self.tau * incr
                t_stage = t_n + c[i] * self.tau
                if self._arity_is_unary:
                    k_i = np.asarray(self.f(x_stage), dtype=float).reshape(-1)
                else:
                    k_i = np.asarray(self.f(t_stage, x_stage), dtype=float).reshape(-1)
                if k_i.shape != (d,):
                    raise ValueError(f"f returned shape {k_i.shape}, expected {(d,)}")
                k_list.append(k_i)

            # Update step
            increment = np.zeros(d, dtype=float)
            for i in range(self.s):
                if b[i] != 0.0:
                    increment += b[i] * k_list[i]
            X[n + 1] = x_n + self.tau * increment

        return t, X


def exRungeKutta(
    f: Callable,
    x0: Iterable[float],
    T: float,
    tau: float,
    A: Iterable[Iterable[float]],
    b: Iterable[float],
    c: Iterable[float],
) -> Tuple[Array, Array]:
    """Convenience wrapper for :class:`ExplicitRungeKutta`.

    Returns ``(t, X)`` on the uniform grid ``t_n = n * tau`` with
    ``n = 0..floor(T/tau)``.
    """

    return ExplicitRungeKutta(f, x0, T, tau, A, b, c).run()


__all__ = ["ExplicitEuler", "explEuler", "ExplicitRungeKutta", "exRungeKutta"]
