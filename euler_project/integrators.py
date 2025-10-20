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


class EmbeddedRungeKuttaAdaptive:
    """Adaptive explicit embedded Runge–Kutta method.

    Uses a single strictly lower-triangular tableau ``A`` with two weight
    vectors ``b_high`` and ``b_low`` sharing the same nodes ``c``. On each
    step, both solutions are formed; their difference provides an error
    estimate which controls the step size.

    The right-hand side may be autonomous ``f(x)`` or non-autonomous
    ``f(t, x)``; the arity is detected automatically.

    Parameters
    ----------
    f
        Right-hand side function.
    x0
        Initial value (1-D array-like).
    T
        Final time ``T > 0``.
    tau_max
        Maximal step size.
    rho
        Safety factor in ``(0, 1)``.
    q
        Step-size growth factor ``> 1``.
    tol
        Absolute tolerance for the local error estimate.
    A, b_high, b_low, c
        Embedded tableau components. ``b_high`` is the higher-order rule used
        for the accepted solution; ``b_low`` is the embedded lower-order rule
        used for the error estimate.
    p_error
        Exponent base for the step-size controller. Typically ``p_error`` is
        the order of the lower rule (e.g. 2 for Bogacki–Shampine 3(2)), so the
        exponent ``1 / (p_error + 1)`` is used.
    """

    def __init__(
        self,
        f: Callable,
        x0: Iterable[float],
        T: float,
        tau_max: float,
        rho: float,
        q: float,
        tol: float,
        A: Iterable[Iterable[float]],
        b_high: Iterable[float],
        b_low: Iterable[float],
        c: Iterable[float],
        p_error: int,
    ) -> None:
        if not (0.0 < rho < 1.0):
            raise ValueError("rho must satisfy 0 < rho < 1")
        if not (q > 1.0):
            raise ValueError("q must be > 1")
        if not (T > 0.0 and tau_max > 0.0 and tol > 0.0):
            raise ValueError("T, tau_max, tol must be positive")
        if p_error < 1:
            raise ValueError("p_error must be >= 1")

        self.f = f
        self.x0 = np.atleast_1d(np.array(x0, dtype=float))
        if self.x0.ndim != 1:
            raise ValueError("x0 must be 1-D")
        self.T = float(T)
        self.tau_max = float(tau_max)
        self.rho = float(rho)
        self.q = float(q)
        self.tol = float(tol)

        b_high = np.asarray(b_high, dtype=float).reshape(-1)
        b_low = np.asarray(b_low, dtype=float).reshape(-1)
        c = np.asarray(c, dtype=float).reshape(-1)
        if not (b_high.size == b_low.size == c.size):
            raise ValueError("b_high, b_low, c must have the same length")
        s = int(c.size)

        A = np.asarray(A, dtype=float)
        if A.shape != (s, s):
            raise ValueError(f"A must have shape {(s, s)}")
        if not np.allclose(A, np.tril(A, k=-1), atol=1e-14, rtol=0.0):
            raise ValueError("A must be strictly lower triangular for explicit RK")

        self.A = A
        self.b_high = b_high
        self.b_low = b_low
        self.c = c
        self.s = s
        self.p_error = int(p_error)

        import inspect as _inspect

        self._arity_is_unary = len(_inspect.signature(f).parameters) == 1

    def _rk_step(self, t: float, x: Array, h: float) -> tuple[Array, Array, list[Array]]:
        """Perform one embedded step of size ``h``.

        Returns ``(x_high, x_low, ks)`` where ``ks`` is the list of stage
        derivatives.
        """

        d = x.size
        k_list: list[Array] = []
        for i in range(self.s):
            incr = np.zeros(d, dtype=float)
            if i > 0:
                for j in range(i):
                    aij = self.A[i, j]
                    if aij != 0.0:
                        incr += aij * k_list[j]
            x_stage = x + h * incr
            t_stage = t + self.c[i] * h
            if self._arity_is_unary:
                k_i = np.asarray(self.f(x_stage), dtype=float).reshape(-1)
            else:
                k_i = np.asarray(self.f(t_stage, x_stage), dtype=float).reshape(-1)
            if k_i.shape != (d,):
                raise ValueError(f"f returned shape {k_i.shape}, expected {(d,)}")
            k_list.append(k_i)

        inc_high = np.zeros(d, dtype=float)
        inc_low = np.zeros(d, dtype=float)
        for i in range(self.s):
            bi = self.b_high[i]
            if bi != 0.0:
                inc_high += bi * k_list[i]
            bl = self.b_low[i]
            if bl != 0.0:
                inc_low += bl * k_list[i]
        x_high = x + h * inc_high
        x_low = x + h * inc_low
        return x_high, x_low, k_list

    def run(self) -> tuple[Array, Array]:
        """Run the adaptive integrator and return ``(t_grid, X)``.

        The returned grid includes the initial time ``0`` and the final time
        ``T``. The state at each accepted step is the higher-order solution.
        """

        t_list: list[float] = [0.0]
        X_list: list[Array] = [self.x0.copy()]

        t = 0.0
        x = self.x0.copy()
        h = min(self.tau_max, self.T)
        pe = float(self.p_error + 1)  # exponent denominator

        tiny = 1e-16

        while t < self.T - 1e-15:
            # Avoid overshoot
            h = min(h, self.T - t)

            x_high, x_low, _ = self._rk_step(t, x, h)
            err = float(np.linalg.norm(x_high - x_low, ord=np.inf))

            # Suggest next step
            if err <= tiny:
                # error estimate vanished – grow conservatively
                h_suggest = min(self.q * h, self.tau_max)
            else:
                fac = (self.tol / err) ** (1.0 / pe)
                h_suggest = min(self.q * h, self.tau_max, max(tiny, (self.rho ** (1.0 / pe)) * fac * h))

            if err <= self.tol:
                # accept step
                t = t + h
                x = x_high
                t_list.append(t)
                X_list.append(x.copy())
                h = min(h_suggest, self.T - t)
            else:
                # reject: retry with smaller step
                h = max(tiny, h_suggest)

            # Guard against stagnation
            if h < 1e-14:
                # Prevent an endless loop in degenerate cases
                h = min(self.tau_max, self.T - t)
                if h <= 0.0:
                    break

        return np.array(t_list, dtype=float), np.vstack(X_list)


def adaptive_embedded_rk(
    f: Callable,
    x0: Iterable[float],
    T: float,
    tau_max: float,
    rho: float,
    q: float,
    tol: float,
    A: Iterable[Iterable[float]],
    b_high: Iterable[float],
    b_low: Iterable[float],
    c: Iterable[float],
    p_error: int,
) -> Tuple[Array, Array]:
    """Convenience wrapper for :class:`EmbeddedRungeKuttaAdaptive`."""

    return EmbeddedRungeKuttaAdaptive(
        f,
        x0,
        T,
        tau_max,
        rho,
        q,
        tol,
        A,
        b_high,
        b_low,
        c,
        p_error,
    ).run()


__all__ = [
    "ExplicitEuler",
    "explEuler",
    "ExplicitRungeKutta",
    "exRungeKutta",
    "EmbeddedRungeKuttaAdaptive",
    "adaptive_embedded_rk",
]
