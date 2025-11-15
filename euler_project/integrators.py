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
from scipy import optimize as _opt

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


class ImplicitEuler:
    """Implicit Euler for general (possibly nonlinear) IVPs.

    Solves the step equation x_{n+1} = x_n + h f(t_{n+1}, x_{n+1}) using
    SciPy's nonlinear solvers. Works for autonomous ``f(x)`` and
    non-autonomous ``f(t, x)``; the arity is detected automatically.
    """

    def __init__(
        self,
        f: Callable,
        x0: Iterable[float],
        T: float,
        tau: float,
        *,
        solver: str = "root",
    ) -> None:
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
        self._solver = solver

    def _rhs(self, t: float, x: Array) -> Array:
        if self._arity_is_unary:
            val = self.f(x)
        else:
            val = self.f(t, x)
        return np.asarray(val, dtype=float).reshape(-1)

    def _solve_step(self, t_np1: float, x_n: Array, h: float) -> Array:
        d = x_n.size
        # explicit Euler predictor as initial guess
        f_n = self._rhs(t_np1 - h, x_n)
        x_guess = x_n + h * f_n

        def G(y: Array) -> Array:
            y = np.asarray(y, dtype=float).reshape(d)
            return y - x_n - h * self._rhs(t_np1, y)

        if self._solver == "fsolve":
            sol = _opt.fsolve(G, x_guess, full_output=True)
            y = np.asarray(sol[0], dtype=float).reshape(d)
            ier = sol[2]
            if ier != 1:
                raise RuntimeError(f"fsolve failed (ier={ier})")
            return y
        else:
            res = _opt.root(G, x_guess, method="hybr")
            if not res.success:
                raise RuntimeError(f"root() failed at t={t_np1:.6g}: {res.message}")
            return np.asarray(res.x, dtype=float).reshape(d)

    def run(self) -> Tuple[Array, Array]:
        d = self.x0.size
        n_steps = int(np.floor(self.T / self.tau))
        if n_steps < 1:
            raise ValueError("integration horizon too short")
        t = self.tau * np.arange(n_steps + 1, dtype=float)
        X = np.empty((n_steps + 1, d), dtype=float)
        X[0] = self.x0
        for k in range(n_steps):
            t_np1 = t[k + 1]
            X[k + 1] = self._solve_step(t_np1, X[k], self.tau)
        return t, X


def implicitEuler(
    f: Callable,
    x0: Iterable[float],
    T: float,
    tau: float,
    *,
    solver: str = "root",
) -> Tuple[Array, Array]:
    """Wrapper for :class:`ImplicitEuler` (exercise 4 API)."""

    return ImplicitEuler(f, x0, T, tau, solver=solver).run()


class ImplicitEulerLinear:
    """Implicit Euler specialized for linear/affine RHS.

    Handles problems of the form ``x' = A(t) x + b(t)`` where ``A`` and ``b``
    may be functions of time or constants. Each step solves

        (I - h A(t_{n+1})) x_{n+1} = x_n + h b(t_{n+1})

    via a direct linear solve. This avoids nonlinear iterations.
    """

    def __init__(
        self,
        A: np.ndarray | Callable[[float], np.ndarray],
        b: np.ndarray | Callable[[float], np.ndarray] | None,
        x0: Iterable[float],
        T: float,
        tau: float,
    ) -> None:
        if tau <= 0:
            raise ValueError("tau > 0 required")
        if T <= 0:
            raise ValueError("T > 0 required")
        x = np.atleast_1d(np.array(x0, dtype=float))
        if x.ndim != 1:
            raise ValueError("x0 must be 1-D")
        self.x0 = x
        self.T = float(T)
        self.tau = float(tau)
        self._A = A
        self._b = b

    def _A_of(self, t: float) -> np.ndarray:
        A = self._A(t) if callable(self._A) else self._A
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square (dxd)")
        return A

    def _b_of(self, t: float, d: int) -> np.ndarray:
        if self._b is None:
            return np.zeros(d, dtype=float)
        b = self._b(t) if callable(self._b) else self._b
        b = np.asarray(b, dtype=float).reshape(-1)
        if b.size != d:
            raise ValueError("b must have size d")
        return b

    def run(self) -> Tuple[Array, Array]:
        d = self.x0.size
        n_steps = int(np.floor(self.T / self.tau))
        if n_steps < 1:
            raise ValueError("integration horizon too short")
        t = self.tau * np.arange(n_steps + 1, dtype=float)
        X = np.empty((n_steps + 1, d), dtype=float)
        X[0] = self.x0
        I = np.eye(d, dtype=float)
        for k in range(n_steps):
            t_np1 = t[k + 1]
            A = self._A_of(t_np1)
            b = self._b_of(t_np1, d)
            M = I - self.tau * A
            rhs = X[k] + self.tau * b
            X[k + 1] = np.linalg.solve(M, rhs)
        return t, X


def implicitEuler_linear(
    A: np.ndarray | Callable[[float], np.ndarray],
    b: np.ndarray | Callable[[float], np.ndarray] | None,
    x0: Iterable[float],
    T: float,
    tau: float,
) -> Tuple[Array, Array]:
    """Wrapper for :class:`ImplicitEulerLinear`.

    Examples
    --------
    Linear test equation y' = lambda y:
    >>> import numpy as np
    >>> lam = -3.0
    >>> A = np.array([[lam]])
    >>> t, X = implicitEuler_linear(A, None, [1.0], 1.0, 0.1)
    """
    return ImplicitEulerLinear(A, b, x0, T, tau).run()

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


class ImplicitRungeKutta:
    """Generic implicit Runge–Kutta (IRK) method.

    Solves the stage equations

        k_i = f(t_n + c_i h, x_n + h * sum_j a_{ij} k_j),   i = 1..s

    for the unknown stage derivatives ``k_i`` using a nonlinear solver, and
    then updates

        x_{n+1} = x_n + h * sum_i b_i k_i.

    The right-hand side ``f`` may be autonomous ``f(x)`` or non-autonomous
    ``f(t, x)``; the arity is detected automatically.
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
        *,
        solver: str = "root",
    ) -> None:
        if tau <= 0:
            raise ValueError("tau > 0 required")
        if T <= 0:
            raise ValueError("T > 0 required")

        self.f = f
        self.x0 = np.atleast_1d(np.array(x0, dtype=float))
        if self.x0.ndim != 1:
            raise ValueError("x0 must be 1-D")
        self.T = float(T)
        self.tau = float(tau)

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        c = np.asarray(c, dtype=float).reshape(-1)
        if not (b.size == c.size == A.shape[0] == A.shape[1]):
            raise ValueError("A must be (s,s) and b,c length s")
        self.A = A
        self.b = b
        self.c = c
        self.s = int(c.size)

        self._arity_is_unary = len(inspect.signature(f).parameters) == 1
        self._solver = solver

    def _rhs(self, t: float, x: Array) -> Array:
        if self._arity_is_unary:
            val = self.f(x)
        else:
            val = self.f(t, x)
        return np.asarray(val, dtype=float).reshape(-1)

    def _residual(self, Z: Array, t_n: float, x_n: Array, h: float) -> Array:
        """Residual for stage states ``Z_i`` stacked into a vector.

        We solve for stage states ``Z_i`` satisfying

            Z = x_n + h * A * K,   with K_i = f(t_n + c_i h, Z_i).
        """

        d = x_n.size
        Z = np.asarray(Z, dtype=float).reshape(self.s, d)
        # Evaluate K_i from Z_i
        K = np.empty_like(Z)
        for i in range(self.s):
            t_stage = t_n + self.c[i] * h
            K[i] = self._rhs(t_stage, Z[i])
        AZ = self.A @ K  # shape (s, d)
        R = Z - (x_n + h * AZ)
        return R.reshape(-1)

    def _solve_stages(self, t_n: float, x_n: Array, h: float) -> Array:
        d = x_n.size
        # Initial guess for Z: explicit Euler predictor for the state
        x_pred = x_n + h * self._rhs(t_n, x_n)
        Z0 = np.tile(x_pred, self.s)

        G = lambda Zvec: self._residual(Zvec, t_n, x_n, h)

        if self._solver == "fsolve":
            sol = _opt.fsolve(G, Z0, full_output=True)
            Zvec = np.asarray(sol[0], dtype=float)
            ier = sol[2]
            if ier != 1:
                raise RuntimeError(f"fsolve failed (ier={ier}) at t={t_n:.6g}")
            Z = Zvec.reshape(self.s, d)
        else:
            res = _opt.root(G, Z0, method="hybr")
            if not res.success:
                raise RuntimeError(f"root() failed at t={t_n:.6g}: {res.message}")
            Z = np.asarray(res.x, dtype=float).reshape(self.s, d)

        # Recover K from Z
        K = np.empty_like(Z)
        for i in range(self.s):
            t_stage = t_n + self.c[i] * h
            K[i] = self._rhs(t_stage, Z[i])
        return K

    def run(self) -> Tuple[Array, Array]:
        d = self.x0.size
        n_steps = int(np.floor(self.T / self.tau))
        if n_steps < 1:
            raise ValueError("integration horizon too short")
        t = self.tau * np.arange(n_steps + 1, dtype=float)
        X = np.empty((n_steps + 1, d), dtype=float)
        X[0] = self.x0

        for n in range(n_steps):
            t_n = t[n]
            K = self._solve_stages(t_n, X[n], self.tau)
            incr = np.zeros(d, dtype=float)
            for i in range(self.s):
                bi = self.b[i]
                if bi != 0.0:
                    incr += bi * K[i]
            X[n + 1] = X[n] + self.tau * incr
        return t, X


def implicitRungeKutta(
    f: Callable,
    x0: Iterable[float],
    T: float,
    tau: float,
    A: Iterable[Iterable[float]],
    b: Iterable[float],
    c: Iterable[float],
    *,
    solver: str = "root",
) -> Tuple[Array, Array]:
    """Convenience wrapper for :class:`ImplicitRungeKutta`."""

    return ImplicitRungeKutta(f, x0, T, tau, A, b, c, solver=solver).run()


# Short alias matching the exercise wording
implicitRK = implicitRungeKutta


__all__ = [
    "ExplicitEuler",
    "explEuler",
    "ExplicitRungeKutta",
    "exRungeKutta",
    "EmbeddedRungeKuttaAdaptive",
    "adaptive_embedded_rk",
    "ImplicitRungeKutta",
    "implicitRungeKutta",
    "implicitRK",
]
