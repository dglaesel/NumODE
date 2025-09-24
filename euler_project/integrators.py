"""Explicit Euler integrator class and function wrapper."""

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


__all__ = ["ExplicitEuler", "explEuler"]
