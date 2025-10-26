"""Public exports for the ODE project (exercises 1 and 2)."""

from __future__ import annotations  # postpone evaluation of type hints (forward refs)

from .integrators import (
    ExplicitEuler,
    explEuler,
    ImplicitEuler,
    implicitEuler,
    ImplicitEulerLinear,
    implicitEuler_linear,
    ExplicitRungeKutta,
    exRungeKutta,
    EmbeddedRungeKuttaAdaptive,
    adaptive_embedded_rk,
)
from .problems import (
    rhs_cubic,
    rhs_lorenz,
    rhs_logistic,
    logistic_analytic,
    rhs_forced_lorenz,
    rhs_cos2_arctan_problem,
    arctan_analytic,
)

__all__ = [
    "ExplicitEuler",
    "explEuler",
    "ImplicitEuler",
    "implicitEuler",
    "ImplicitEulerLinear",
    "implicitEuler_linear",
    "ExplicitRungeKutta",
    "exRungeKutta",
    "EmbeddedRungeKuttaAdaptive",
    "adaptive_embedded_rk",
    "rhs_cubic",
    "rhs_lorenz",
    "rhs_logistic",
    "logistic_analytic",
    "rhs_forced_lorenz",
    "rhs_cos2_arctan_problem",
    "arctan_analytic",
]
