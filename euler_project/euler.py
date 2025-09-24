"""Compatibility wrappers for the explicit Euler integrator."""

from __future__ import annotations  # postpone evaluation of type hints (forward refs)

from .integrators import ExplicitEuler, explEuler

__all__ = ["ExplicitEuler", "explEuler"]
