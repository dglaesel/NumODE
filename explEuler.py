"""Convenience access to the explEuler function."""

from __future__ import annotations

from euler_project.integrators import explEuler

if __name__ == "__main__":
    import numpy as np

    f = lambda x: -x
    t, X = explEuler(f, 1.0, 1.0, 0.1)
    print("explEuler OK:", X.shape, "final≈", X[-1, 0])
