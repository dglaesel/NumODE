"""Canonical answers content for the assignment sheet.

This module centralizes the text shown in answers.txt and appended to the
per-run results PDF. Keeping it here ensures runs are reproducible and the
answers are consistent across machines.
"""

ANSWERS = {
    "b": "(b) Answer intentionally left blank.",
    "c": "(c) Answer intentionally left blank.",
    "d": "(d) Answer intentionally left blank.",
}


def answers_as_latex() -> str:
    """Concatenate sections (b)-(d) as LaTeX for convenience."""
    return (
        r"\section*{Answer (b)}" + "\n" + ANSWERS["b"]
        + r"\section*{Answer (c)}" + "\n" + ANSWERS["c"]
        + r"\section*{Answer (d)}" + "\n" + ANSWERS["d"]
    )

