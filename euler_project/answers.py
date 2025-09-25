"""Canonical answers content for the assignment sheet.

This module centralizes the text shown in answers.txt and appended to the
per-run results PDF. Keeping it here ensures runs are reproducible and the
answers are consistent across machines.
"""

ANSWERS = {
    "b": r"""
\textbf{(b) Long-term behaviour as a function of $q$.}

We solve $x'(t) = qx - x^3$ with $x(0)=2$ and $q>0$. The equilibria are
$0$ and $\pm\sqrt{q}$. For $q>0$, $x=0$ is unstable and $x=\pm\sqrt{q}$ are
asymptotically stable since $f'(x)=q-3x^2$ implies $f'(\pm\sqrt{q})=-2q<0$.

With $x(0)=2>0$, the trajectory approaches the stable equilibrium $+\sqrt{q}$:
\[
\begin{cases}
q<4\!: & \sqrt{q}<2 \Rightarrow x(t)\ \text{decreases monotonically to}\ \sqrt{q},\\[2pt]
q=4\!: & x(t)\equiv 2\ \text{(equilibrium; adding this case gives a flat line at 2)},\\[2pt]
q>4\!: & \sqrt{q}>2 \Rightarrow x(t)\ \text{increases monotonically to}\ \sqrt{q}.
\end{cases}
\]
This matches the parameter-sweep plot: for small $q$ the approach to $\sqrt{q}$ is slower,
so at $T=10$ the solution can still be slightly above the limiting value.
""",

    "c": r"""
\textbf{(c) Method comparison (Euler vs.\ LSODA) and effect of $q$.}

We compare explicit Euler with step sizes $\tau=0.1$ and $\tau=0.01$ against an
LSODA reference on $[0,10]$.

\emph{Accuracy order.} Explicit Euler is first order: the global error scales
as $\mathcal{O}(\tau)$ for smooth problems on a fixed time horizon. Hence,
reducing $\tau$ from $0.1$ to $0.01$ should reduce the error by about a factor
of $10$ (modulo transients).

\emph{Linear stability near the attractor.} Linearizing at the stable equilibrium
$x^*=\sqrt{q}$ gives $y' = f'(x^*)\,y = -2q\,y$. For the test equation
$y'=\lambda y$ with $\lambda=-2q$, explicit Euler is stable iff
\[
|1+\tau\lambda|<1 \quad\Longleftrightarrow\quad 0<\tau<\frac{1}{q}.
\]

\emph{Case $q=10$.} The stability bound is $\tau<0.1$, so $\tau=0.1$ lies
\emph{on the boundary} and yields visible phase/amplitude error and mild
oscillation around the equilibrium; $\tau=0.01$ is well inside the stable
region and closely tracks LSODA. Empirically, the absolute error curve for
$\tau=0.1$ sits roughly an order of magnitude above that for $\tau=0.01$ over
most of $[0,10]$, consistent with first-order convergence \emph{and} the
stability-edge effect at $\tau=0.1$.

\emph{Case $q=0.1$.} The bound is $\tau<10$, so both $\tau=0.1$ and $0.01$
are deep inside the stability region and the dynamics are slow. Both Euler
solutions lie very close to LSODA; the $\tau=0.01$ error is still smaller
(by about the expected $\sim 10\times$ factor), but the difference is barely
visible in the solution plot because all errors are small.
""",

    "d": r"""
\textbf{(d) Sensitivity for the Lorenz system.}

With standard parameters $(a,b,c)=(10,25,8/3)$ the Lorenz system exhibits
sensitive dependence on initial conditions (positive largest Lyapunov exponent).
We integrate on $[0,10]$ with explicit Euler ($\tau=0.001$) from
$(x_1(0),x_2(0),x_3(0))=(10,5,12)$ and from the perturbed
$(10,5.01,12)$.

The two trajectories coincide initially but separate clearly after a short time,
ultimately exploring different parts of the attractor. This is the expected
behaviour for a chaotic system: for a small perturbation $\|\delta x(0)\|$ the
separation typically grows like $\|\delta x(t)\|\approx $ $\|\delta x(0)\|\,e^{\lambda t}$
with $\lambda>0$.

\emph{Conclusion.} Yes, the solution changes significantly when $x_2(0)$ is
perturbed to $5.01$; the 3D plot makes this divergence clearly visible.
""",
}


def answers_as_latex() -> str:
    """Concatenate sections (b)–(d) as LaTeX for convenience."""
    return (
        r"\section*{Answer (b)}" + "\n" + ANSWERS["b"]
        + r"\section*{Answer (c)}" + "\n" + ANSWERS["c"]
        + r"\section*{Answer (d)}" + "\n" + ANSWERS["d"]
    )

