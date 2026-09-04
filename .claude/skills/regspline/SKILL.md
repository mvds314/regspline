---
name: regspline
description: Guide to regspline, a 1-D regression spline library whose splines are plain callable objects - LinearSpline (hinge/MARS basis) and NaturalCubicSpline - fitted via the from_data classmethod using OLS, WLS, LASSO, quantile regression, SVR or NuSVR across statsmodels, sklearn and pyqreg backends. Use when working in the regspline repo, or when the user mentions regression splines, spline knots, knot pruning, hinge or MARS basis functions, natural cubic splines, spline quantile regression, piecewise-linear fits to noisy 1-D data, or extrapolation beyond the knot range. Also use before extending regspline with a new estimation method, spline type, or extrapolation mode, since those must respect non-obvious invariants documented here.
---

# regspline

`regspline` fits 1-D regression splines. Its distinguishing choice is that a spline
is *just a callable object* — not a fitted model wrapper, not a sklearn estimator.
`s(x)` evaluates it, `s.eval_basis(x)` gives you the design matrix, and that's the
whole surface. Estimation lives in one classmethod, `from_data`, which is a
convenience on top of statsmodels/sklearn rather than a framework of its own.

Keep that separation in mind when helping users: if someone wants pipelines,
cross-validation, or multi-dimensional inputs, regspline is deliberately not that,
and the honest answer is to point at py-earth or `sklearn.preprocessing.SplineTransformer`
rather than to bolt a framework onto this one.

## The central design idea

Both spline families are chosen so that **each coefficient corresponds one-to-one
with a knot**. That is the whole point of the library, and nearly every design
decision follows from it.

For the linear spline the basis is hinge functions `h_i(x) = max(x - k_i, 0)`.
Read the fit as a sequence of slope adjustments: the constant sets the level at the
first knot, `h_1` sets the initial slope, and each later `h_i` *adjusts* the slope
at knot `k_i`. If the true function has no kink at `k_i`, that adjustment is zero,
its coefficient comes out statistically insignificant, and the knot can be deleted
**without disturbing any other basis function**.

This is what makes the regression sparse and the pruning meaningful. A B-spline
basis does not have this property — deleting one B-spline knot rebuilds the entire
basis. So when a user asks "why not just use B-splines", the answer is knot pruning
and interpretability, not numerical superiority.

## Choosing a spline

| | `LinearSpline` | `NaturalCubicSpline` |
|---|---|---|
| Basis | hinge `max(x-k_i, 0)` | natural cubic (Hastie et al., ch. 5.3) |
| Shape | piecewise linear, kinks at knots | smooth (C²) |
| Min knots | 2 | 3 |
| Basis count | `n_knots - 1` | `n_knots - 1` |
| Good for | kinked/regime-like relationships, fast eval, MARS-style interpretation | smooth underlying signal, controlled tail behaviour |

Natural cubic splines are constrained to be **linear beyond the boundary knots**,
which is exactly why you'd choose them for extrapolation-sensitive work — cubic tails
would fly off. Linear splines extrapolate with whatever slope the last segment had.

## Constructing a spline directly

```python
from regspline import LinearSpline, NaturalCubicSpline

s = LinearSpline([0, 0.5, 1], [0, 1, 1])   # 3 knots, 3 coeffs -> first is the constant
s(0.75)                                     # scalar in, scalar out
s(np.linspace(0, 1, 50))                    # array in, array out
s(pd.Series(x))                             # Series in, Series out (index preserved)
```

**The constant is implicit in the coefficient count.** This trips people up
constantly, so state it plainly: if `len(coeffs) == len(knots)`, `coeffs[0]` is the
constant. If `len(coeffs) == len(knots) - 1`, there is no constant. The property
`has_const` just checks `n_knots == n_coeffs`. There is no separate intercept field
and no flag to set — the shape *is* the flag. Anything else is rejected by validation.

Other conveniences: `len(s)` is the knot count, `s.copy()`, and `==`/`hash()` compare
knots, coeffs, and extrapolation method (so splines work as dict keys and in caches).
Hashing works before a fit, while `coeffs` is still `None`.

**Knots must be strictly increasing.** Coincident knots are rejected by the setter,
along with gaps negligible against the knot span. Two knots at the same place give two
identical basis functions and a singular design matrix, which statsmodels solves via
pseudo-inverse rather than refusing, so the fit returns a plausible-looking answer
with the slope split arbitrarily across the pair. The check compares gaps to the span
rather than to the knot values, so knots far from zero — epoch timestamps, price
levels — are fine. Build candidate knots from quantiles of skewed data with
`np.unique(np.quantile(x, ...))`, since ties are common there.

## Extrapolation modes

Set via `extrapolation_method=` at construction, or assign to the property later.
Default is `"nan"`.

- `"nan"` — outside `[k_1, k_N]` returns NaN. The safe default: it makes
  out-of-range use *loud* instead of silently plausible.
- `"const"` — clamps to the boundary values by clipping `x` before evaluation.
- `"basis"` — just keeps evaluating the basis functions. For a natural cubic spline
  this is already linear in the tails by construction.
- `"linear"` — only meaningfully different for `LinearSpline`, where it swaps the
  first hinge for `x - k_1` so the fit is linear left of the first knot too.

Choosing a mode is a modelling decision, not a formatting one, so when a user is
extrapolating far beyond their data, say so rather than quietly picking `"basis"`.

## Fitting from data

`from_data` is a classmethod on both spline types:

```python
spline = LinearSpline.from_data(x, y, knots=20, method="OLS", prune=True)
spline, result = LinearSpline.from_data(..., return_estim_result=True)
```

Key arguments:

- `knots` — an array of knots, an `int` (that many, equally spaced over the data
  range), or `None` for 10 equally spaced. Equally spaced is a lazy default; quantile
  spacing (`np.quantile(x, np.linspace(0, 1, k))`) usually fits better when `x` is
  skewed, so suggest it when the data is clearly non-uniform.
- `add_constant` (default `True`) — controls whether a constant column enters the design.
- `prune` (default `False`) — drop uninformative knots, see below.
- `return_estim_result` — also return the underlying statsmodels/sklearn result
  object, which is where standard errors, t-values and diagnostics live.
- `backend` — force `"statsmodels"`, `"sklearn"`, or `"pyqreg"`.
- Everything else is forwarded to the underlying fit call.

### Methods and backends

| `method` | Backends | Needs | Notable kwargs |
|---|---|---|---|
| `"OLS"` | statsmodels | — | — |
| `"WLS"` | statsmodels | — | `weights` (default `1.0`) |
| `"LASSO"` | statsmodels (`sqrt_lasso`) | `cvxopt` | `alpha` |
| `"QuantileRegression"` | statsmodels (default), `sklearn`, `pyqreg` | sklearn/pyqreg optional | `q` (default 0.5), `max_iter` |
| `"SVR"` | sklearn (`LinearSVR`) | `scikit-learn` | `C`, `epsilon` |
| `"NuSVR"` | sklearn (`NuSVR`) | `scikit-learn` | `C`, `nu` |

Backend guidance worth volunteering: for quantile regression, statsmodels is the
default and returns t-values (so it supports significance-based pruning), `pyqreg`
is markedly faster on large samples and also gives t-values, and `sklearn` uses the
HiGHS LP solver but gives no inference. `examples/example_quantile_regression.py`
times all three side by side.

With `add_constant=False`, the sklearn `QuantileRegression` and `SVR` branches return
`n_knots - 1` coefficients and `has_const` is `False`, matching the statsmodels
branches. `NuSVR` always estimates an intercept and asserts `add_constant`.

### Finding a few knots from many candidates

When the task is "find the kinks", "start with many possible knots but end with a
handful", or "print the knot locations", reach for `prune=True`. That is the sparse
workflow this basis was built for: over-specify candidate knots, let the regression
identify unneeded slope changes, then read the retained locations from `spline.knots`.

```python
from regspline import LinearSpline
import numpy as np

# Quantile spacing gives candidates across the data density when x is bunched.
# Keep the candidate set modest: see the note below on why more is not better.
candidate_knots = np.unique(np.quantile(x, np.linspace(0, 1, 20)))

spline = LinearSpline.from_data(
    x,
    y,
    knots=candidate_knots,
    method="OLS",
    prune=True,
)
spline.extrapolation_method = "basis"  # choose this deliberately before evaluating at 45

print(spline.knots)      # retained knots; interior ones are the kink locations
print(spline(45))        # deliberate extrapolation beyond the data range
```

For OLS and statsmodels/pyqreg quantile regression, `prune=True` already performs the
significance-based elimination agents are tempted to hand-roll: fit all candidates,
drop coefficients with `|t| < 1.96`, then **refit from scratch** on the surviving
knots. The refit matters because coefficients change after pruning, and the built-in
path also handles the constant-vs-knot index alignment that is easy to get wrong.

For LASSO, SVR, NuSVR, and sklearn quantile regression there are no t-values, so
`prune=True` calls `prune_knots()` and drops coefficients within `tol` of zero. That
makes LASSO + pruning the idiomatic sparse workflow when you want regularization:
start with many knots, let the penalty zero most of them, then prune to get a compact
spline.

Manual pruning is still available for custom criteria:
`s.prune_knots(method="isclose", tol=1e-6)` or
`s.prune_knots(method="coeffs", coeffs_to_prune=[False, True, ...])`.

#### Do not over-specify the candidate set

"Over-specify then prune" has a ceiling. Adjacent hinge basis functions are highly
collinear, so packing in more candidate knots inflates the standard errors of all
of them, drives the t-values down, and makes significance pruning discard the very
kinks you are trying to find. More candidates give a *worse* answer, not a slower
one.

Measured on 8 synthetic datasets with known kinks at 3.5, 9.0 and 22.0 (n=5000, x
skewed toward the low end, noise sd 2.0), scoring the mean distance from each true
kink to the nearest retained knot:

| candidates | 10 | 15 | 20 | 25 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| quantile-spaced | 1.28 | **0.43** | 0.92 | 1.70 | 3.55 | 4.04 | 4.83 |
| uniform | 1.06 | 1.03 | **0.98** | 2.30 | 1.90 | 2.72 | 4.77 |

Start with roughly 15-25 candidates for a handful of expected kinks. If you need
many more knots than that, prefer LASSO with a penalty over t-value pruning, since
the penalty handles collinear candidates gracefully where individual t-tests do
not.

## The design matrix

```python
X = s.eval_basis(x, include_constant=True)
```

Returns an array, or a DataFrame with columns `const, b0, b1, ...` if `x` was pandas.
This is the escape hatch: any estimator that accepts a design matrix can fit a
regspline spline. If a user wants ridge, mixed effects, Bayesian sampling, or
anything else not in the `method` table, hand them `eval_basis` output rather than
trying to widen `from_data`.

Assigning the resulting parameter vector back to `s.coeffs` gives a working spline —
just respect the constant convention above.

## Optional dependencies

Probed once at import into module-level flags in `base.py`: `_has_sklearn`,
`_has_cvxopt`, `_has_pyqreg`. Install with extras: `.[SVR]`, `.[LASSO]`, `.[FASTQR]`.

`cvxopt == 1.3.0` is deliberately rejected (a domain-error bug makes SVR misbehave),
so a user on exactly that version gets the "missing dependency" path even though the
package is installed. Check the version before concluding it isn't installed.

## Repo commands

```bash
pip install -e .
pytest
pytest tests/test_linear_spline.py::test_from_data
pytest -m "not tofix"
ruff format --check     # CI enforces this
```

Tests guard optional paths with `if _has_sklearn:` rather than skip markers, so a
passing local run does not mean every branch was exercised. CI installs only
`.[SVR,LASSO]`, leaving the pyqreg branch untested.

## Extending the library

Read `references/internals.md` before changing anything under `regspline/` — adding
an estimation method, adding a spline type, or touching extrapolation. It documents
the class contract, the state invariants the setters enforce, the basis cache, and a
full worked example of adding a weighted-least-squares method. Those invariants are
enforced by asserts that fail in confusing ways if you don't know they exist.

For plain usage questions, this file is enough.
