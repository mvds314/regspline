# regspline internals

Read this before modifying anything under `regspline/`. The library is small, but it
enforces its invariants with `assert`s inside property setters, which means a violation
surfaces as an assertion error somewhere far from the line that caused it. Knowing the
rules up front saves a lot of confused debugging.

## Contents

- [Module map](#module-map)
- [Class contract](#class-contract)
- [Invariant 1: the constant lives in the coefficient count](#invariant-1-the-constant-lives-in-the-coefficient-count)
- [Invariant 2: knots and coeffs cross-validate](#invariant-2-knots-and-coeffs-cross-validate)
- [Invariant 3: the basis cache must be invalidated](#invariant-3-the-basis-cache-must-be-invalidated)
- [Invariant 4: pandas in, pandas out](#invariant-4-pandas-in-pandas-out)
- [How pruning is implemented](#how-pruning-is-implemented)
- [Worked example: adding a WLS method](#worked-example-adding-a-wls-method)
- [Adding a new spline type](#adding-a-new-spline-type)
- [Adding an extrapolation mode](#adding-an-extrapolation-mode)
- [Testing conventions](#testing-conventions)

## Module map

```
regspline/
├── base.py                  all abstraction + the entire from_data dispatch
├── linear_spline.py         HingeBasisFunction, LinearSpline
├── natural_cubic_spline.py  di, NaturalCubicSplineBasisFunction, NaturalCubicSpline
└── util.py                  Timer, type_wrapper (re-exports statsmodels' PandasWrapper)
```

`base.py` carries the weight; the two concrete spline modules are thin. If you find
yourself adding substantial logic to a concrete spline, check whether it belongs in
the base class instead — the odds are good that the other spline needs it too.

## Class contract

```
BasisFuncInterface (ABC)          one basis function; subclass implements _apply(x)
    └── __call__ handles masking x outside [xmin, xmax] to `val`

KnotsInterface (ABC)              knot storage + validation (sorted, >= 2)

RegressionSplineBase(KnotsInterface, ABC)
    s(x) = const + sum(c_i * b_i(x))
```

A concrete spline supplies exactly four things:

| Member | Purpose |
|---|---|
| `_validate_knots_coeffs(knots, coeffs)` | shape rules; called from *both* setters |
| `_bi` | the list of basis function objects (cached, see below) |
| `has_const` | almost always `self.n_knots == self.n_coeffs` |
| `prune_knots(...)` | which knots may be dropped, given how coeffs map to knots |

`prune_knots` is abstract-with-a-body in the base class: it raises
`NotImplementedError` explaining that the coeff↔knot relationship is spline-specific.
That is intentional — there is no safe generic implementation.

Note that basis functions can themselves hold knots: the natural cubic basis
inherits from both `KnotsInterface` and `BasisFuncInterface`, because `N_i(x)`
depends on the full knot vector (it references the last knot), not just its own.
Hinge functions only need their own `ref`, so they inherit from `BasisFuncInterface`
alone.

## Invariant 1: the constant lives in the coefficient count

There is no intercept attribute. `has_const` is derived:

```python
# N coeffs for N knots -> coeffs[0] is the constant
# N-1 coeffs for N knots -> no constant
has_const = n_knots == n_coeffs
```

`const`, `_ci` (the non-constant coefficients), the `from_data` branches, and the
pruning index arithmetic all branch on this. Two consequences that bite:

- Setting `s.const = value` on a spline that has no constant **prepends** an element
  to `coeffs`, changing its length. That's deliberate — it's the only way to add a
  constant under this representation.
- Any new estimation branch must produce a coefficient vector whose length matches
  the `add_constant` it was called with. sklearn branches do this by writing
  `np.append(result.intercept_, result.coef_)`; statsmodels branches get it for free
  because the constant is a column of the design matrix.

## Invariant 2: knots and coeffs cross-validate

Both setters call `_validate_knots_coeffs` against the *current* value of the other
attribute. So you cannot change both in one step — the intermediate state is invalid
and the assert fires. The established idiom is to null both out first:

```python
self.coeffs = None
self.knots = None
self.coeffs = new_coeffs
self.knots = new_knots
```

`prune_knots` does exactly this, wrapped in `try/except` that restores the original
knots and coeffs before re-raising. **Preserve that fail-safe.** A half-updated spline
is worse than an exception: it evaluates without complaint and returns wrong numbers.
If you add a method that mutates both attributes, copy the same pattern. Restore from
copies taken before any mutation, and slice from those copies too: the cubic
implementation once shortened `self.coeffs` early to drop an insignificant constant,
which flipped `has_const` midway through and desynchronised the masks built after it.

The knots setter additionally requires strictly increasing values. Coincident
neighbours duplicate a basis function and make the design matrix singular, which
surfaces as arbitrary-looking coefficients rather than an error. Judge closeness
against the knot *span*, never against the knot values: `np.isclose` defaults to a
relative tolerance, which would reject hourly knots on unix timestamps as
"coincident" because the gaps are small next to 1.7e9.

Equality and hashing must agree. `KnotsInterface.__eq__` compares the concrete type
and the knots, and the basis-function classes extend both `__eq__` and `__hash__`
with `_i`, since two basis functions over the same knots still differ by the knot
they are anchored to. Defining `__eq__` alone sets `__hash__ = None` and makes the
class unhashable.

## Invariant 3: the basis cache must be invalidated

`_bi` memoizes into `self._bi_cache`. Anything that changes what the basis functions
should be must delete it:

```python
if hasattr(self, "_bi_cache"):
    del self._bi_cache
```

Currently the `knots` setter and the `extrapolation_method` setter both do this.
Note that the `coeffs` setter does *not*, and correctly so — coefficients scale the
basis but don't define it. If you add any other state that `_bi` reads, its setter
joins the list. Missing this produces a spline that silently keeps evaluating a stale
basis, which is one of the nastier bugs to chase here.

`NaturalCubicSplineBasisFunction` has its own inner caches (`_dim1_cache`,
`_dNm1_cache`) for the helper `d_i` functions. Those objects are immutable in
practice — the spline throws the whole basis list away rather than mutating it — so
they need no invalidation, but don't start mutating basis objects in place.

## Invariant 4: pandas in, pandas out

`util.type_wrapper(xloc=...)` wraps a function so that argument `xloc` is coerced to
an array on the way in and the result is re-wrapped on the way out via statsmodels'
`PandasWrapper`: Series in → Series out with the index preserved, 0-d result →
Python scalar. `RegressionSplineBase.__call__` and `BasisFuncInterface.__call__` both
use it; `eval_basis` uses `PandasWrapper` directly because it also has to attach
column names.

Any new public method that takes `x` and returns values aligned with it should use
the same decorator. Users rely on being able to hand it a DataFrame column and get
something they can assign straight back.

## How pruning is implemented

The tricky part is index alignment, because the `to_prune` mask is indexed by
*coefficient* and the thing being cut is *knots*, and the two are offset by the
constant at the front and by the unused last knot at the back.

```python
to_prune  # length n_coeffs
self.coeffs = coeffs[~to_prune]

to_prune = np.append(
    to_prune[1:] if len(knots) == len(coeffs) else to_prune,  # drop const slot
    False,  # last knot has no coeff
)
self.knots = knots[~to_prune]
```

Read it as: strip the constant's entry if there is one, then append a `False` because
the final knot has no basis function and must always survive. `NaturalCubicSpline`
does the same with its own minimum-knot floor of 3.

Two supported selection methods: `"isclose"` (magnitude, uses `tol` as `atol`) and
`"coeffs"` (caller supplies an explicit boolean mask of length `n_coeffs`).

## Worked example: adding a WLS method

This is the canonical "add an estimation method" change. Every method is a branch in
`RegressionSplineBase.from_data` in `base.py`; there is no registry or plugin system,
and adding one shouldn't invent one.

Follow the shape of the existing `"OLS"` branch:

```python
elif method == "WLS":
    assert backend is None or backend == "statsmodels", "sklearn backend not implemented"
    weights = kwargs.pop("weights", 1.0)
    refit_settings["weights"] = weights   # <- popped, so record it for the refit
    smkwargs = dict(
        exog=spline.eval_basis(x, include_constant=add_constant),
        weights=weights,
        hasconst=True,
        missing=missing,
    )
    model = sm.WLS(y, **smkwargs)
    result = model.fit(**kwargs)
    spline.coeffs = result.params
    insignificant = np.abs(result.tvalues) < 1.96
    if prune and np.any(insignificant):
        add_constant = add_constant and not insignificant[0]
        spline.prune_knots(method="coeffs", coeffs_to_prune=insignificant)
        return refit(add_constant)
```

Points that generalise to any new method:

1. **Build the design with `spline.eval_basis(x, include_constant=add_constant)`.**
   Never construct basis columns by hand in the branch.
2. **Assign to `spline.coeffs`, don't return a bare parameter vector.** The setter is
   where validation happens.
3. **Recursive refit is the pruning pattern for t-value backends.** Prune, then
   re-enter `cls.from_data` with the surviving knots and `prune=False` to stop the
   recursion. The second fit is what makes the pruned coefficients correct. All four
   refit sites go through the nested `refit()` helper; don't hand-roll a fifth
   `cls.from_data` call.
4. **When you pop a kwarg only to pass it explicitly, use the downstream default.**
   `statsmodels.WLS` defaults `weights` to `1.0`, meaning unweighted. A more
   Python-looking sentinel such as `None` is not equivalent: it is forwarded to
   statsmodels, which passes it on to `np.sqrt` and fails before fitting.
5. **Anything you `pop` from `kwargs` must go into `refit_settings`**, or it silently
   vanishes on the refit and the pruned fit answers a different question than the one
   asked. This bug shipped three times before the helper existed: `missing="drop"`
   returned all-NaN coefficients, `extrapolation_method` reset itself to `"nan"`, and
   `q=0.10` came back as a median fit. Settings left *inside* `kwargs` need no action,
   which is why the statsmodels quantile branch can use
   `kwargs.setdefault("q", 0.5)` and forward nothing. Never record a key in both
   places: `refit()` splats `**refit_settings` and `**kwargs` together, so a
   duplicate raises `TypeError`.
6. **`weights` is per-observation and pruning removes columns, not rows**, so the
   weight vector stays aligned across the refit. No subsetting needed.
7. A default-path test is worth adding for every new estimation method. A test that
   always supplies `weights` will pass even if the no-weights path crashes; one call
   with no optional method kwargs is the cheapest guard.
8. If the backend has no t-values, use the magnitude path instead:
   `if prune: spline.prune_knots()`.
9. For sklearn-style backends, pack the coefficients with `_sklearn_coeffs(result,
   fit_intercept)` rather than `np.append(result.intercept_, result.coef_)`. sklearn
   exposes `intercept_` even when `fit_intercept=False`, where it is a hard `0.0`;
   prepending it unconditionally invents a constant and breaks the
   `n_knots`/`n_coeffs` invariant. Then assert
   `np.allclose(spline(x), result.predict(spline.eval_basis(x)))` as the existing
   branches do — a cheap guard that the packing was done right.

Then finish the job: add a row to the method table in `SKILL.md`, add a test beside
the existing ones in `tests/`, and mention it in `README.md` if it's user-facing.

## Adding a new spline type

Subclass `RegressionSplineBase` and implement the four contract members. The design
constraint to honour is the one the library exists for: **one coefficient per knot,
such that dropping a knot leaves the other basis functions unchanged.** A basis
without that property (B-splines, for instance) will produce a class where
`prune_knots` cannot be written correctly, which is a sign it doesn't belong here.

Checklist:

- minimum knot count asserted in `_validate_knots_coeffs`
- `_bi` respects every `extrapolation_method`, and caches into `_bi_cache`
- `has_const` consistent with the `n_knots`/`n_coeffs` rule
- `prune_knots` index arithmetic verified against both the with- and without-constant
  cases
- export from `regspline/__init__.py`
- tests mirroring `tests/test_linear_spline.py` (construction, extrapolation modes,
  pruning both ways, `from_data`, hash/equality)

## Adding an extrapolation mode

Add the name to the assert list in the `extrapolation_method` setter, then implement
it in `_bi` — not in `__call__`. `__call__` only special-cases `"const"`, because
clamping is a transformation of `x` rather than of the basis. Everything else is a
property of the basis functions: `"nan"` passes `xmin`/`xmax`/`val=np.nan` down to
them, and `"linear"` substitutes a different first basis function.

Implement the mode for *both* spline types, or make it explicit what happens on the
one that doesn't distinguish it. `"linear"` is currently a no-op on
`NaturalCubicSpline` because that basis is already linear beyond the boundary knots —
which is fine, but it's the kind of thing worth a comment or a test rather than
leaving a reader to infer it.

## Testing conventions

- Optional-dependency branches are guarded with `if _has_sklearn:` / `if _has_cvxopt:`
  / `if _has_pyqreg:` imported from `regspline.base`, not with `pytest.mark.skipif`.
  Follow the local style, but be aware it means those branches silently don't run.
- Each test file ends with an `if __name__ == "__main__":` block calling
  `pytest.main([...])` with `--pdb`, so files can be run directly for debugging.
  Add one to any new test file.
- There's a `tofix` marker registered in `pyproject.toml` for known-broken tests:
  `pytest -m "not tofix"`.
- `ruff format --check` is a CI gate, so run `ruff format` before finishing.
