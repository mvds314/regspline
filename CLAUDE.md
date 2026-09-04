# regspline — Copilot instructions

1-D regression splines implemented as *callable function objects*, deliberately kept
outside any modelling/estimation framework (unlike py-earth or basis-expansions).
The only estimation entry point is the `from_data` classmethod.

## Commands

```bash
pip install -e .                    # editable install (flit backend)
pip install .[SVR,LASSO,FASTQR]     # optional deps: scikit-learn, cvxopt!=1.3.0, pyqreg2

pytest                              # full suite
pytest tests/test_linear_spline.py::test_from_data   # single test
pytest -m "not tofix"               # skip tests marked `tofix`

ruff format --check                 # CI fails on formatting; run `ruff format` to fix
ruff check                          # lint (E, F, W; line-length 99)
```

CI (`.github/workflows/test.yml`) runs `pytest` then `ruff format --check` on Python 3.x
with `.[SVR,LASSO]` installed only — `pyqreg` paths are not exercised there.

Each test file has an `if __name__ == "__main__":` block that calls `pytest.main([...])`
with `--pdb`, so files can be run directly for debugging.

## Architecture

`base.py` holds all the abstraction; the two concrete splines are thin.

- `BasisFuncInterface` — a basis function object. Subclasses implement `_apply(x)`;
  the base `__call__` handles clipping outside `[xmin, xmax]` to `val` (used to produce
  NaN extrapolation).
- `KnotsInterface` — knot storage plus validation (strictly increasing, >= 2 knots).
- `RegressionSplineBase(KnotsInterface)` — the spline: `s(x) = const + sum(c_i * b_i(x))`.
  Subclasses supply only `_validate_knots_coeffs`, `_bi` (the basis function list),
  `has_const`, and `prune_knots`.
- `LinearSpline` uses hinge basis `max(x - k_i, 0)`; `NaturalCubicSpline` uses the
  natural cubic basis from Hastie/Tibshirani/Friedman ch. 5.3 (min 3 knots).

The design constraint that drives everything: **each coefficient corresponds one-to-one
to a knot**, so a knot whose coefficient is insignificant can be dropped without
changing the other basis functions. Preserve this property in any new spline type.

## Conventions

- **Constant is implicit in the coefficient count.** `has_const` is `n_knots == n_coeffs`;
  if there is one fewer coefficient than knots, there is no constant. `const`, `_ci`, and
  the pruning logic all branch on this. Never store the constant separately.
- **Coefficient/knot setters cross-validate.** Setting `knots` or `coeffs` asserts
  consistency against the other, so to change both you must first null one out
  (`self.coeffs = None; self.knots = None; ...`). `prune_knots` does exactly this and
  restores the old state in an `except` block — keep that fail-safe behaviour.
- **`_bi` is cached in `_bi_cache`.** Any setter that invalidates the basis (`knots`,
  `extrapolation_method`) must `del self._bi_cache`.
- **Four extrapolation methods**: `"nan"`, `"const"`, `"basis"`, `"linear"`. `"const"` is
  handled by clipping `x` in `__call__`; the others are handled inside `_bi`
  (`"nan"` passes `xmin`/`xmax`/`val=nan` to the basis functions; `"linear"` replaces the
  first hinge with `x - k_1`). New behaviour usually belongs in `_bi`, not `__call__`.
- **pandas in / pandas out.** Public functions are wrapped with `@type_wrapper(xloc=...)`
  or use statsmodels' `PandasWrapper` (see `util.py`) so a Series in yields a Series out
  and a scalar yields a scalar. Apply the same wrapper to new public methods.
- **Adding an estimation method** means adding a branch to `RegressionSplineBase.from_data`
  in `base.py`, which then dispatches on `method` and an optional `backend`
  (`"statsmodels"` / `"sklearn"` / `"pyqreg"`). Follow the existing branch shape: build the
  design matrix with `spline.eval_basis(x, include_constant=add_constant)`, assign
  `spline.coeffs`, then honour `prune`. For t-value-based backends, pruning calls the
  nested `refit()` helper, which re-enters `cls.from_data` with the pruned knots and
  `prune=False`; for backends without t-values it calls `spline.prune_knots()`
  (magnitude-based). **Anything popped out of `kwargs` must be recorded in
  `refit_settings`**, or the pruned refit silently answers a different question — this
  bug shipped three times, most recently as a `q=0.10` request returning a median fit.
  sklearn branches pack coefficients with `_sklearn_coeffs(result, fit_intercept)`,
  because `intercept_` is a hard `0.0` when no intercept was fitted, and assert
  `spline(x)` matches `result.predict(...)`.
- **Optional dependencies** are probed at import in `base.py` into module-level
  `_has_sklearn`, `_has_cvxopt`, `_has_pyqreg` flags. Code paths `assert _has_*` with a
  "Missing optional dependency" message; tests import these flags and guard with
  `if _has_x:` rather than skip markers. Note cvxopt 1.3.0 is deliberately rejected.
