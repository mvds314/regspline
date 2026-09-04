#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import pytest
import warnings
from pathlib import Path

from regspline import LinearSpline, HingeBasisFunction
from regspline.base import _has_sklearn, _has_cvxopt, _has_pyqreg


def test_basis():
    f = HingeBasisFunction(2)
    assert f(1) == 0
    assert f(f.ref + 1) == 1
    assert isinstance(f(1), int)
    assert isinstance(f([1]), np.ndarray)
    assert isinstance(f(pd.Series([1])), pd.Series)
    with pytest.raises(Exception):
        HingeBasisFunction(2, xmin=3)
    with pytest.raises(Exception):
        HingeBasisFunction(2, xmax=1)
    f = HingeBasisFunction(2, xmin=1, xmax=3, val=200)
    assert f(4) == 200 == f(0)


def test_spline():
    knots = [0.1, 0.5, 0.9]
    coeffs = [2, 1, 1]
    spline = LinearSpline(knots, coeffs)
    # nan tests
    assert np.isnan(spline(knots[0] - 3))
    assert np.isnan(spline(knots[-1] + 3))
    # value tests
    spline = LinearSpline(knots, coeffs, extrapolation_method="basis")
    assert spline(knots[0] - 3) == coeffs[0]
    assert spline(knots[1]) == (knots[1] - knots[0]) * coeffs[1] + coeffs[0]
    assert (
        spline(knots[2])
        == (knots[2] - knots[0]) * coeffs[1] + (knots[1] - knots[0]) * coeffs[2] + coeffs[0]
    )
    spline = LinearSpline(knots, coeffs, extrapolation_method="const")
    assert spline(knots[-1] + 1) == spline(knots[-1])
    assert spline(knots[0] - 1) == spline(knots[0])
    # type test
    assert isinstance(spline(1), float)
    assert isinstance(spline([1]), np.ndarray)
    assert isinstance(spline(pd.Series([1])), pd.Series)
    # eval basis test
    x = np.linspace(knots[0], knots[-1], num=10)
    assert np.allclose(spline(x), spline.eval_basis(x, include_constant=True).dot(spline.coeffs))
    assert isinstance(spline.eval_basis(1), np.ndarray)
    assert isinstance(spline.eval_basis([1, 2]), np.ndarray)
    assert isinstance(spline.eval_basis(pd.Series([1, 2])), pd.DataFrame)


def test_pruning():
    # Test 1
    knots = [0.1, 0.5, 0.8, 0.9]
    coeffs = [2, 1, 0, 1]
    ls = LinearSpline(knots, coeffs)
    ls.prune_knots(method="isclose")
    ls2 = LinearSpline(knots, coeffs)
    ls2.prune_knots(method="coeffs", coeffs_to_prune=[False, False, True, False])
    assert np.allclose(ls.knots, [0.1, 0.8, 0.9])
    assert np.allclose(ls.coeffs, [2, 1, 1])
    assert np.allclose(ls.knots, ls2.knots)
    assert np.allclose(ls.coeffs, ls2.coeffs)
    # Test 2
    knots = [0.1, 0.5, 0.8, 0.9]
    coeffs = [0, 1, 0, 1]
    ls = LinearSpline(knots, coeffs)
    ls.prune_knots(method="isclose")
    ls2 = LinearSpline(knots, coeffs)
    ls2.prune_knots(method="coeffs", coeffs_to_prune=[True, False, True, False])
    assert np.allclose(ls.knots, [0.1, 0.8, 0.9])
    assert np.allclose(ls.coeffs, [1, 1])
    assert np.allclose(ls.knots, ls2.knots)
    assert np.allclose(ls.coeffs, ls2.coeffs)
    # Test 3
    knots = [0.1, 0.5, 0.8, 0.9]
    coeffs = [1, 0, 1]
    ls = LinearSpline(knots, coeffs)
    ls.prune_knots(method="isclose")
    ls2 = LinearSpline(knots, coeffs)
    ls2.prune_knots(method="coeffs", coeffs_to_prune=[False, True, False])
    assert np.allclose(ls.knots, [0.1, 0.8, 0.9])
    assert np.allclose(ls.coeffs, [1, 1])
    assert np.allclose(ls.knots, ls2.knots)
    assert np.allclose(ls.coeffs, ls2.coeffs)
    # Raise errors
    with pytest.raises(ValueError):
        ls.prune_knots(method="invalid_method")
    with pytest.raises(AssertionError):
        ls.prune_knots(method="coeffs", coeffs_to_prune=[False])


def test_from_data():
    np.random.seed(101)
    # Basic OLS test
    knots = [0.1, 0.5, 0.9]
    coeffs = [2, 1, 1]
    spline = LinearSpline(knots, coeffs)
    x = np.linspace(knots[0], knots[-1], num=100)
    y = spline(x) + 0.001 * np.random.randn(*x.shape)
    fs = LinearSpline.from_data(x, y, knots=knots)
    assert np.allclose(fs.coeffs, coeffs, atol=1e-2)
    # More knots OLS test
    knots = np.linspace(0.1, 0.9, 7)
    x = np.repeat(x, 50)
    y = spline(x) + 0.01 * np.random.randn(*x.shape)
    fs = LinearSpline.from_data(x, y, knots=knots)
    overlapping_knots = [0] + [i + 1 for i, k in enumerate(fs.knots[:-1]) if k in spline.knots]
    other_knots = [i + 1 for i, k in enumerate(fs.knots[:-1]) if k not in spline.knots]
    assert np.allclose(fs.coeffs[overlapping_knots], spline.coeffs, atol=1e-2)
    assert np.allclose(fs.coeffs[other_knots], 0, atol=1e-2)
    # Test pruning manually
    fs.prune_knots(tol=1e-2)
    assert np.allclose(fs.knots, spline.knots)
    assert np.allclose(fs.coeffs, spline.coeffs, atol=1e-2)
    # Test prune insignificant in estmation
    fs = LinearSpline.from_data(x, y, knots=knots, prune=True)
    assert np.allclose(fs.knots, spline.knots)
    assert np.allclose(fs.coeffs, spline.coeffs, atol=1e-2)
    if _has_cvxopt:
        # Test LASSO estimation
        fs = LinearSpline.from_data(x, y, method="LASSO", knots=knots, prune=True, alpha=1)
        fs.prune_knots(tol=1e-2)
        assert np.allclose(fs.knots, spline.knots)
        assert np.allclose(fs.coeffs, spline.coeffs, atol=1e-2)
    else:
        warnings.warn("Optional dependency cvxopt not found, cannot test LASSO")
    if _has_sklearn:
        # Test SVR estimation
        fs = LinearSpline.from_data(x, y, method="SVR", knots=knots)
        fs.prune_knots(tol=5e-2)
        assert np.allclose(fs.knots, spline.knots)
        assert np.allclose(fs.coeffs, spline.coeffs, atol=2e-2)
        # Test NuSVR estimation
        fs = LinearSpline.from_data(x, y, method="NuSVR", knots=knots)
        fs.prune_knots(tol=5e-2)
        assert np.allclose(fs.knots, spline.knots)
        assert np.allclose(fs.coeffs, spline.coeffs, atol=2e-2)
    else:
        warnings.warn("Optional dependency scikit learn not found, cannot test SVR")
    # Test Quantile estimation
    fs = LinearSpline.from_data(
        x,
        y,
        method="QuantileRegression",
        q=0.5,
        knots=[0.1, 0.3, 0.5, 0.8, 0.9],
        prune=True,
    )
    fs.prune_knots(tol=1e-2)
    assert np.allclose(fs.knots, spline.knots)
    assert np.allclose(fs.coeffs, spline.coeffs, atol=1e-2)
    if _has_sklearn:
        fs = LinearSpline.from_data(
            x,
            y,
            method="QuantileRegression",
            q=0.5,
            backend="sklearn",
            knots=[0.1, 0.3, 0.5, 0.8, 0.9],
            prune=True,
        )
        fs.prune_knots(tol=1e-2)
        assert np.allclose(fs.knots, spline.knots)
        assert np.allclose(fs.coeffs, spline.coeffs, atol=1e-2)
    else:
        warnings.warn("Optional dependency scikit learn not found, cannot quantile regression")
    if _has_pyqreg:
        fs = LinearSpline.from_data(
            x,
            y,
            method="QuantileRegression",
            q=0.5,
            backend="pyqreg",
            knots=[0.1, 0.3, 0.5, 0.8, 0.9],
            prune=True,
        )
        fs.prune_knots(tol=1e-2)
        assert np.allclose(fs.knots, spline.knots)
        assert np.allclose(fs.coeffs, spline.coeffs, atol=1e-2)
    else:
        warnings.warn("Optional dependency pyqreg learn not found, cannot quantile regression")


def _wls_fixture():
    np.random.seed(101)
    knots = [0.1, 0.5, 0.9]
    coeffs = [2, 1, 1]
    spline = LinearSpline(knots, coeffs)
    x = np.repeat(np.linspace(knots[0], knots[-1], num=100), 50)
    y = spline(x) + 0.01 * np.random.randn(*x.shape)
    return spline, x, y


def test_from_data_wls():
    spline, x, y = _wls_fixture()
    knots = np.linspace(0.1, 0.9, 7)
    ols = LinearSpline.from_data(x, y, knots=knots, method="OLS")
    # Unit weights must reproduce the OLS fit
    wls = LinearSpline.from_data(x, y, knots=knots, method="WLS", weights=np.ones_like(y))
    assert np.allclose(wls.coeffs, ols.coeffs, atol=1e-8)
    # Omitting weights entirely must also work and match OLS
    wls_noweights = LinearSpline.from_data(x, y, knots=knots, method="WLS")
    assert np.allclose(wls_noweights.coeffs, ols.coeffs, atol=1e-8)
    # Near-zero weights must suppress the influence of corrupted observations
    x_dup = np.concatenate([x, x])
    y_dup = np.concatenate([y, y + 10])
    weights = np.concatenate([np.ones_like(y), np.full_like(y, 1e-12)])
    wls = LinearSpline.from_data(x_dup, y_dup, knots=knots, method="WLS", weights=weights)
    assert np.allclose(wls.coeffs, ols.coeffs, atol=1e-2)
    # Weighting must actually change the fit relative to unweighted OLS
    unweighted = LinearSpline.from_data(x_dup, y_dup, knots=knots, method="OLS")
    assert not np.allclose(unweighted.coeffs, wls.coeffs, atol=1e-2)
    # Pruning must stay weighted on the refit and recover the true knots
    pruned = LinearSpline.from_data(
        x_dup, y_dup, knots=knots, method="WLS", weights=weights, prune=True
    )
    assert np.allclose(pruned.knots, spline.knots)
    assert np.allclose(pruned.coeffs, spline.coeffs, atol=1e-2)


@pytest.mark.parametrize("method", ["OLS", "WLS", "QuantileRegression"])
def test_from_data_missing_is_forwarded_to_pruning_refit(method):
    """missing='drop' must survive the recursive refit triggered by prune=True."""
    _, x, y = _wls_fixture()
    y = y.copy()
    y[::40] = np.nan
    knots = np.linspace(0.1, 0.9, 7)
    fs = LinearSpline.from_data(x, y, knots=knots, method=method, missing="drop", prune=True)
    assert not np.any(np.isnan(fs.coeffs))


def test_prune_knots_never_prunes_first_knot():
    """The first knot is the left domain boundary and must survive pruning."""
    knots = [0.0, 1.0, 2.0, 3.0]
    # First coefficient after the constant is the overall slope; ask to prune it
    ls = LinearSpline(knots, [0.5, 0.0, 2.0, 0.0])
    ls.prune_knots(method="coeffs", coeffs_to_prune=[False, True, False, True])
    assert ls.knots[0] == 0.0
    assert ls.knots[-1] == 3.0
    assert len(ls.knots) == len(ls.coeffs)
    # Magnitude pruning must protect the boundary too
    ls = LinearSpline(knots, [0.0, 2.0, 0.0])
    ls.prune_knots(tol=1e-6)
    assert ls.knots[0] == 0.0
    assert ls.knots[-1] == 3.0


def test_from_data_prune_keeps_domain_when_first_basis_insignificant():
    """Pruning must not shrink the domain and crash the recursive refit."""
    rng = np.random.default_rng(0)
    truth = LinearSpline([0, 1, 3, 4], [0.0, 2.0, -1.5, 0.5])
    x = np.linspace(0, 4, 2000)
    y = truth(x) + (0.05 + 0.6 * x) * rng.standard_normal(x.shape[0])
    knots = np.linspace(0, 4, 40)
    fs = LinearSpline.from_data(x, y, knots=knots, method="OLS", prune=True)
    assert fs.knots[0] == knots[0]
    assert fs.knots[-1] == knots[-1]
    assert not np.any(np.isnan(fs(x)))


def test_from_data_extrapolation_method_is_forwarded_to_pruning_refit():
    """extrapolation_method must survive the recursive refit triggered by prune=True."""
    _, x, y = _wls_fixture()
    knots = np.linspace(0.1, 0.9, 7)
    fs = LinearSpline.from_data(
        x, y, knots=knots, method="OLS", prune=True, extrapolation_method="basis"
    )
    assert fs.extrapolation_method == "basis"
    assert not np.isnan(fs(1.5))


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                # "-k",
                # "test_pruning",
                "--tb=auto",
                "--pdb",
            ]
        )
