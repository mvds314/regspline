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
