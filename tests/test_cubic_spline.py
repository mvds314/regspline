#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import pytest
import warnings
from pathlib import Path

from regspline.natural_cubic_spline import di
from regspline import NaturalCubicSpline, NaturalCubicSplineBasisFunction
from regspline.base import _has_cvxopt

def test_di():
    knots = [0.1, 0.5, 0.9]
    f = di(2, knots)
    assert f(0.4) == 0
    f = di(2, knots, xmin=knots[0], xmax=knots[-1], val=10)
    assert f(f.knots[0] - 1) == 10 == f(f.knots[-1] + 1)
    assert isinstance(f([1]), np.ndarray)
    assert isinstance(f(pd.Series([1])), pd.Series)
    with pytest.raises(AssertionError):
        di(2, knots, xmin=3)
    with pytest.raises(AssertionError):
        di(2, knots, xmax=0.6)
    with pytest.raises(AssertionError):
        di(3, knots)
    with pytest.raises(AssertionError):
        di(-1, knots)


def test_basis():
    knots = [0.1, 0.5, 0.9]
    f = NaturalCubicSplineBasisFunction(2, knots)
    assert f(0.1) == 0
    delta = 0.1
    assert np.isclose(
        (f(knots[-1] + delta) - f(knots[-1])) / delta * 2 * delta + f(knots[-1]),
        f([knots[-1] + 2 * delta]),
    ), "It should be linear outside of the range"
    assert np.isclose(
        (f(knots[0] - delta) - f(knots[0])) / delta * 2 * delta + f(knots[0]),
        f([knots[0] - 2 * delta]),
    ), "It should be linear outside of the range"
    assert isinstance(f(1), float)
    assert isinstance(f([1]), np.ndarray)
    assert isinstance(f(pd.Series([1])), pd.Series)
    with pytest.raises(AssertionError):
        NaturalCubicSplineBasisFunction(2, knots, xmin=3)
    with pytest.raises(AssertionError):
        NaturalCubicSplineBasisFunction(2, knots, xmax=0)
    f = NaturalCubicSplineBasisFunction(2, knots, xmin=0.1, xmax=3, val=200)
    assert f(4) == 200 == f(0)


def test_spline():
    knots = [0.1, 0.5, 0.9]
    coeffs = [2, 1, 1]
    spline = NaturalCubicSpline(knots, coeffs)
    # nan tests
    assert np.isnan(spline(knots[0] - 3))
    assert np.isnan(spline(knots[-1] + 3))
    assert spline(knots[0]) == coeffs[0]
    # value tests
    spline = NaturalCubicSpline(knots, coeffs, extrapolation_method="basis")
    assert spline(knots[0] - 3) == -3 * coeffs[1] + spline(knots[0])
    spline = NaturalCubicSpline(knots, coeffs, extrapolation_method="linear")
    assert spline(knots[0] - 3) == -3 * coeffs[1] + spline(knots[0])
    spline = NaturalCubicSpline(knots, coeffs, extrapolation_method="const")
    assert spline(knots[-1] + 1) == spline(knots[-1])
    assert spline(knots[0] - 1) == spline(knots[0])
    # type test
    assert isinstance(spline(1), float)
    assert isinstance(spline([1]), np.ndarray)
    assert isinstance(spline(pd.Series([1])), pd.Series)
    # eval basis test
    x = np.linspace(knots[0], knots[-1], num=10)
    assert np.allclose(
        spline(x), spline.eval_basis(x, include_constant=True).dot(spline.coeffs)
    )
    assert isinstance(spline.eval_basis(1), np.ndarray)
    assert isinstance(spline.eval_basis([1, 2]), np.ndarray)
    assert isinstance(spline.eval_basis(pd.Series([1, 2])), pd.DataFrame)


def test_pruning():
    # Test 1: first knot is never pruned
    knots = [0.1, 0.5, 0.8, 0.9]
    coeffs = [2, 1, 0, 1]
    ncs = NaturalCubicSpline(knots, coeffs)
    ncs.prune_knots(method="isclose")
    ncs2 = NaturalCubicSpline(knots, coeffs)
    ncs2.prune_knots(method="coeffs", coeffs_to_prune=[False, False, True, False])
    assert np.allclose(ncs.knots, knots)
    assert np.allclose(ncs.coeffs, coeffs)
    assert np.allclose(ncs.knots, ncs2.knots)
    assert np.allclose(ncs.coeffs, ncs2.coeffs)
    # Test 2: first knot is never pruned
    knots = [0.1, 0.5, 0.8, 0.9]
    coeffs = [1, 0, 1]
    ncs = NaturalCubicSpline(knots, coeffs)
    ncs.prune_knots(method="isclose")
    ncs2 = NaturalCubicSpline(knots, coeffs)
    ncs2.prune_knots(method="coeffs", coeffs_to_prune=[False, True, False])
    assert np.allclose(ncs.knots, knots)
    assert np.allclose(ncs.coeffs, coeffs)
    assert np.allclose(ncs.knots, ncs2.knots)
    assert np.allclose(ncs.coeffs, ncs2.coeffs)
    # Test 3: prune something
    knots = [0.1, 0.5, 0.8, 0.9]
    coeffs = [1, 2, 0]
    ncs = NaturalCubicSpline(knots, coeffs)
    ncs.prune_knots(method="isclose")
    ncs2 = NaturalCubicSpline(knots, coeffs)
    ncs2.prune_knots(method="coeffs", coeffs_to_prune=[False, False, True])
    assert np.allclose(ncs.knots, [0.1, 0.8, 0.9])
    assert np.allclose(ncs.coeffs, [1, 2])
    assert np.allclose(ncs.knots, ncs2.knots)
    assert np.allclose(ncs.coeffs, ncs2.coeffs)
    # Raise errors
    with pytest.raises(ValueError):
        ncs.prune_knots(method="invalid_method")
    with pytest.raises(AssertionError):
        ncs.prune_knots(method="coeffs", coeffs_to_prune=[False])


def test_from_data():
    np.random.seed(101)
    # Basic OLS test
    knots = [0.1, 0.5, 0.8, 0.9]
    coeffs = [2, 1, 1, 1]
    spline = NaturalCubicSpline(knots, coeffs)
    x = np.linspace(knots[0], knots[-1], num=100)
    y = spline(x) + 0.001 * np.random.randn(*x.shape)
    fs = NaturalCubicSpline.from_data(x, y, knots=knots)
    assert np.allclose(fs.coeffs, coeffs, atol=1e-2)
    # More knots OLS test
    knots = np.linspace(0.1, 0.9, 9)
    x = np.repeat(x, 500)
    y = spline(x) + 0.001 * np.random.randn(*x.shape)
    fs = NaturalCubicSpline.from_data(x, y, knots=knots, prune=True)
    overlapping_knots = [0] + [
        i + 1 for i, k in enumerate(fs.knots[:-1]) if k in spline.knots
    ]
    other_knots = [i + 1 for i, k in enumerate(fs.knots[:-1]) if k not in spline.knots]
    assert np.allclose(fs.coeffs[overlapping_knots], spline.coeffs, atol=1e-2)
    assert np.allclose(fs.coeffs[other_knots], 0, atol=1e-2)
    if _has_cvxopt:
        # Test LASSO estimation
        fs = NaturalCubicSpline.from_data(
            x, y, method="LASSO", knots=[0.1, 0.3, 0.5, 0.8, 0.9], prune=True, alpha=10
        )
        fs.prune_knots(tol=1e-2)
        assert np.allclose(fs.knots, spline.knots)
        assert np.allclose(fs.coeffs, spline.coeffs, atol=1e-2)
    else:
        warnings.warn("Optional dependency cvxopt not found, cannot test LASSO")
    # Test Quantile estimation
    fs = NaturalCubicSpline.from_data(
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
