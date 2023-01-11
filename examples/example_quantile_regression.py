#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import SplineTransformer


from regspline import LinearSpline, NaturalCubicSpline, Timer


plt.close("all")
np.random.seed(100)

# Make data
with Timer("Initializing data"):
    knots = [0, 0.5, 1]
    coeffs = [0, 1, 1]
    spline = LinearSpline(knots, coeffs)
    signal = lambda x: spline(x)
    noise = lambda x: x ** 2 + 0.1
    x = np.linspace(0, 1, 500)
    y = signal(x)
    xx = np.repeat(x, 10)
    yy = signal(xx) + noise(xx) * np.random.randn(*xx.shape)
    quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    n_knots = 5


# Get quantiles by fitting a linear spline
with Timer("Running linear spline quantile regression with sklearn"):
    skllsq = {}
    for q in quantiles:
        skllsq[q] = LinearSpline(np.linspace(0, 1, n_knots), None)
        model = QuantileRegressor(quantile=q, alpha=0, solver="highs")
        res = model.fit(skllsq[q].eval_basis(xx), yy)
        skllsq[q].coeffs = np.append(res.intercept_, res.coef_)
        print(".", end="")

# Get B-spline quantiles
with Timer("Running B-spline quantile regression with sklearn"):
    bsq = {}
    for q in quantiles:
        bs = SplineTransformer(n_knots=n_knots)
        bs.fit(x.reshape(-1, 1))
        tfx = bs.transform(xx.reshape(-1, 1))
        model = QuantileRegressor(quantile=q, alpha=0, solver="highs")
        res = model.fit(bs.transform(xx.reshape(-1, 1)), yy)
        bsq[q] = res.predict(bs.transform(x.reshape(-1, 1)))
        print(".", end="")

# Get linear spline with statsmodels
with Timer("Running linear spline quantile regression with statsmodels"):
    smlsq = {}
    for q in quantiles:
        smlsq[q] = LinearSpline.from_data(
            xx,
            yy,
            knots=np.linspace(0, 1, n_knots),
            q=q,
            method="QuantileRegression",
            max_iter=int(1e8),
        )
        print(".", end="")

# Get cubic spline with statsmodels
with Timer("Running natural cubic spline quantile regression with statsmodels"):
    smcsq = {}
    for q in quantiles:
        smcsq[q] = NaturalCubicSpline.from_data(
            xx,
            yy,
            knots=np.linspace(0, 1, n_knots),
            q=q,
            method="QuantileRegression",
            max_iter=int(1e8),
        )
        print(".", end="")

# Plot everything
with Timer("Plotting"):
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, label="Signal and its quantiles", color="k")
    print(".", end="")
    ax.scatter(xx, yy, alpha=0.1, label="Observed signal + noise")
    print(".", end="")
    for q in quantiles:
        ax.plot(x, y + noise(x) * sps.norm.ppf(q), label=None, color="k")
        print(".", end="")
    label = "Linear spline quantiles sklearn"
    for q, f in skllsq.items():
        ax.plot(x, f(x), label=label, color="r", linestyle="--")
        label = None
        print(".", end="")
    label = "B-spline quantiles"
    for q, val in bsq.items():
        ax.plot(x, val, label=label, linestyle=":", color="g")
        label = None
        print(".", end="")
    label = "Linear spline quantiles"
    for q, f in smlsq.items():
        ax.plot(x, f(x), label=label, linestyle="-.", color="m")
        label = None
        print(".", end="")
    label = "Natural cubic spline quantiles"
    for q, f in smcsq.items():
        ax.plot(x, f(x), label=label, linestyle="-.", color="c")
        label = None
        print(".", end="")
    ax.legend()
