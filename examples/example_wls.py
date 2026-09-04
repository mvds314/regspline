#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted least squares.

Observations here come from two instruments: a precise one and a crude one that
is roughly sixty times noisier. OLS treats both alike, so the crude readings drag
the fitted curve around. WLS with weights proportional to the inverse noise
variance discounts them and recovers the underlying curve far more closely.

Note that a plain unweighted RMSE does not always flatter WLS. When the noise
grows smoothly with x and the basis is local, as with hinge functions, WLS mostly
reallocates fit from the noisy region to the quiet one rather than improving the
curve overall. The clear win is the case below, where observations genuinely
differ in measurement precision.
"""

import numpy as np
import matplotlib.pyplot as plt

from regspline import LinearSpline


plt.close("all")
rng = np.random.default_rng(0)

# Piecewise linear truth with kinks at 1 and 3
truth = LinearSpline([0, 1, 3, 4], [0.0, 2.0, -1.5, 0.5])

n = 600
x = np.linspace(0, 4, n)
precise = rng.random(n) < 0.7
sigma = np.where(precise, 0.05, 3.0)
y = truth(x) + sigma * rng.standard_normal(n)

# The usual WLS choice: weight each observation by its inverse noise variance
weights = 1.0 / sigma**2

knots = np.linspace(x.min(), x.max(), num=12)

ols = LinearSpline.from_data(x, y, knots=knots, method="OLS")
wls = LinearSpline.from_data(x, y, knots=knots, method="WLS", weights=weights)

# Omitting weights reproduces OLS, since sm.WLS defaults to unit weights
unweighted = LinearSpline.from_data(x, y, knots=knots, method="WLS")
assert np.allclose(unweighted(x), ols(x))

for name, spline in [("OLS", ols), ("WLS", wls)]:
    rmse = np.sqrt(np.mean((spline(x) - truth(x)) ** 2))
    print(f"{name}: RMSE vs truth {rmse:.4f}")

fig, ax = plt.subplots(1, 1)
ax.scatter(x[precise], y[precise], alpha=0.5, s=10, label="precise instrument")
ax.scatter(x[~precise], y[~precise], alpha=0.3, s=10, label="crude instrument")
ax.plot(x, truth(x), "k--", lw=2, label="truth")
ax.plot(x, ols(x), lw=2, label="OLS")
ax.plot(x, wls(x), lw=2, label="WLS, weights = 1 / sigma^2")
ax.set_ylim(truth(x).min() - 3, truth(x).max() + 3)
ax.legend()
