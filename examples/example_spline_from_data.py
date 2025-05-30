#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from regspline import LinearSpline, NaturalCubicSpline


plt.close("all")
np.random.seed(100)
# Initialize
x = np.linspace(0, 4, 1000)
y = np.sin(x) + 0.05 * np.random.randn(*x.shape)
fig, ax = plt.subplots(1, 1)
ax.scatter(x, y, alpha=0.2, label="sin(x) + noise")
ax.plot(x, np.sin(x), label="sin(x)")

# Spline with OLS
knots = np.linspace(x.min(), x.max(), num=100)
ls, er = LinearSpline.from_data(x, y, knots=knots, return_estim_result=True)
ax.plot(x, ls(x), label=f"Linear spline estimated with OLS, {len(ls)} knots")
ncs, er = NaturalCubicSpline.from_data(x, y, knots=knots, return_estim_result=True)
ax.plot(x, ncs(x), label=f"Natural Cubic spline estimated with OLS, {len(ncs)} knots")


# Spline with LASSO
ls, er = LinearSpline.from_data(
    x,
    y,
    method="LASSO",
    alpha=1,
    knots=knots,
    return_estim_result=True,
    prune=True,
)
ax.plot(x, ls(x), label=f"Linear spline estimated with LASSO, {len(ls)} knots")
ncs, er = NaturalCubicSpline.from_data(
    x,
    y,
    method="LASSO",
    alpha=1,
    knots=knots,
    return_estim_result=True,
    prune=True,
)
ax.plot(x, ncs(x), label=f"Natural Cubic spline estimated with LASSO, {len(ncs)} knots")

# Spline with SVR
ls, er = LinearSpline.from_data(
    x,
    y,
    method="SVR",
    knots=knots,
    return_estim_result=True,
    prune=True,
    epsilon=0.25,
)
ax.plot(x, ls(x), label=f"Linear spline estimated with SVR, {len(ls)} knots")
ncs, er = NaturalCubicSpline.from_data(
    x,
    y,
    method="SVR",
    knots=knots,
    return_estim_result=True,
    prune=True,
    epsilon=0.25,
)
ax.plot(x, ncs(x), label=f"Natural Cubic estimated with SVR, {len(ncs)} knots")

# Spline with NuSVR
ls, er = LinearSpline.from_data(
    x,
    y,
    method="NuSVR",
    knots=knots,
    return_estim_result=True,
    prune=True,
    nu=0.005,
)
ax.plot(
    x,
    ls(x),
    label=f"Spline estimated with NuSVR, {er.dual_coef_.shape[1]} support vectors",
)
ls, er = NaturalCubicSpline.from_data(
    x,
    y,
    method="NuSVR",
    knots=knots,
    return_estim_result=True,
    prune=True,
    nu=0.005,
)
ax.plot(
    x,
    ncs(x),
    label=f"Natural Cubic spline estimated with NuSVR, {er.dual_coef_.shape[1]} support vectors",
)

# Finish
ax.legend()
