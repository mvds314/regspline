#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from regspline import LinearSpline, NaturalCubicSpline

plt.close("all")
x = np.linspace(0, 1, 1000)
knots = np.linspace(0, 1, 5)
coeffs = [1, -2, 3, -4]

ls = LinearSpline(knots, coeffs, extrapolation_method="basis")
ncs = NaturalCubicSpline(knots, coeffs, extrapolation_method="basis")

fig, axs = plt.subplots(1, 3)
ax = axs[0]
ax.plot(x, ls(x), label="Linear spline")
ax.plot(x, ncs(x), label="Natural cubic spline")
ax.legend()
ax.set_title("Splines")

ax = axs[1]
x = pd.Series(x)
for col, sr in ls.eval_basis(x, include_constant=False).items():
    ax.plot(x.values, sr.values)
ax.set_title("Linear spline basis functions")

ax = axs[2]
x = pd.Series(x)
for col, sr in ncs.eval_basis(x, include_constant=False).items():
    ax.plot(x.values, sr.values)
ax.set_title("Natural cubic spline basis functions")
