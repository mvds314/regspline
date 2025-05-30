#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import timeit

from regspline import LinearSpline, NaturalCubicSpline

plt.close("all")
x = np.linspace(-1, 2, 50)
knots = [0, 0.5, 1]
coeffs = [0, 1, 1]

ls = LinearSpline(knots, coeffs, extrapolation_method="basis")
dt = timeit.timeit(lambda: ls(x), number=100)
print(f"Evaluation of linear spline takes {dt:.4f} seconds")

ncs = NaturalCubicSpline(knots, coeffs, extrapolation_method="basis")
dt = timeit.timeit(lambda: ncs(x), number=100)
print(f"Evaluation of natural cubic takes {dt:.4f} seconds")

fig, ax = plt.subplots(1, 1)
ax.plot(x, ls(x), label="Linear spline")
ax.plot(x, ncs(x), label="Natural cubic spline")
ax.legend()

x = pd.Series(x)
ls.eval_basis(x, include_constant=True)
ncs.eval_basis(x, include_constant=True)
