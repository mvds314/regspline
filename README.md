# Regression splines

This module includes two spline implementations splines suitable for regression: linear splines using a hinge function basis, and natural cubic splines.

```python
import numpy as np
import matplotlib.pyplot as plt
from regspline import LinearSpline
plt.close('all')

knots = [0,1,2]
coeffs = [1,2,3]
s = LinearSpline(knots, coeffs)
y=s(np.linspace(0,1))

x=np.linspace(0,np.pi)
y=np.sin(x)
xobs = np.repeat(x,50)
yobs = np.repeat(y,50) + 0.01*np.random.randn(*xobs.shape)

s, res = LinearSpline.from_data(xobs, yobs,
                                knots=np.linspace(0,np.pi,30),
                                method='OLS',
                                return_estim_result=True,
                                prune=True)

plt.plot(x,y)
plt.plot(x,s(x))
```

Several regression types are supported to extract the splines from data, including OLS, LASSO, and quantile regression. See the example files.

## Installation

You can install this library directly from github:

```bash
pip install git+https://github.com/mvds314/regspline.git
```

## Background

The module contains two splines:

* A linear spline represented by Hinge functions: $h_i(x) = \max(x-k_i,0)$, where $k_i$ are the knots.
* A natural cubic spline.

All splines can be written in terms of a linear combination of basis functions. Many splines representations don't have a one-to-one relation with the knots. For example, linear B-splines, have basis spiked basis functions: they zero up to  $k_{i-1}$, increase linearly up to $k_i$, and then decrease linearly back to zero up to $k_{i+1}$. This makes it hard to prune knots in regression, as removing one knots, changes several basis functions at the same time.

The splines chosen:

* have coefficents that have a correspondence with the knots.
* have the ability that knots can be removed when the corresponding coefficient is small, or insignificant without changing the other basis functions.
* have the ability to represent functions with sparse basis.

One way to interpret, e.g., the linear spline in the hings basis is as follows. $h_1(x)$ sets an initial slope from the first knot onwards. Then next basis function $h_2(x)$ can adjust the slope at the knot $k_2$, if no adjustment is required, its coefficient is insignificant and the knot can be removed from the spline without any impact on the other basis functions.

## Related projects

Some projects with related methods:

* [basis-expansions](https://github.com/madrury/basis-expansions)
* [py-earth](https://github.com/scikit-learn-contrib/py-earth)
* Quantile regression using decision trees [scikit-garden](https://scikit-garden.github.io/)

The module differs from these implementations as it directly implements the splines as functions, i.e., you can use them directly if you specify knots and coefficients, and they are not directly integrated with an estimation framework.

## Development

For development purposes, clone the repo:

```bash
git clone https://github.com/mvds314/regspline.git
```

Then navigate to the folder containing `setup.py` and run

```bash
pip install -e .
```
to install the package in edit mode.

Run unittests with `pytest`.

