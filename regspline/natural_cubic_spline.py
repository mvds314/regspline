#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from abc import ABC
from .base import BasisFuncInterface, KnotsInterface, RegressionSplineBase


class _NaturalCubicSplineBasisFuncInterface(KnotsInterface, BasisFuncInterface, ABC):
    r"""
    Abstraction for basis functions relative to knot :math:`k_i`, where
    :math:`k=1\ldots N`
    """

    def __init__(self, i, knots, xmin=-np.inf, xmax=np.inf, val=0):
        KnotsInterface.__init__(self, knots)
        BasisFuncInterface.__init__(self, xmin=xmin, xmax=xmax, val=val)
        assert 1 <= i <= self.n_knots - 1, (
            f"Cannot initialize for {i}th knot, only possible up to {self.n_knots - 1}th knot"
        )
        self._i = i
        if np.isfinite(xmin):
            assert xmin <= self.knots[0]
        if np.isfinite(xmax):
            assert xmax >= self.knots[-1]

    @property
    def ref(self):
        return self.knots[self._i - 1]


class di(_NaturalCubicSplineBasisFuncInterface):
    r"""
    Defines helper functions for the Natural spline basis

    .. math::
        d_i(x) = \frac{\max(x-k_i,0)^3-\max(x-k_N,0)^3}{k_N-k_i}
               = \frac{ h(x,k_i)^3-h(x,k_n)^3}{k_N-k_i}\\,

    for :math:`i=1\ldots N-1`. Here :math:`k_1<k_2<\ldots <k_{N}`
    are the knots, and the :math:`h(x,k_i)=\max(x-k_i,0)` are Hinge functions.

    See [1] Chapter 5 for further details.

    References
    ----------
    [1] Hastie, Tibshirani, Friedman (2009) - The elements of statistical learning

    """

    def _apply(self, x):
        return (np.fmax(x - self.ref, 0) ** 3 - np.fmax(x - self.knots[-1], 0) ** 3) / (
            self.knots[-1] - self.ref
        )


class NaturalCubicSplineBasisFunction(_NaturalCubicSplineBasisFuncInterface):
    r"""
    Defines the natural cubic spline basis function

    .. math::
        N_1(x) = x-k_1,

    .. math::
        N_i(x) = d_{i-1}(x)-d_{N-1}(x)

    for :math:`i=2\ldots N-1`. Here :math:`k_1<k_2<\ldots <k_{N}`
    are the knots. In total, we have :math:`N-1` basis functions, plus
    a constant.

    See [1] Chapter 5.3 for further details. Note we have changed notation
    compared to the referenc. First, the index has shifted. We left out
    the constant and handle it manually. Sometimes we can to leave it out
    of regression to prevent misspecification. Second, The first basis
    is relative to the first knot, this keeps the splines more properly
    scaled.

    References
    ----------
    [1] Hastie, Tibshirani, Friedman (2009) - The elements of statistical learning

    """

    @property
    def ref(self):
        # Note: first and second basis function are both relative to first knot
        return self.knots[self._i - 1] if self._i == 1 else self.knots[self._i - 2]

    @property
    def _dim1(self):
        if not hasattr(self, "_dim1_cache"):
            assert self._i > 1 and self._i <= self.n_knots - 1
            self._dim1_cache = di(
                self._i - 1, self.knots, xmin=self.xmin, xmax=self.xmax, val=self.val
            )
        return self._dim1_cache

    @property
    def _dNm1(self):
        if not hasattr(self, "_dNm1_cache"):
            self._dNm1_cache = di(
                self.n_knots - 1,
                self.knots,
                xmin=self.xmin,
                xmax=self.xmax,
                val=self.val,
            )
        return self._dNm1_cache

    def _apply(self, x):
        if self._i == 1:
            return x - self.ref
        else:
            return self._dim1(x) - self._dNm1(x)


class NaturalCubicSpline(RegressionSplineBase):
    r"""
    Natural cubic spline represented by basis functions :math:`N_i`:

    .. math::
        N_1(x) = x-k_1,

    .. math::
        N_i(x) = d_{i-1}(x)-d_{N-1}(x)

    for :math:`i=2\ldots N-1`. Here :math:`k_1<k_2<\ldots <k_{N}`
    are the knots. In total, we have :math:`N-1` basis functions, plus
    a constant.

    The result is a cubic spline linear function :math:`s(x)`
    between the knots of the form:

    .. math::
        s(x)=c_0+\sum\limits_{i=1}^{N-1} c_i h_i(x),

    where :math:`c_0` is a constant and the other :math:`c_i` are coefficients.
    Note that, by construction, :math:`s(k_1)=c_0`. Also, by construction,
    the spline is linear in the range outside of the knots.

    The spline intended to be defined for :math:`k_1\leq x \leq k_N`. Outside
    of this range, the function can be extrapolated by either:

    * simply evaluating the basis functions, in which case it becomes linear;
    * with the value :math:`s(k_0)=c_0` left of :math:`k_0`, and with the value :math:`s(k_N)` right of :math:`k_N`;
    * with NaN.

    Some notes on the interpretation of the coefficients:

    * First coefficient represent the constant, if there is one.
    * Next one is the coefficient of a linear function which is zero at the first knot
    * Next ones are the spline basis functions for knots :math:`k_1` to :math:`k_{N-2}`
    * Interval in between the last two knots is 'cleanup', there is no coefficient.
    """

    def _validate_knots_coeffs(self, knots, coeffs):
        if knots is not None:
            knots = np.asanyarray(knots)
            assert len(knots) >= 3, "Must specify at least 3 knots"
        if coeffs is not None:
            coeffs = np.asanyarray(coeffs)
        if coeffs is not None and knots is not None:
            assert len(knots) == len(coeffs) or len(knots) - 1 == len(coeffs)
        return True

    @property
    def _bi(self):
        assert self.knots is not None
        if not hasattr(self, "_bi_cache"):
            if self.extrapolation_method == "nan":
                kwargs = dict(xmin=self.knots[0], xmax=self.knots[-1], val=np.nan)
            else:
                kwargs = {}
            self._bi_cache = [
                NaturalCubicSplineBasisFunction(k, self.knots, **kwargs)
                for k in range(1, self.n_knots)
            ]
        return self._bi_cache

    @property
    def has_const(self):
        return self.n_knots == self.n_coeffs

    def prune_knots(self, method="isclose", tol=1e-6, coeffs_to_prune=None, **kwargs):
        """
        Prunes knots based on criterion

        Parameters
        ----------
        method : string, optional
            Method used to determine which knots to prune. The default is 'isclose'.
        tol : float, optional
            Tolerance, passed to method. The default is 1e-6.
        kwargs : dictionary
            Other keyword arguments passed to method.
        """
        # Initialize
        coeffs = np.copy(self.coeffs)
        knots = np.copy(self.knots)
        # Determine to prune
        if method == "isclose":
            kwargs.setdefault("atol", tol)
            to_prune = np.isclose(coeffs, 0, **kwargs)
        elif method == "coeffs":
            assert len(coeffs_to_prune) == self.n_coeffs
            to_prune = np.asanyarray(coeffs_to_prune)
        else:
            raise ValueError(f"Method {method} invalid")
        # Prune
        if len(to_prune) > 0:
            try:
                # Note first coeff can correspond to const, last knot has no coeff
                # Never prune the first knot, and never prune the last knot
                if self.has_const and to_prune[0]:
                    self.coeffs = self.coeffs[1:]
                    to_prune = to_prune[1:]
                # Never prune the first knot
                # And never prune the last two knots
                if self.n_knots <= 3:
                    return
                if self.has_const:
                    knots_to_prune = np.append(np.append(False, to_prune[3:]), [False, False])
                    coeffs_to_prune = np.append([False, False, False], to_prune[3:])
                else:
                    knots_to_prune = np.append(np.append(False, to_prune[2:]), [False, False])
                    coeffs_to_prune = np.append([False, False], to_prune[2:])
                self.coeffs = None
                self.knots = None
                self.coeffs = coeffs[~coeffs_to_prune]
                self.knots = knots[~knots_to_prune]
            except:
                # Cleanup, don't leave the spline in an invalid state
                self.knots = None
                self.coeffs = None
                self.knots = knots
                self.coeffs = coeffs
                raise
