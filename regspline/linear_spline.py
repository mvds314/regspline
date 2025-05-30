#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .base import BasisFuncInterface, RegressionSplineBase


class HingeBasisFunction(BasisFuncInterface):
    r"""
    Defines the Hinge basis function

    .. math::
        \text{max}(x-\text{ref}, 0)

    Furthermore, input values outside the range xmin to xmax are mapped to val.

    """

    def __init__(self, ref, xmin=-np.inf, xmax=np.inf, val=0):
        super().__init__(xmin=xmin, xmax=xmax, val=val)
        assert np.isscalar(ref) and np.isreal(ref) and np.isfinite(ref)
        if np.isfinite(xmin):
            assert xmin <= ref
        if np.isfinite(xmax):
            assert xmax > ref
        self.ref = ref

    def _apply(self, x):
        return np.fmax(x - self.ref, 0)


class LinearSpline(RegressionSplineBase):
    r"""
    Linear regression spline represented by Hinge basis functions :math:`h_i`:

    .. math:: h_i(x) = \max(x-k_i,0),

    for :math:`i=1\ldots N-1`, and where :math:`k_1<k_2<\ldots <k_{N}`
    are the knots. The result is a piecewise linear function :math:`s(x)`
    between the knots of the form:

    .. math::
        s(x)=c_0+\sum\limits_{i=1}^{N-1} c_i h_i(x),

    where :math:`c_0` is a constant and the other :math:`c_i` are coefficients.
    Note that, by construction, :math:`s(k_1)=c_0`.

    The spline intended to be defined for :math:`k_1\leq x \leq k_N`. Outside
    of this range, the function can be extrapolated by either:

    * simply evaluating the basis functions;
    * with the value :math:`s(k_0)=c_0` left of :math:`k_0`, and with the value :math:`s(k_N)` right of :math:`k_N`;
    * with NaN.

    One to interpret the spline in this basis is:

    * :math:`h_1(x)` equals zero at :math:`k_0`,sets slope in the interval :math:`k_1\leq x\leq k_1`.
    * :math:`h_2(x)` compensates and sets the slope in the interval :math:`k_2 \leq x\leq k_3`.
    * etc.

    Contrary to other basis functions,, such as spikes, one can represent a linear
    function with only one basis function. Each kink adds requires another basis
    function. Consequently, in regression, coefficients of basis functions become
    insignificant, unless there is a kink at the knot. This keeps the regression
    sparse.

    Note that this approach is similar to the MARS splines, see the py-earth package.
    This implementation is more lean, and merely implements the spline (in 1 dimension)
    without algorithms, or a modelling framework around it.

    Params
    --
        x           - float, 1d numpy array or pd.Series object
        coeffs      - pd.Series with coeffs and knots as index
                        - coeffs created by linear regression on :math:`f_i(x)` and the observed y's
    """

    def _validate_knots_coeffs(self, knots, coeffs):
        if knots is not None:
            knots = np.asanyarray(knots)
            assert len(knots) >= 2, "Must specify at least 2 knots"
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
            if self.extrapolation_method == "linear":
                self._bi_cache = [lambda x, ref=self.knots[0]: x - ref] + [
                    HingeBasisFunction(k, **kwargs) for k in self.knots[1:-1]
                ]
            else:
                self._bi_cache = [HingeBasisFunction(k, **kwargs) for k in self.knots[:-1]]
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
                # Note first can correspond to const, last knot has no coeff
                self.coeffs = None
                self.knots = None
                self.coeffs = coeffs[~to_prune]
                to_prune = np.append(
                    to_prune[1:] if len(knots) == len(coeffs) else to_prune, False
                )
                self.knots = knots[~to_prune]
            except:
                # Cleanup, don't leave the spline in an invalid state
                self.knots = None
                self.coeffs = None
                self.knots = knots
                self.coeffs = coeffs
                raise
