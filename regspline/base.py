#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
from abc import ABC, abstractmethod

try:
    from sklearn.svm import LinearSVR, NuSVR
    from sklearn.linear_model import QuantileRegressor
except ImportError:
    _has_sklearn = False
else:
    _has_sklearn = True

try:
    import cvxopt

    if cvxopt.__version__ == "1.3.0":
        raise ImportError("SVR doesn not work well with cvxopt 1.3.0")
        # Note there is a domain error bug in 1.3.0: https://github.com/cvxopt/cvxopt/issues/202
except ImportError:
    _has_cvxopt = False
else:
    _has_cvxopt = True

try:
    from pyqreg import QuantReg as qrQuantReg
except ImportError:
    _has_pyqreg = False
else:
    _has_pyqreg = True


from .util import type_wrapper, PandasWrapper


class BasisFuncInterface(ABC):
    r"""
    Abstraction of basis functions
    """

    def __init__(self, xmin=-np.inf, xmax=np.inf, val=0):
        assert np.isscalar(xmin) and np.isreal(xmin)
        assert np.isscalar(xmax) and np.isreal(xmax)
        assert np.isscalar(val) and np.isreal(val)
        assert xmin < xmax
        self.xmax = xmax
        self.xmin = xmin
        self.val = val

    @abstractmethod
    def _apply(self, x):
        pass

    @type_wrapper(xloc=1)
    def __call__(self, x):
        y = self._apply(x)
        if np.isfinite(self.xmax):
            y = np.where(x <= self.xmax, y, self.val)
        if np.isfinite(self.xmin):
            y = np.where(x >= self.xmin, y, self.val)
        return y


class KnotsInterface(ABC):
    """
    Abstraction for a class containing knots
    """

    def __init__(self, knots):
        self._knots = None
        self.knots = knots

    def __eq__(self, other):
        if not isinstance(other, KnotsInterface):
            return False
        return np.array_equal(self.knots, other.knots)

    @property
    def knots(self):
        return self._knots

    @knots.setter
    def knots(self, value):
        if value is not None:
            value = np.asanyarray(value)
            assert len(value) >= 2, "Must specify at least 2 knots"
            assert np.all(value[:-1] <= value[1:]), "Knots are assumed to be sorted and unique"
        self._knots = value

    @property
    def n_knots(self):
        return None if self.knots is None else len(self.knots)

    def __len__(self):
        return self.n_knots


class RegressionSplineBase(KnotsInterface, ABC):
    r"""
    Regression linear spline represented by basis functions
    :math:`b_1\ldots b_M`:, and with knots :math:`k_1<k_2\ldots <k{N}`.
    The result is a function of the form
    .. math::
        s(x)=c_0+\sum\limits_{i=1}^{M} c_i b_i(x).

    The relation between the basis functions and the knots differs per
    spline. Nevertheless, in all cases, :math:`c_0` is a constant
    and the other :math:`c_i` are coefficients.

    The spline intended to be defined for :math:`k_1\leq x \leq k_N`. Outside
    of this range, the function can be extrapolated by either:

    * simply evaluating the basis functions;
    * with the value :math:`s(k_0)=c_0` left of :math:`k_0`, and with the value :math:`s(k_N)` right of :math:`k_N`;
    * with NaN.
    """

    def __init__(self, knots, coeffs, extrapolation_method="nan"):
        self._knots = None
        self._coeffs = None
        super().__init__(knots)
        self.coeffs = coeffs
        self.extrapolation_method = extrapolation_method

    def __hash__(self):
        return hash(
            (
                self.__class__.__name__,
                tuple(self.knots),
                tuple(self.coeffs),
            )
        )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        elif not KnotsInterface.__eq__(self, other):
            return False
        else:
            return (
                np.array_equal(self.coeffs, other.coeffs)
                and self.extrapolation_method == other.extrapolation_method
            )

    @property
    def extrapolation_method(self):
        return self._extrapolation_method

    @extrapolation_method.setter
    def extrapolation_method(self, value):
        assert value in ["nan", "const", "basis", "linear"]
        self._extrapolation_method = value
        if hasattr(self, "_bi_cache"):
            del self._bi_cache

    @KnotsInterface.knots.setter
    def knots(self, value):
        if value is not None and self.coeffs is not None:
            value = np.asanyarray(value)
            assert self._validate_knots_coeffs(value, self.coeffs)
        super(RegressionSplineBase, RegressionSplineBase).knots.__set__(self, value)
        if hasattr(self, "_bi_cache"):
            del self._bi_cache

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        if value is not None:
            value = np.asanyarray(value)
            assert self._validate_knots_coeffs(self.knots, value)
        self._coeffs = value

    @abstractmethod
    def _validate_knots_coeffs(self, knots, coeffs):
        return False

    @property
    @abstractmethod
    def _bi(self):
        pass

    @property
    def _ci(self):
        assert self.knots is not None and self.coeffs is not None
        return self.coeffs[1:] if self.has_const else self.coeffs

    @property
    def const(self):
        if self.coeffs is None:
            return None
        elif self.knots is None:
            return 0
        else:
            return self.coeffs[0] if self.has_const else 0

    @property
    @abstractmethod
    def has_const(self):
        return True

    @const.setter
    def const(self, value):
        assert self.knots is not None, "Cannot set constant if knots are not specified"
        assert self.coeffs is not None, "Cannot set constant if coeffs are not specified"
        assert np.isscalar(value) and np.isreal(value)
        if self.has_const:
            self.coeffs[0] = value
        else:
            self.coeffs = np.insert(self.coeffs, 0, value, axis=0)

    @property
    def n_coeffs(self):
        return None if self.coeffs is None else len(self.coeffs)

    @type_wrapper(xloc=1)
    def __call__(self, x):
        if self.knots is None or self.coeffs is None:
            return np.nan * x
        if self.extrapolation_method == "const":
            x = np.clip(x, self.knots[0], self.knots[-1])
        # Note: bi takes care of other extrapolation methods
        return self.const + self._ci.dot([bi(x) for bi in self._bi])

    def eval_basis(self, x, include_constant=False):
        """
        Evaluates the basis functions of the spline on x, usefull for
        linear regression pruposes.
        """
        wrapper = PandasWrapper(x)
        x = np.asanyarray(x, dtype=np.float64)
        y = [np.ones(x.shape)] if include_constant else []
        y += [bi(x) for bi in self._bi]
        y = np.asanyarray(y).T
        y = y.tolist() if len(y.shape) == 0 else wrapper.wrap(y)
        if isinstance(y, pd.DataFrame):
            columns = ["const"] if include_constant else []
            columns += [f"b{i}" for i, _ in enumerate(self._bi)]
            y.columns = columns
        return y

    @abstractmethod
    def prune_knots(self, method="isclose", tol=1e-6, **kwargs):
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
        raise NotImplementedError(
            "Which knots can be removed depends on coeffs and knots are related"
        )

    @classmethod
    def from_data(
        cls,
        x,
        y,
        knots=None,
        method="OLS",
        add_constant=True,
        prune=False,
        return_estim_result=False,
        backend=None,
        **kwargs,
    ):
        """
        Estimates a spline from data

        Parameters
        ----------
        x : numpy array
            observations of the independent variable
        y : numpy array
            observations of the dependent variable
        knots : numpy array, optional
            knots, defaults to 10 knots between min and max x-value
        method : string, optional
            Estimation  method. The default is "OLS".
        add_constant : boolean, optional
            Add a constant to the spline. The default is True.
        prune : boolean, optional
            Prunes insignificant knots after estimation, and estimates again. The default is False.
        return_estim_result : boolean, optional
            If True, estimation results are returned. The default is False.
        backend : None or string
            Force a certain backend, can be statsmodels, sklearn, or pyqreg. Defaults to using statsmodels where possible.

        Notable kwargs
        ------
        Dictionary with additional kwargs passed to estimation (statsmodels fit methods).
        q : float
            quantile used for quantile regression, defaults to the median (0.5)
        C : float
            Regularization parameter for SVR. The strength of the regularization is inversely proportional to C. Must be strictly positive.
        epsilon : float
            Epsilon parameter in the epsilon-insensitive loss function when using SVR. Note that the value of this parameter depends on the scale of the target variable y.
        nu : float
            Used in NuSVR. An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1]. By default 0.5 will be taken.


        Returns
        -------
        Spline object
        """
        # Initialize
        x = np.asanyarray(x)
        y = np.asanyarray(y)
        if knots is None:
            knots = np.linspace(np.min(x), np.max(x), num=10)
        elif isinstance(knots, int):
            knots = np.linspace(np.min(x), np.max(x), num=knots)
        else:
            knots = np.asanyarray(knots)
        spline = cls(knots, None, extrapolation_method=kwargs.pop("extrapolation_method", "nan"))
        # Estimate
        if method == "OLS":
            assert backend is None or backend == "statsmodels", "sklearn backend not implemented"
            smkwargs = dict(
                exog=spline.eval_basis(x, include_constant=add_constant),
                hasconst=True,
                missing=kwargs.pop("missing", "none"),
            )
            model = sm.OLS(y, **smkwargs)
            result = model.fit(**kwargs)
            spline.coeffs = result.params
            insignificant = np.abs(result.tvalues) < 1.96
            if prune and np.any(insignificant):
                add_constant = add_constant and not insignificant[0]
                spline.prune_knots(method="coeffs", coeffs_to_prune=insignificant)
                return cls.from_data(
                    x,
                    y,
                    knots=spline.knots,
                    method=method,
                    add_constant=add_constant,
                    prune=False,
                    return_estim_result=return_estim_result,
                    **kwargs,
                )
        elif method == "LASSO":
            assert backend is None or backend == "statsmodels", "sklearn backend not implemented"
            assert _has_cvxopt, "Mising optional dependency cvxopt"
            smkwargs = dict(
                exog=spline.eval_basis(x, include_constant=add_constant),
                hasconst=True,
                missing=kwargs.pop("missing", "none"),
            )
            model = sm.OLS(y, **smkwargs)
            result = model.fit_regularized(method="sqrt_lasso", **kwargs)
            spline.coeffs = result.params
            if prune:
                spline.prune_knots()
        elif method == "QuantileRegression":
            if backend is None or backend == "statsmodels":
                smkwargs = dict(
                    exog=spline.eval_basis(x, include_constant=add_constant),
                    hasconst=True,
                    missing=kwargs.pop("missing", "none"),
                )
                model = sm.QuantReg(y, **smkwargs)
                kwargs.setdefault("q", 0.5)
                result = model.fit(**kwargs)
                spline.coeffs = result.params
                insignificant = np.abs(result.tvalues) < 1.96
                if prune and np.any(insignificant):
                    add_constant = add_constant and not insignificant[0]
                    spline.prune_knots(method="coeffs", coeffs_to_prune=insignificant)
                    return cls.from_data(
                        x,
                        y,
                        knots=spline.knots,
                        method=method,
                        add_constant=add_constant,
                        prune=False,
                        return_estim_result=return_estim_result,
                        **kwargs,
                    )
            elif backend == "sklearn":
                assert _has_sklearn, "Mising optional dependency scikit learn"
                kwargs.setdefault("fit_intercept", add_constant)
                kwargs.setdefault("solver", "highs")
                kwargs.setdefault("quantile", kwargs.pop("q", 0.5))
                kwargs.setdefault("alpha", 0)
                model = QuantileRegressor(**kwargs)
                result = model.fit(spline.eval_basis(x, include_constant=False), y)
                spline.coeffs = np.append(result.intercept_, result.coef_)
                assert np.allclose(spline(x), result.predict(spline.eval_basis(x))), (
                    "Something is wrong, this should give the same result"
                )
                if prune:
                    spline.prune_knots()
            elif backend == "pyqreg":
                assert _has_pyqreg, "Mising optional dependency pyqreg"
                exog = spline.eval_basis(x, include_constant=add_constant)
                model = qrQuantReg(y, exog)
                q = kwargs.pop("q", 0.5)
                result = model.fit(q, **kwargs)
                spline.coeffs = result.params
                insignificant = np.abs(result.tvalues) < 1.96
                if prune and np.any(insignificant):
                    add_constant = add_constant and not insignificant[0]
                    spline.prune_knots(method="coeffs", coeffs_to_prune=insignificant)
                    return cls.from_data(
                        x,
                        y,
                        knots=spline.knots,
                        method=method,
                        add_constant=add_constant,
                        prune=False,
                        return_estim_result=return_estim_result,
                        **kwargs,
                    )
            else:
                raise ValueError("Invalid backend")
        elif method == "SVR":
            assert backend is None or backend == "sklearn", "statsmodels backend not implemented"
            assert _has_sklearn, "Mising optional dependency scikit learn"
            kwargs.setdefault("random_state", 0)
            kwargs.setdefault("tol", 1e-5)
            kwargs.setdefault("loss", "epsilon_insensitive")
            kwargs.setdefault("C", 1)
            kwargs.setdefault("max_iter", int(1e7))
            kwargs.setdefault("fit_intercept", add_constant)
            model = LinearSVR(**kwargs)
            result = model.fit(spline.eval_basis(x, include_constant=False), y)
            spline.coeffs = np.append(result.intercept_, result.coef_)
            assert np.allclose(spline(x), result.predict(spline.eval_basis(x))), (
                "Something is wrong, this should give the same result"
            )
            if prune:
                spline.prune_knots()
        elif method == "NuSVR":
            assert backend is None or backend == "sklearn", "statsmodels backend not implemented"
            assert _has_sklearn, "Mising optional dependency scikit learn"
            assert add_constant, "A constant is always fitted for NuSVR"
            kwargs.setdefault("tol", 1e-5)
            kwargs.setdefault("nu", 0.5)
            kwargs.setdefault("C", 1)
            kwargs.setdefault("kernel", "linear")
            kwargs.setdefault("max_iter", int(1e7))
            model = NuSVR(**kwargs)
            result = model.fit(spline.eval_basis(x, include_constant=False), y)
            spline.coeffs = np.append(result.intercept_, result.coef_)
            assert np.allclose(spline(x), result.predict(spline.eval_basis(x))), (
                "Something is wrong, this should give the same result"
            )
            if prune:
                spline.prune_knots()
        else:
            raise ValueError(f"Unknown method: {method}")
        # Return
        if return_estim_result:
            return spline, result
        else:
            return spline
