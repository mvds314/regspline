#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from pathlib import Path

from regspline import LinearSpline, NaturalCubicSpline, NaturalCubicSplineBasisFunction
from regspline.natural_cubic_spline import di


def test_hash():
    knots = [0.1, 0.5, 0.9, 1]
    coeffs = [2, 1, 1, 3]
    assert hash(LinearSpline(knots, coeffs)) == hash(LinearSpline(knots, coeffs))
    assert hash(LinearSpline(knots[1:], coeffs[1:])) != hash(LinearSpline(knots, coeffs))
    assert hash(LinearSpline(knots, coeffs[1:])) != hash(LinearSpline(knots, coeffs))
    assert hash(LinearSpline(knots, coeffs)) != hash(NaturalCubicSpline(knots, coeffs))


def test_equality_and_copy():
    knots = [0.1, 0.5, 0.9, 1]
    coeffs = [2, 1, 1, 3]
    spline = LinearSpline(knots, coeffs)
    assert spline == spline.copy()
    assert LinearSpline(knots, coeffs) == LinearSpline(knots, coeffs)
    assert LinearSpline(knots[1:], coeffs[1:]) != spline
    assert LinearSpline(knots, coeffs[1:]) != spline
    assert LinearSpline(knots, coeffs) != NaturalCubicSpline(knots, coeffs)


def test_hash_and_equality_with_unset_coeffs():
    """Splines without coefficients must still be hashable and comparable."""
    knots = [0.1, 0.5, 0.9, 1]
    assert hash(LinearSpline(knots, None)) == hash(LinearSpline(knots, None))
    assert hash(LinearSpline(knots, None)) != hash(LinearSpline(knots, [2, 1, 1, 3]))
    assert LinearSpline(knots, None) == LinearSpline(knots, None)
    assert LinearSpline(knots, None) != LinearSpline(knots, [2, 1, 1, 3])


def test_coincident_knots_are_rejected():
    """Duplicate knots make the design matrix singular, so reject them up front."""
    with pytest.raises(AssertionError, match="unique"):
        LinearSpline([0.0, 1.0, 1.0, 2.0], [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(AssertionError, match="unique"):
        LinearSpline([0.0, 1.0, 1.0 + 1e-14, 2.0], [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(AssertionError, match="sorted"):
        LinearSpline([0.0, 2.0, 1.0], [0.1, 0.2, 0.3])
    LinearSpline([0.0, 1.0, 2.0], [0.1, 0.2, 0.3])


def test_well_separated_knots_far_from_zero_are_accepted():
    """The uniqueness check must scale with the knot span, not the knot magnitude."""
    knots = np.linspace(1e6, 1e6 + 5, 10)
    assert LinearSpline(knots, np.ones(len(knots))).n_knots == 10
    # Hourly knots over a day of unix timestamps: huge offset, wide gaps.
    stamps = np.arange(1.7e9, 1.7e9 + 86400, 3600)
    assert LinearSpline(stamps, np.ones(len(stamps))).n_knots == len(stamps)


def test_basis_functions_differ_by_knot_index():
    """Basis functions over the same knots are distinct, and must hash that way."""
    knots = np.linspace(0, 5, 6)
    assert di(1, knots) != di(3, knots)
    assert di(1, knots) == di(1, knots)
    assert hash(di(1, knots)) != hash(di(3, knots))
    assert hash(di(1, knots)) == hash(di(1, knots))
    assert len({di(i, knots) for i in range(1, 5)}) == 4
    # Different concrete classes over the same knots are not interchangeable.
    assert di(1, knots) != NaturalCubicSplineBasisFunction(1, knots)


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                # "-k",
                # "test_hash",
                "--tb=auto",
                "--pdb",
            ]
        )
