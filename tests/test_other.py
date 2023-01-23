#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from pathlib import Path

from regspline import LinearSpline, NaturalCubicSpline


def test_hash():
    knots = [0.1, 0.5, 0.9, 1]
    coeffs = [2, 1, 1, 3]
    assert hash(LinearSpline(knots, coeffs)) == hash(LinearSpline(knots, coeffs))
    assert hash(LinearSpline(knots[1:], coeffs[1:])) != hash(LinearSpline(knots, coeffs))
    assert hash(LinearSpline(knots, coeffs[1:])) != hash(LinearSpline(knots, coeffs))
    assert hash(LinearSpline(knots, coeffs)) != hash(NaturalCubicSpline(knots, coeffs))


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
