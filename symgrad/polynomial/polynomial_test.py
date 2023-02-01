# Copyright 2022, Christopher Ham

from .polynomial import extract_polynomial

from ..expression import Expression
from ..sets import Reals
from ..variable import Variable


def test_extract_coeff_single_term():
    x = Variable("x", Reals())
    a = Variable("a", Reals())

    assert extract_polynomial(x**2, wrt=x).coefficients() == [0, 0, 1]
    assert extract_polynomial(10 * a * x**4, wrt=x).coefficients() == [0, 0, 0, 0, 10 * a]


def test_extract_coeff_no_term():
    x = Variable("x", Reals())
    a = Variable("a", Reals())

    assert extract_polynomial(a**2, wrt=x).coefficients() == [a**2]
    assert extract_polynomial(10 * a + a**2, wrt=x).coefficients() == [a**2 + 10 * a]


def test_extract_coeff_multi_term_single_var():
    x = Variable("x", Reals())

    poly = extract_polynomial(4 + 3 * x + x**2, wrt=x)
    assert poly.coefficients() == [4, 3, 1]


def test_extract_coeff_multi_term_multi_var():
    x = Variable("x", Reals())
    a = Variable("a", Reals())

    poly = extract_polynomial(a + 2 * x**2 + 10 * a * x**4, wrt=x)
    assert poly.coefficients() == [a, 0, 2, 0, 10 * a]


def test_extract_factorized():
    x = Variable("x", Reals())

    poly = extract_polynomial((2 * x - 3) * (x + 2) * (-x + 4), wrt=x)
    assert poly.coefficients() == [-24, 10, 7, -2]


def test_eval_floats():
    def poly_func(t):
        return 3 - 7 * t + 3 * t**2

    x = Variable("x", Reals())
    expr = poly_func(x)

    poly = extract_polynomial(expr)
