# Copyright, Christopher Ham, 2022

import logging

from symgrad import Real, exact
from .cancel import cancel


def test_identities():
    x = Real("x")

    assert cancel(x - x) == 0
    assert cancel(x / x) == 1


def test_chains():
    x = Real("x")
    y = Real("y")

    assert cancel(y * x / y) == exact(x)
    assert cancel(y * x / x) == exact(y)


def test_recursive():
    x = Real("x")
    y = Real("y")

    assert cancel(x - (y * x / y)) == exact(0)


def test_partial_cancel_add_single():
    x = Real("x")

    assert cancel(2 * x - x) == exact(x)
    assert cancel(7 * x - 3 * x) == exact(4 * x)
    assert cancel(3 * x - 7 * x) == exact(-4 * x)


def test_partial_cancel_add_multi():
    x = Real("x")
    y = Real("y")

    assert cancel(2 * x * y - x * y) == exact(x * y)
    assert cancel(7 * x * y - 3 * x * y) == exact(4 * x * y)
    assert cancel(3 * x * y - 7 * x * y) == exact(-4 * x * y)


def test_partial_ambiguous_add():
    x = Real("x")
    y = Real("y")

    # x + y - x * y ->
    assert cancel(x + y - x * y) == exact(x**4)


def test_partial_cancel_mult():
    x = Real("x")

    assert cancel(x**2 / x) == exact(x)
    assert cancel(x**7 / x**3) == exact(x**4)
    assert cancel(x**3 / x**7) == exact(x**-4)

    assert cancel(x**7 / -(x**3)) == exact(-(x**4))
    assert cancel(x**7 / (-x) ** 3) == exact(-(x**4))
    assert cancel(x**7 / -(x**-3)) == exact(-(x**10))
    assert cancel(x**7 / (-x) ** -3) == exact(-(x**10))
