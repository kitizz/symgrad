# Copyright, Christopher Ham, 2022

from symgrad import Real
from .exact import exact


def test_add_exact():
    x = Real("x")
    y = Real("y")

    assert x + 1 == exact(x + 1)
    assert x + y == exact(x + y)
    assert x + y + 10 == exact(x + y + 10)

    assert x + 1 != exact(1 + x)
    assert x + y != exact(y + x)
    assert x + y + 10 != exact(y + x + 10)
    assert x + y + 10 != exact(y + 10 + x)
    assert x + y + 10 != exact(10 + y + x)


def test_multiply_exact():
    x = Real("x")
    y = Real("y")

    assert x * 30 == exact(x * 30)
    assert 30 * x == exact(30 * x)
    assert x * y == exact(x * y)

    assert x * 30 != exact(30 * x)
    assert x * y != exact(y * x)


def test_polynomial():
    x = Real("x")
    y = Real("y")

    base = 3 * x**3 - y + 4 * x + 5

    rearrangements = (
        x**3 * 3 - y + 4 * x + 5,
        3 * x**3 + 4 * x - y + 5,
        3 * x**3 + x * 4 - y + 5,
    )

    assert base == exact(3 * x**3 - y + 4 * x + 5)
    for other in rearrangements:
        assert base == other
        assert base != exact(other)
        assert exact(base) != other
        assert exact(base) != exact(other)
