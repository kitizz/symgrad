from . import Real
from .exact import exact


def test_identical():
    x = Real("x")

    assert x == x
    assert 2 * x == 2 * x
    assert x - 2 == x - 2
    assert x**4 == x**4
    assert -3 * x**4 + 5 == -3 * x**4 + 5


def test_duplicate_symbol():
    x1 = Real("x")
    x2 = Real("x")

    assert x1 == x2
    assert 2 * x1 == 2 * x2
    assert x1 - 2 == x2 - 2
    assert x1**4 == x2**4
    assert -3 * x1**4 + 5 == -3 * x2**4 + 5


def test_duplicate_symbol_hash():
    x1 = Real("x")
    x2 = Real("x")

    assert (x1)._hash() == (x2)._hash()
    assert (2 * x1)._hash() == (2 * x2)._hash()
    assert (x1 - 2)._hash() == (x2 - 2)._hash()
    assert (x1**4)._hash() == (x2**4)._hash()
    assert (-3 * x1**4 + 5)._hash() == (-3 * x2**4 + 5)._hash()


def test_equal_associative():
    x = Real("x")
    y = Real("y")

    assert x + 1 == 1 + x
    assert x + y == y + x
    assert x + y + 10 == y + x + 10
    assert x + y + 10 == y + 10 + x
    assert x + y + 10 == 10 + y + x

    assert x * 30 == 30 * x
    assert x * y == y * x


# def test_equal_rearrange():
#     x = Real("x")

#     # How to do this?
#     # Need some kind of canonical form to start with...
#     # I feel it's always possible to expand. In other words, make sure the all child operations
#     # are of higher or same "order" as parent op.

#     assert (x + 1) * (x + 2) == x**2 + 3 * x + 2
