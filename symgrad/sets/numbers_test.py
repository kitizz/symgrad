# Copyright, Christopher Ham, 2022

import logging
import math

from .numbers import *
from ..constant import Constant
from ..variable import Variable
from ..rules.equiv import equiv
from ..exact import exact
from ..expression import Expression
from ..operators import Multiply, Power


def test_set_map_real():
    assert Constant(3.7).output_set == Reals()
    assert Constant(-3.7).output_set == Reals()
    assert Constant(math.inf).output_set == Reals()
    assert Constant(-math.inf).output_set == Reals()
    assert Constant(math.nan).output_set == Reals()

    assert Constant(2.0) in Reals()
    assert Constant(1.0) in Reals()
    assert Constant(0.0) in Reals()


def test_set_map_int():
    assert Constant(3.0).output_set == NonNegativeInts()
    assert Constant(2).output_set == NonNegativeInts()
    assert Constant(-2).output_set == Ints()

    assert Constant(2.0) in Ints()
    assert Constant(1.0) in Ints()
    assert Constant(1) in Ints()
    assert Constant(0.0) in Ints()


def test_set_map_one():
    assert Constant(1.0) == One(Reals())
    assert Constant(1) == One(Ints())
    assert Constant(1) == One(NonNegativeInts())


def test_set_map_zero():
    assert Constant(0.0) == Zero(Reals())
    assert Constant(-0.0) == Zero(Reals())
    assert Constant(0) == Zero(Ints())
    assert Constant(0) == Zero(NonNegativeInts())


def test_constant_sets():
    assert Expression.wrap(0) in Ints()
    assert Expression.wrap(1) in Ints()
    assert Expression.wrap(2) in Ints()


def test_add_ints():
    a = Expression.wrap(1)
    logging.warning(f"a type: {type(a)} {a.output_set}")
    assert (a + a) in Ints()


def test_add_reals():
    a = Expression.wrap(1.5)
    assert (a + a) in Reals()
    assert (a + a) in Ints()
    assert a not in Ints()


def test_multiply_ints():
    a = Expression.wrap(1)
    assert (a * a) in Ints()


def test_multiply_reals():
    a = Expression.wrap(2.5)
    assert (a * a) in Reals()


def test_expression():
    x = Variable("x", Reals())

    assert (2 * x) is not None


def test_identical():
    x = Variable("x", Reals())

    assert exact(x) == x
    assert exact(2 * x) == 2 * x
    assert exact(x - 2) == x - 2
    assert exact(x**4) == x**4
    assert exact(-3 * x**4 + 5) == -3 * x**4 + 5


def test_duplicate_symbol():
    x1 = Variable("x", Reals())
    x2 = Variable("x", Reals())

    assert exact(x1) == x2
    assert exact(2 * x1) == 2 * x2
    assert exact(x1 - 2) == x2 - 2
    assert exact(x1**4) == x2**4
    assert exact(-3 * x1**4 + 5) == -3 * x2**4 + 5


def test_equiv_associative():
    x = Variable("x", Reals())
    y = Variable("y", Reals())

    assert equiv((x * y) * 10) == x * (y * 10)
    assert equiv((x * y) * 10) == x * (y * 10)


def test_equiv_commutative():
    x = Variable("x", Reals())
    y = Variable("y", Reals())

    assert equiv(x + 1) == 1 + x
    assert equiv(x + y) == y + x
    assert equiv(x + y + 10) == y + x + 10
    assert equiv(x + y + 10) == y + 10 + x
    assert equiv(x + y + 10) == 10 + y + x

    assert equiv(x * 30) == 30 * x
    assert equiv(x * y) == y * x


def test_auto_condense_constants():
    x = Variable("x", Reals())
    a = Expression.wrap(5)
    b = Expression.wrap(1)

    assert a * b * x == exact(5 * x)
    assert a * x * b == exact(5 * x)
    assert b * x * a == exact(5 * x)

    assert a + b + x == exact(x + 6)
    assert a + x + b == exact(x + 6)
    assert b + x + a == exact(x + 6)


def test_auto_condense_identity():
    x = Variable("x", Reals())

    assert 1 * x == exact(x)
    assert x * 1 == exact(x)

    assert 0 + x == exact(x)
    assert x + 0 == exact(x)

    assert Multiply(x**2, Power(2, 0)) == exact(x**2)


def test_auto_condense_negative():
    x = Variable("x", Reals())
    y = Variable("y", Reals())

    assert -1 * x == exact(-x)
    assert x * (-1) == exact(-x)
    assert -x * -y == exact(x * y)
    assert -x * y == exact(-(x * y))
    assert x * -y == exact(-(x * y))


def test_auto_condense_zero_out():
    x = Variable("x", Reals())

    assert 0 * x == exact(0)
    assert x * 0 == exact(0)

    assert x**0 == exact(1)


def test_auto_condense_powers():
    x = Variable("x", Reals())
    y = Variable("y", Reals())

    assert x * x == exact(x**2)
    assert x + x == exact(2 * x)
    assert x**2 + x**2 == exact(2 * x**2)
    assert 0.5 * x**2 + 1.5 * x**2 == exact(2 * x**2)


def test_auto_condense_nested_powers():
    x = Variable("x", Reals())

    assert (x**2) ** 4 == exact(x**8)
