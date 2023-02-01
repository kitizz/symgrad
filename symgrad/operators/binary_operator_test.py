# Copyright, Christopher Ham, 2022

import itertools
from typing import Generator, Sequence
import logging

import pytest

from ..exact import exact, exact_seq
from ..rules.equiv import equiv
from ..expression import Expression
from ..set import Set
from ..variable import Variable
from .binary_operator import BinaryOperator
from .unary_operator import UnaryOperator


class RootBinaryOp(BinaryOperator):
    code = ""
    subclasses = []

    def __init_subclass__(cls):
        cls.subclasses.append(cls)
        return super().__init_subclass__()

    @classmethod
    def _reset(cls):
        for subcls in cls.subclasses:
            subcls._rules.clear()
            subcls._power_rules.clear()
            subcls.find_rule.cache_clear()


class RootUnaryOp(UnaryOperator):
    code = ""
    subclasses = []

    def __init_subclass__(cls):
        cls.subclasses.append(cls)
        return super().__init_subclass__()

    @classmethod
    def _reset(cls):
        for subcls in cls.subclasses:
            subcls._rules.clear()


class MyOp(RootBinaryOp):
    code = "MyOp({a}, {b})"

    @classmethod
    def apply(cls, a, b, *, condense=True) -> "MyOp":
        expr = super().apply(a, b, condense=condense)
        assert isinstance(expr, MyOp)
        return expr


class InvOp(RootUnaryOp):
    code = "Inv({a})"


class MyVec(Set):
    """Meant to be a little bit like a vector type."""

    size: int = 0


class MyNumber(Set):
    """Meant to be a little bit like a Real number."""

    ...


@pytest.fixture(scope="function")
def reset_state():
    RootBinaryOp._reset()
    RootUnaryOp._reset()
    yield None
    RootBinaryOp._reset()
    RootUnaryOp._reset()


def vars(**kwargs: Set) -> tuple[Variable, ...]:
    return tuple(Variable(name, _set) for name, _set in kwargs.items())


def test_register(reset_state):
    (rule,) = BinaryRule.register(
        MyOp,
        args=(MyVec(), MyVec()),
        out=MyVec(),
    )

    assert rule.operator == MyOp
    assert rule.args == (MyVec(), MyVec())
    assert rule.out == MyVec()
    assert rule.commutative == False
    assert rule.associative == False
    assert rule.identity is None
    assert rule.inverse is None


def test_bad_add_rule(reset_state):
    class OtherOp(BinaryOperator):
        code = ""

    rule = BinaryRule(
        MyOp,
        args=(MyVec(), MyVec()),
        out=MyVec(),
    )

    with pytest.raises(TypeError):
        OtherOp.add_rule(rule)


def test_find_rule_basic(reset_state):
    (expected_rule,) = BinaryRule.register(
        MyOp,
        args=(MyVec(), MyVec()),
        out=MyVec(),
    )

    actual_rule = MyOp.find_rule(MyVec(), MyVec())

    assert expected_rule == actual_rule


def test_extract_chain_same(reset_state):
    assert MyOp._rules == []
    BinaryRule.register(
        MyOp,
        args=(MyVec(), MyVec()),
        out=MyVec(),
        associative=True,
    )
    assert len(MyOp._rules) == 1

    x, y, z = vars(x=MyVec(), y=MyVec(), z=MyVec())

    assert MyOp.apply(x, MyOp.apply(y, z))._extract_chain() == exact_seq([x, y, z])


def test_extract_chain_mixed_non_assoc(reset_state):
    BinaryRule.register(MyOp, args=(MyVec(), MyVec()), out=MyVec(), associative=True)
    BinaryRule.register(MyOp, args=(MyVec(), MyNumber()), out=MyVec(), associative=False)

    x, y, z = vars(x=MyVec(), y=MyVec(), z=MyNumber())

    assert MyOp.apply(x, MyOp.apply(y, z))._extract_chain() == exact_seq([x, MyOp.apply(y, z)])
    assert MyOp.apply(MyOp.apply(x, y), z)._extract_chain() == exact_seq([MyOp.apply(x, y), z])


def test_extract_chain_mixed_assoc(reset_state):
    BinaryRule.register(MyOp, args=(MyVec(), MyVec()), out=MyVec(), associative=True)
    BinaryRule.register(MyOp, args=(MyVec(), MyNumber()), out=MyVec(), associative=True)

    x, y, z = vars(x=MyVec(), y=MyVec(), z=MyNumber())

    assert MyOp.apply(x, MyOp.apply(y, z))._extract_chain() == exact([x, y, z])
    assert MyOp.apply(MyOp.apply(x, y), z)._extract_chain() == exact([x, y, z])
