from symgrad.operators.binary_operator import BinaryOperator
from symgrad.operators.unary_operator import UnaryOperator
from symgrad.set import Set
from symgrad.sets.numbers import Ints, Reals
from symgrad.sets.set_type import SetType
from symgrad.variable import Variable
from .match_sets import match_sets


class Foo(Set):
    a: int
    b: int


class Bar(Set):
    a: int

    @classmethod
    def _supersets(cls):
        return {Foo: (cls.a, cls.a)}


class FooTyped(Set):
    t: Set
    a: int


class MyBinOp(BinaryOperator):
    code = "MyBinOp({a}, {b})"


class MyUnOp(UnaryOperator):
    code = "MyUnOp({a})"


def test_literal_single():
    foo_set = Foo(1, 2)

    assert match_sets(foo_set, Foo(1, 2), {})
    assert match_sets(foo_set, Foo(2, 1), {}) == False


def test_symbols():
    M = Variable("M", Ints())
    N = Variable("N", Ints())

    set_params = {}
    assert match_sets(Foo(M, N), Foo(2, 1), set_params)
    assert set_params == {M: 2, N: 1}

    assert match_sets(Foo(N, N), Foo(2, 1), {}) == False

    set_params = {}
    assert match_sets(Foo(M, N), Foo(N, M), set_params)
    assert set_params == {M: N, N: M}

    assert match_sets(Foo(M, M), Foo(N, M), {}) == False


def test_unary_set_symbols_typed():
    T = Variable("T", SetType(Ints()))
    M = Variable("M", Ints())

    set_params = {}
    assert match_sets(FooTyped(T, M), FooTyped(Ints(), 4), set_params)
    assert set_params == {T: Ints(), M: 4}

    assert match_sets(FooTyped(T, 4), FooTyped(Reals(), 4), {}) == False


def test_unary_subsets():
    M = Variable("M", Ints())

    set_params = {}
    assert match_sets(Foo(M, M), Bar(3), set_params)
    assert set_params == {M: 3}
