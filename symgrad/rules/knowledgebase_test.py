import logging
import random

from ..constant import Constant
from ..exact import exact
from ..expression import Expression
from ..sets import Ints, Reals, SetType
from ..shared_context import no_auto_simplify
from ..variable import Variable
from .knowledgebase import Knowledgebase
from .rule_matcher import RuleMatcher
from .matcher import Matcher
from ..set import Set
from ..operators import BinaryOperator, UnaryOperator


def test_default_operators():
    x = Variable("x", Reals())
    y = Variable("y", Reals())

    kbase = Knowledgebase()

    kbase.add(x + y, y + x)

    result = list(kbase.query(y * x))
    assert result == []


def test_can_query_each_added():
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    z = Variable("z", Reals())

    kbase = Knowledgebase()

    with no_auto_simplify():
        zero = Expression.wrap(0)

        equations = [
            (x + 0, x),
            (x + (-x), zero),
            (x + y, y + x),
            ((x + y) + z, x + (y + z)),
        ]

        # Add in all equations
        for lhs, rhs in equations:
            kbase.add(lhs, rhs)

        # Query both the LHS and RHS to check bidirectionality.
        for lhs, rhs in equations:
            for match in list(kbase.query(lhs)):
                if match.lhs != exact(lhs):
                    continue
                assert match.rhs == exact(rhs)
                assert match.lhs.sub(match.mapping) == exact(lhs)
                break
            else:
                assert False, "No matches returned the queried LHS"

            for match in list(kbase.query(rhs)):
                if match.lhs != exact(rhs):
                    continue
                assert match.rhs == exact(lhs)
                assert match.lhs.sub(match.mapping) == exact(rhs)
                break
            else:
                assert False, "No matches returned the queried RHS"


def test_query_benchmark(benchmark):
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    z = Variable("z", Reals())
    n = Variable("n", Ints())
    m = Variable("m", Ints())

    with no_auto_simplify():
        zero = Expression.wrap(0)
        one = Expression.wrap(1)

        equations = [
            (x + 0, x),
            (x + (-x), zero),
            (x + y, y + x),
            ((x + y) + z, x + (y + z)),
            (x * 1, x),
            (x / x, one),  # TODO: Specify non-zero
            (x * y, y * x),
            ((x * y) * z, x * (y * z)),
            (x * (y + z), x * y + x * z),
            (x**1, x),
            (x**0, one),
            (x ** (n + m), x**n * x**m),
            ((x * y) ** n, x**n * y**n),
        ]

    # Add in all equations
    kbase = Knowledgebase()
    for lhs, rhs in equations:
        kbase.add(lhs, rhs)

    random.seed(1337)

    def query():
        eq = random.choice(equations)
        expr = random.choice(eq)
        matches = list(kbase.query(expr))
        assert len(matches) > 0

    benchmark(query)


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


def test_unary_set_literal_single():
    kbase = Knowledgebase()

    foo_set = Foo(1, 2)
    kbase.add_unary_set_rule(MyUnOp, foo_set, foo_set)

    assert kbase.query_unary_set_rule(MyUnOp, foo_set) is foo_set
    assert kbase.query_unary_set_rule(MyUnOp, Foo(2, 1)) is None


def test_binary_set_literal_single():
    foo_set = Foo(1, 2)
    bar_set = Bar(5)

    kbase = Knowledgebase()
    kbase.add_binary_set_rule(MyBinOp, foo_set, bar_set, foo_set)

    assert kbase.query_binary_set_rule(MyBinOp, foo_set, bar_set) is foo_set
    assert kbase.query_binary_set_rule(MyBinOp, Foo(2, 1), bar_set) is None
    assert kbase.query_binary_set_rule(MyBinOp, bar_set, foo_set) is None


def test_unary_set_symbols():
    M = Variable("M", Ints())
    N = Variable("N", Ints())
    O = Variable("O", Ints())
    a_set = Foo(M, N)
    out_set = Foo(N, M)

    kbase = Knowledgebase()
    kbase.add_unary_set_rule(MyUnOp, a_set, out_set)

    assert kbase.query_unary_set_rule(MyUnOp, a_set) is out_set
    assert kbase.query_unary_set_rule(MyUnOp, Foo(2, 3)) is Foo(3, 2)
    assert kbase.query_unary_set_rule(MyUnOp, Foo(2, N)) is Foo(N, 2)


def test_unary_set_symbols_square():
    M = Variable("M", Ints())
    a_set = Foo(M, M)
    out_set = Foo(M, M)

    kbase = Knowledgebase()
    kbase.add_unary_set_rule(MyUnOp, a_set, out_set)

    assert kbase.query_unary_set_rule(MyUnOp, a_set) is out_set
    assert kbase.query_unary_set_rule(MyUnOp, Foo(2, 2)) is Foo(2, 2)
    assert kbase.query_unary_set_rule(MyUnOp, Foo(2, 3)) is None


def test_unary_set_symbols_typed():
    T = Variable("T", SetType())
    M = Variable("M", Ints())
    a_set = FooTyped(T, M)
    out_set = FooTyped(T, M)
    # concrete = (FooTyped(Ints(), 3), FooTyped(Ints(), 3))

    kbase = Knowledgebase()
    kbase.add_unary_set_rule(MyUnOp, a_set, out_set)

    assert kbase.query_unary_set_rule(MyUnOp, a_set) is out_set
    assert kbase.query_unary_set_rule(MyUnOp, FooTyped(Ints(), 3)) is FooTyped(Ints(), 3)


def test_binary_set_symbols():
    M = Variable("M", Ints())
    N = Variable("N", Ints())
    O = Variable("O", Ints())
    a_set = Foo(M, N)
    b_set = Foo(N, O)
    out_set = Foo(M, O)

    kbase = Knowledgebase()
    kbase.add_binary_set_rule(MyBinOp, a_set, b_set, out_set)

    assert kbase.query_binary_set_rule(MyBinOp, a_set, b_set) is out_set
    assert kbase.query_binary_set_rule(MyBinOp, Foo(2, 3), Foo(3, 4)) is Foo(2, 4)
    assert kbase.query_binary_set_rule(MyBinOp, Foo(2, 3), Foo(3, N)) is Foo(2, N)
    assert kbase.query_binary_set_rule(MyBinOp, Foo(2, 3), Foo(4, 3)) is None


def test_unary_subsets():
    M = Variable("M", Ints())
    a_set = Foo(M, M)
    out_set = Foo(M, M)

    kbase = Knowledgebase()
    kbase.add_unary_set_rule(MyUnOp, a_set, out_set)

    assert kbase.query_unary_set_rule(MyUnOp, a_set) is out_set
    assert kbase.query_unary_set_rule(MyUnOp, Bar(3)) is Foo(3, 3)


def test_binary_subsets():
    M = Variable("M", Ints())
    a_set = Foo(M, M)
    out_set = Foo(M, M)

    kbase = Knowledgebase()
    kbase.add_binary_set_rule(MyBinOp, Reals(), Reals(), Reals())

    assert kbase.query_binary_set_rule(MyBinOp, a_set, a_set) is out_set
    assert kbase.query_binary_set_rule(MyBinOp, Bar(3), Bar(3)) is Foo(3, 3)
