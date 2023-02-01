import logging
import random

import pytest

from symgrad.constant import Constant
from symgrad.exact import exact
from symgrad.expression import Expression
from symgrad.operators import Add, BinaryOperator, Inverse, Multiply, Neg, Power
from symgrad.rules.knowledgebase import Knowledgebase
from symgrad.rules.matcher import Matcher
from symgrad.set import Set
from symgrad.sets import Ints, NonNegativeInts, Reals, SetType
from symgrad.shared_context import define_rules, no_auto_simplify, run_test_with_knowledgebase
from symgrad.variable import Variable


# Quickly define an Array type for use in tests.
class Array(Set):
    dtype: Set
    size: int


@pytest.fixture
def array_rules():
    kbase = Knowledgebase()

    n = Variable("n", Ints())
    m = Variable("m", Ints())
    u = Variable("u", Array(Reals(), n))
    v = Variable("v", Array(Reals(), m))

    with run_test_with_knowledgebase(kbase):
        with define_rules():
            (u + v) in Array(Reals(), n)

        yield kbase


def test_single():
    x = Variable("x", Ints())
    result = Matcher("a").match(x)

    assert result
    assert result.expression("a") == exact(x)


def test_add_plus():
    x = Variable("x", Ints())
    y = Variable("y", Ints())
    result = Matcher("a + b").match(x + y)

    assert result
    assert result.expression("a") == exact(x)
    assert result.expression("b") == exact(y)


def test_add_function():
    x = Variable("x", Ints())
    y = Variable("y", Ints())
    result = Matcher("Add(a, b)").match(x + y)

    assert result
    assert result.expression("a") == exact(x)
    assert result.expression("b") == exact(y)


def test_unmatched_defaults():
    x = Variable("x", Ints())
    y = Variable("y", Ints())
    z = Variable("z", Reals())

    # Intentionally choosing operators that don't match with defaults.
    assert Matcher("a + b").match(x * y) is None
    assert Matcher("a - b").match(x * y) is None
    assert Matcher("a * b").match(x + y) is None
    assert Matcher("a / b").match(x + y) is None
    assert Matcher("a ** b").match(x * y) is None
    assert Matcher("Neg(a)").match(Inverse(x)) is None
    assert Matcher("Inverse(a)").match(Neg(x)) is None


def test_unmatched_defaults_expr():
    x = Variable("x", Ints())
    y = Variable("y", Ints())
    z = Variable("z", Reals())

    # Intentionally choosing operators that don't match with defaults.
    assert Matcher(x + y).match(x * y) is None
    assert Matcher(x - y).match(x * y) is None
    assert Matcher(x * y).match(x + y) is None
    assert Matcher(x / y).match(x + y) is None
    assert Matcher(x**y).match(x * y) is None
    assert Matcher(Neg(y)).match(Inverse(x)) is None
    assert Matcher(Inverse(y)).match(Neg(x)) is None


def test_operator_wildcard():
    x = Variable("x", Ints())
    y = Variable("y", Ints())

    test_values = (
        (x + y, Add),
        (x * y, Multiply),
        (x**y, Power),
    )
    for expr, binary_operator in test_values:
        result = Matcher("a ? b").match(expr)

        assert result
        assert result.expression("a") == exact(x)
        assert result.expression("b") == exact(y)
        assert result.binary_operator("?") is binary_operator


def test_match_variables_one_to_one():
    x = Variable("x", Reals())

    result = Matcher(x).match(x)

    assert result and result.variable_map() == {"x": exact(x)}


def test_match_variables_constant():
    result = Matcher(Expression.wrap(2)).match(Expression.wrap(2))

    assert result and result.variable_map() == {}


def test_match_variables_one_to_many():
    x = Variable("x", Reals())

    result = Matcher(x).match(x**2)

    assert result
    matched = result.variable_map()
    assert matched == {"x": exact(x**2)}
    assert x.sub(matched) == exact(x**2)


def test_match_variables_no_match():
    x = Variable("x", Reals())

    result = Matcher(2 * x).match(x)

    assert result is None


# Maybe auto-generate a few...
def test_match_variables_multiple_consistency_simple():
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    with no_auto_simplify():
        pattern = x + x
        expr = 3 * y**2 + 3 * y**2

    result = Matcher(pattern).match(expr)
    assert result
    matched = result.variable_map()

    assert matched == {"x": exact(3 * y**2)}


def test_match_variables_multiple_consistency():
    x = Variable("x", Reals())
    y = Variable("y", Reals())

    with no_auto_simplify():
        pattern = x * (x + y)
        x_expected = 3 * y**2
        y_expected = y**3
        expr = x_expected * (x_expected + y_expected)

    result = Matcher(pattern).match(expr)
    assert result
    matched = result.variable_map()

    assert matched == {"x": exact(x_expected), "y": exact(y_expected)}


def test_match_variables_multiple_inconsistency():
    x = Variable("x", Reals())
    y = Variable("y", Reals())

    pattern = x * (x + y)
    x_expected = 3 * y**2
    y_expected = y**3
    expr = x_expected * (x_expected * 2 + y_expected)

    result = Matcher(pattern).match(expr)

    assert result is None


def test_match_variables_set_compatible():
    x = Variable("x", Reals())
    n = Variable("n", Ints())

    pattern = x**2
    expr = n**2

    result = Matcher(pattern).match(expr)
    assert result
    matched = result.variable_map()

    assert matched == {"x": exact(n)}


def test_match_variables_set_incompatible():
    x = Variable("x", Reals())
    n = Variable("n", Ints())

    pattern = n**2
    expr = x**2

    result = Matcher(pattern).match(expr)

    assert result is None


def test_match_str_neg_const():
    x = Variable("x", Reals())
    n = Variable("n", Ints())

    # Ok, so we have a problem with constants and negatives. The Matcher is trying to match
    # the value of 1, under a Subtract op. While the expr is a value of -1 under an Add op.
    # I think we need to use the constraints to guide the match.
    matcher = Matcher("(x * n) + (-1)")
    expected_match = {"x": x, "n": n}

    result = matcher.match(x * n - 1)
    assert result and result.variable_map() == expected_match

    result = matcher.match(x * n + (-1))
    assert result and result.variable_map() == expected_match
    

def test_match_neg():
    x = Variable("x", Reals())
    
    matcher = Matcher((-1) * x)
    expected_match = {"x": x}
    
    result = matcher.match(Neg(1) * x)
    assert result and result.variable_map() == expected_match
    
    result = matcher.match((-1) * x)
    assert result and result.variable_map() == expected_match


def test_match_variables_set_templates(array_rules):
    n = Variable("n", Ints())
    m = Variable("m", Ints())
    u_n = Variable("u_n", Array(Reals(), n))
    v_n = Variable("v_n", Array(Reals(), n))
    w_m = Variable("w_m", Array(Reals(), m))
    u = Variable("u", Array(Reals(), 3))
    v = Variable("v", Array(Reals(), 3))
    z = Variable("z", Array(Reals(), 2))

    result = Matcher(u_n + v_n).match(u + v)
    assert result
    matched = result.variable_map()
    assert matched == {"u_n": exact(u), "v_n": exact(v)}

    not_result = Matcher(u_n + v_n).match(u + z)
    assert not_result is None

    result = Matcher(u_n + w_m).match(u + z)
    assert result
    matched = result.variable_map()
    assert matched == {"u_n": exact(u), "w_m": exact(z)}


def test_match_subsets():
    x = Variable("x", Reals())
    n = Variable("n", Ints())
    m = Variable("m", NonNegativeInts())

    assert Matcher(x).match(x)
    assert Matcher(x).match(n)
    assert Matcher(x).match(m)


def test_reverse_match():
    x = Variable("x", Reals())
    n = Variable("n", Ints())
    m = Variable("m", Ints())

    matcher = Matcher("(a**1) ? (a**(n - 1))")
    matcher.constrain_set("a", Reals())
    matcher.constrain_set("n", NonNegativeInts())

    result = matcher.reverse_match(x**n * x**m)

    assert result
    assert result.expression("n") == exact(Variable("n_pattern", NonNegativeInts()))
    assert result.binary_operator("?") == Multiply


def test_match_parts():
    x = Variable("x", Reals())
    i = Variable("i", Ints())
    j = Variable("j", Ints())

    with no_auto_simplify():
        # a = 2 * (x + 4)
        # n = i - 3 * j
        a = Variable("a", Reals())
        n = Variable("n", Ints())
        full_expr = (a**n) * (a ** (n - 1))

    matcher = Matcher("(a**n) ? (a**(n - 1))")

    expected_match = matcher.match(full_expr)
    assert expected_match is not None

    sub_match1 = matcher.match_parts({"a**n": a**n, "?": Multiply})

    assert sub_match1 == expected_match


def benchmark_exprs():
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    n = Variable("n", Ints())
    m = Variable("m", Ints())

    with no_auto_simplify():
        return [
            (x + y) ** n,
            n,
            x * y,
            x * n,
            n + m,
            m + x,
            y**m,
            n**m,
            y - x,
            (x + y) * (m * y**n),
            (x * y + x * m) ** m,
        ]


def test_benchmark_hardcoded_key(benchmark):
    exprs = benchmark_exprs()

    # On M1 Max, this takes on average 0.3us to run
    def hardcoded_key() -> tuple[Set, Set, type[BinaryOperator], type[BinaryOperator]] | None:
        expr = random.choice(exprs)
        if not isinstance(expr, BinaryOperator):
            return None
        if not isinstance(expr.a, BinaryOperator):
            return None
        if not isinstance(expr.b, Constant) or not expr.b in NonNegativeInts():
            return None

        return (expr.a.a.output_set, expr.a.b.output_set, type(expr.a), type(expr))

    random.seed(123)
    benchmark(hardcoded_key)


def test_benchmark_match_key(benchmark):
    exprs = benchmark_exprs()

    matcher = Matcher("(a ? b) ?? n")
    matcher.constrain_type("n", Constant)
    matcher.constrain_set("n", NonNegativeInts())

    # On M1 Max, this takes on average 3us to run.
    # Hopefully this is fast enough for now, but I have some ideas for speeding this up:
    #  - Auto-generate code like the hardcoded example
    #  - Quicker bailouts and reduced init/copy of the Results, etc.
    def match_key() -> tuple[Set, Set, type[BinaryOperator], type[BinaryOperator]] | None:
        expr = random.choice(exprs)
        result = matcher.match(expr)
        if result is None:
            return None
        return (
            result.set("a"),
            result.set("b"),
            result.binary_operator("?"),
            result.binary_operator("??"),
        )

    random.seed(123)
    benchmark(match_key)
