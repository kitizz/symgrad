import logging

from symgrad.exact import exact
from symgrad.expression import Expression
from symgrad.sets.numbers import Ints, Reals
from symgrad.variable import Variable

from .auto_simplify import auto_simplify


def test_no_change(basic_knowledgebase):
    x = Variable("x", Reals())
    y = Variable("y", Reals())

    assert auto_simplify(x) == exact(x)
    assert auto_simplify(x + y) == exact(x + y)
    assert auto_simplify(x - y) == exact(x - y)
    assert auto_simplify(x * y) == exact(x * y)
    assert auto_simplify(x**y) == exact(x**y)


def test_identities(basic_knowledgebase):
    x = Variable("x", Reals())

    assert auto_simplify(x + 0) == exact(x)
    assert auto_simplify(x * 1) == exact(x)
    assert auto_simplify(0 + x) == exact(x)
    assert auto_simplify(1 * x) == exact(x)


def test_group_constants(basic_knowledgebase):
    x = Variable("x", Reals())
    c1 = Expression.wrap(5.0)
    c2 = Expression.wrap(-2)

    c12 = (c1 + c2).eval()

    valid_configs = {
        exact(x + c12),
        exact(c12 + x),
    }

    assert auto_simplify((x + c1) + c2) == exact(x + c12)
    assert auto_simplify(x + (c1 + c2)) == exact(x + c12)
    assert auto_simplify((c1 + c2) + x) == exact(c12 + x)
    assert auto_simplify(c1 + (c2 + x)) == exact(c12 + x)

    # handle ambuiguity in the preferred order.
    assert exact(auto_simplify(c1 + (x + c2))) in valid_configs
    assert exact(auto_simplify((c1 + x) + c2)) in valid_configs


def test_negatives(basic_knowledgebase):
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    c1 = Expression.wrap(3.5)
    c1_neg = Expression.wrap(-3.5)

    assert auto_simplify(-1 * x) == exact(-x)
    assert auto_simplify(x * (-1)) == exact(-x)
    assert auto_simplify(-x * -y) == exact(x * y)
    assert auto_simplify((-x) * y) == exact((-x) * y)
    assert auto_simplify(c1 * (-x)) == exact(c1_neg * x)
    assert auto_simplify((c1 * (-x)) * y) == exact((c1_neg * x) * y)


def test_zero_out(basic_knowledgebase):
    x = Variable("x", Reals())

    assert auto_simplify(0 * x) == exact(0)
    assert auto_simplify(x * 0) == exact(0)
    assert auto_simplify(x**0) == exact(1)


def test_powers(basic_knowledgebase):
    x = Variable("x", Reals())
    y = Variable("y", Reals())

    assert auto_simplify(x * x) == exact(x**2)
    assert auto_simplify(x + x) == exact(2 * x)
    assert auto_simplify(x + 3 * x) == exact(4 * x)
    assert auto_simplify(-2 * x + x) == exact(-x)
    assert auto_simplify(x**2 + x**2) == exact(2 * x**2)
    assert auto_simplify(0.5 * x**2 + 1.5 * x**2) == exact(2.0 * x**2)


def test_nested_powers(basic_knowledgebase):
    x = Variable("x", Reals())

    assert auto_simplify((x**2) ** 4) == exact(x**8)


def test_auto_simplify_benchmark(basic_knowledgebase, benchmark):
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    c1 = Expression.wrap(3.5)
    c2 = Expression.wrap(-2)

    def run():
        auto_simplify((c1 * (-x)) * y)
        auto_simplify(c1 + (c2 + x))
        auto_simplify(0.5 * x**2 + 1.5 * x**2)

    benchmark(run)
