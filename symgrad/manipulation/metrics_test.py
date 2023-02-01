import logging
import pytest

from symgrad.expression import Expression

from ..sets import Ints, Reals
from ..shared_context import no_auto_simplify, run_test_with_knowledgebase, define_rules
from ..variable import Variable
from ..rules.knowledgebase import Knowledgebase
from .metrics import can_distribute_from_left, complexity, factorization, constant_grouping


@pytest.fixture(scope="module")
def metrics_test_knowledgebase():
    kbase = Knowledgebase()

    x = Variable("x", Reals())
    y = Variable("y", Reals())
    z = Variable("z", Reals())
    n = Variable("n", Ints())
    m = Variable("m", Ints())

    with run_test_with_knowledgebase(kbase):
        with define_rules():
            (x + y) in Reals()
            (x * y) in Reals()
            (n + m) in Ints()
            (n * m) in Ints()
            (x**y) in Reals()
            (n**m) in Ints()
            (-x) in Reals()
            (-n) in Ints()

            x + y == y + x
            x * y == y * x
            (x + y) + z == x + (y + z)
            (x + y) * z == x * z + y * z
            x * (y + z) == x * y + x * z
            (x * y) ** n == x**n * y**n
            x ** (n + m) == (x**n) * (x**m)
            -(x + y) == (-x) + (-y)

            # TODO: Delete
            -(-x) == x
            -x == -1 * x
            (-x) * (-y) == x * y
            -(x * y) == x * (-y)
            -(x * y) == (-x) * y
            x**1 == x
            x**0 == 1
            x * x == x**2
            (x**n) * x == x ** (n + 1)
            x * (x**n) == x ** (n + 1)
            x + x == 2 * x
            (x**n) ** m == x ** (n * m)

        with no_auto_simplify():
            yield kbase


def test_complexity(metrics_test_knowledgebase):
    x = Variable("x", Ints())

    assert complexity(x) == (0, 1)
    assert complexity(x * x) == (1, 2)
    assert complexity(x**2) == (1, 2)
    assert complexity(-x) == (1, 1)
    assert complexity(-1 * x) == (1, 2)


def test_factorization_none(metrics_test_knowledgebase):
    x = Variable("x", Reals())

    assert factorization(x) == 0
    assert factorization(7 * x) == 0
    assert factorization(5 * x**3) == 0
    assert factorization(-(x**2)) == 0
    assert factorization(-5 * x**2) == 0
    assert factorization(-(5 * x**2)) == 0
    assert factorization(-(x**2) + 5 * x**3) == 0
    assert factorization(-(5 * x**2) + 4 * x**4) == 0


def test_factorization_one_layer(metrics_test_knowledgebase):
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    n = Variable("n", Ints())

    # Right-distributive
    assert factorization((5 + x) * 2) == 2
    assert factorization((5 + x) * x) == 2
    assert factorization((5 + x + y) * 2) == 3
    assert factorization((5 + x + y + x**2) * 2) == 4

    # Left-distributive
    assert factorization(2 * (5 + x)) == 2
    assert factorization(x * (5 + x)) == 2
    assert factorization(2 * (5 + x + y)) == 3
    assert factorization(2 * (5 + x + y + x**2)) == 4

    assert factorization((5 + x) * (y + 2)) == 4
    assert factorization((5 + x + y) * (y + 2)) == 5

    assert factorization((5 * x) ** 2) == 2
    assert factorization((5 * x) ** n) == 2
    assert factorization((5 * x * y) ** 2) == 3
    assert factorization((5 * x * y * x**2) ** 2) == 4


def test_factorization_nested(metrics_test_knowledgebase):
    x = Variable("x", Reals())
    y = Variable("y", Reals())

    assert factorization(x * (5 * (x + 1))) == 2
    assert factorization(x * (5 * (x + 1) + y)) == 4


def test_factorization_n_nomial(metrics_test_knowledgebase):
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    n = Variable("n", Ints())

    assert factorization((5 + x) * (2 * x - 2)) == 4
    assert factorization((5 + x) ** 2) == 4
    assert factorization((5 + x + y) ** 2) == 9

    assert factorization((5 + x) ** 0.5) == 0
    assert factorization((5 + x) ** -2) == 0
    assert factorization((5 + x) ** n) == 0


def test_factorization_unary(metrics_test_knowledgebase):
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    n = Variable("n", Ints())

    assert factorization(-x) == 0
    assert factorization(-(x + y)) == 2
    assert factorization(-(x + y + n)) == 3
    assert factorization(-(x * y + n)) == 2
    assert factorization(-(x * y)) == 0
    assert factorization(-(x - y)) == 2
    assert factorization(-(-x - y)) == 2


def test_factorization_constant(metrics_test_knowledgebase):
    x = Variable("x", Reals())
    c1 = Expression.wrap(1.5)
    c2 = Expression.wrap(0.5)

    assert factorization(c1 * x**2 + c2 * x**2) == 0
    assert factorization((c1 + c2) * (x**2)) == 0


def test_constant_depths(metrics_test_knowledgebase):
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    n = Variable("n", Ints())

    c_int = Expression.wrap(5)
    constant_grouping(c_int) > constant_grouping(x)
    constant_grouping((c_int + 1) + x) > constant_grouping(c_int + (1 + x))
    constant_grouping(c_int + (1 + x)) == constant_grouping(c_int + (x + 1))
    constant_grouping(c_int + x) == constant_grouping(x + c_int)


def test_can_distribute_benchmark(metrics_test_knowledgebase, benchmark):
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    n = Variable("n", Ints())

    expr = x * (y + n + 2 * x)

    # from ..rules.matcher import Matcher

    # matcher = Matcher("x ?? (a ? b)")

    def run():
        can_distribute_from_left(expr)

    benchmark(run)
