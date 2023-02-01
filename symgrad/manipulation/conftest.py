import pytest

from symgrad.rules.knowledgebase import Knowledgebase
from symgrad.rules.rule_collection import RuleCollection
from symgrad.sets.numbers import Ints, Reals
from symgrad.shared_context import define_rules, no_auto_simplify, run_test_with_knowledgebase
from symgrad.variable import Variable


@pytest.fixture(scope="module")
def basic_knowledgebase():
    kbase = Knowledgebase()

    x = Variable("x", Reals())
    y = Variable("y", Reals())
    z = Variable("z", Reals())
    n = Variable("n", Ints())
    m = Variable("m", Ints())

    with run_test_with_knowledgebase(kbase):
        class BasicRules(RuleCollection):
            @classmethod
            def axioms(cls):
                (x + y) in Reals()
                (x * y) in Reals()
                (x**y) in Reals()
                (-x) in Reals()

                (n + m) in Ints()
                (n * m) in Ints()
                (n**m) in Ints()
                (-n) in Ints()

                x + y == y + x
                x * y == y * x
                (x + y) + z == x + (y + z)
                (x + y) * z == x * z + y * z
                (x * y) ** n == x**n * y**n
                x ** (n + m) == (x**n) * (x**m)
                -(x + y) == (-x) + (-y)
                x + 0 == x
                x * 1 == x

                x - x == 0
                x * 0 == 0
                
            @classmethod
            def theorems(cls):
                x * (y + z) == x * y + x * z  # TODO: Derive automatically...
                
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

                # Notice that this implicitly follows the order of arguments.
                (x * y) + x == x * (y + 1)
                x + (x * y) == x * (y + 1)
                (y * x) + x == (y + 1) * x
                x + (y * x) == (y + 1) * x

        with no_auto_simplify():
            yield kbase
