from ..sets import Reals, Ints
from ..shared_context import run_test_with_knowledgebase
from ..variable import Variable
from ..constant import Constant
from ..operators import Multiply
from .knowledgebase import Knowledgebase
from .rule_matcher import RuleMatcher


def test_knowledge_constraint_check():
    kbase = Knowledgebase()
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    z = Variable("z", Reals())

    kbase.add((x + y) + z, x + (y + z))

    with run_test_with_knowledgebase(kbase):
        # Should correctly capture the expression since the Knowledgebase knows
        # that it can be rearranged as requested.
        matcher = RuleMatcher("(a ? b) ? c")
        matcher.constrain_equal("a ? (b ? c)")
        result = matcher.match((x + y) + z)
        assert result is not None

        # Simple capture with no conditions.
        matcher = RuleMatcher("a ? b")
        result = matcher.match(x + y)
        assert result is not None

        # Should fail to match since the Knowledgebase has no record of this
        # operation being commutative.
        matcher.constrain_equal("b ? a")
        result = matcher.match(x + y)
        assert result is None


def test_foo():
    x = Variable("x", Reals())
    n = Variable("n", Ints())
    m = Variable("m", Ints())

    kbase = Knowledgebase()
    kbase.add(x ** (n + m), x**n * x**m)

    with run_test_with_knowledgebase(kbase):
        matcher = RuleMatcher("a ?? n")
        matcher.constrain_type("n", Constant)
        matcher.constrain_set("n", Ints())
        matcher.constrain_equal("(a ?? 1) ? (a ?? (n - 1))")

        result = matcher.match(x**3)
        assert result is not None
        assert result.binary_operator("?") == Multiply
