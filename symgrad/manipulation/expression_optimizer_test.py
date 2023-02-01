from symgrad.expression import Expression
from symgrad.sets import Ints, Reals
from symgrad.shared_context import no_auto_simplify
from symgrad.variable import Variable

from .expression_optimizer import structure_hash


def test_structure_hash(basic_knowledgebase):
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    n = Variable("n", Ints())
    m = Variable("m", Ints())

    with no_auto_simplify():
        assert structure_hash(x + y) == structure_hash(y + x)
        assert structure_hash(x + x) == structure_hash(y + y)
        assert structure_hash(x + x) != structure_hash(y + x)

        assert structure_hash(x + y) != structure_hash(y + n)
        assert structure_hash(x + x) != structure_hash(n + n)
        assert structure_hash(x + x) != structure_hash(y + x)

        assert structure_hash(x + y) != structure_hash(y * x)
        assert structure_hash(x + y) != structure_hash(x * y)
        assert structure_hash((x + y) + y) != structure_hash(x + y)

        assert structure_hash((x + y) * (n - m)) == structure_hash((y + x) * (m - n))


def test_structure_hash_benchmark(basic_knowledgebase, benchmark):
    x = Variable("x", Reals())
    y = Variable("y", Reals())
    c1 = Expression.wrap(3.5)
    c2 = Expression.wrap(-2)

    def run():
        structure_hash((c1 * (-x)) * y)
        structure_hash(c1 + (c2 + x))
        structure_hash(0.5 * x**2 + 1.5 * x**2)

    benchmark(run)
