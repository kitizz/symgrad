# Copyright

from .multiply import Multiply
from ..exact import exact_seq
from ..constant import Constant
from ..variable import Variable
from ..set import Set


class MySet(Set):
    ...


class MyOne(Set):
    @classmethod
    def _supersets(cls) -> dict[type[Set], tuple]:
        return {MySet: ()}


# BinaryRule.register(
#     Multiply,
#     args=(MySet(), MySet()),
#     out=MySet(),
#     identity=MyOne,
#     commutative=True,
#     associative=True,
# )


def test_autosort_multiply_simple():
    x = Variable("x", MySet())
    y = Variable("y", MySet())
    c = Constant(5, MySet())

    assert (x * y * c)._extract_chain() == exact_seq([c, x, y])
    assert (c * x * y)._extract_chain() == exact_seq([c, x, y])
    assert (c * y * x)._extract_chain() == exact_seq([c, x, y])
