# Copyright

from .add import Add
from ..exact import exact_seq
from ..constant import Constant
from ..variable import Variable
from ..set import Set


class MySet(Set):
    ...


class MyZero(Set):
    @classmethod
    def _supersets(cls) -> dict[type[Set], tuple]:
        return {MySet: ()}


# BinaryRule.register(
#     Add,
#     args=(MySet(), MySet()),
#     out=MySet(),
#     identity=MyZero,
#     commutative=True,
#     associative=True,
# )


def test_autosort_add():
    x = Variable("x", MySet())
    y = Variable("y", MySet())
    c = Constant(5, MySet())

    assert (x + y + c)._extract_chain() == exact_seq([x, y, c])
    assert (c + x + y)._extract_chain() == exact_seq([x, y, c])
    assert (c + y + x)._extract_chain() == exact_seq([x, y, c])
