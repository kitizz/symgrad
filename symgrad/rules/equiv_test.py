import itertools
from collections.abc import Sequence

from .equiv import equiv
from ..expression import Expression
from ..operators import Add, BinaryOperator
from ..set import Set
from ..variable import Variable


class MyVec(Set):
    """Meant to be a little bit like a vector type."""

    size: int = 0


class MyNumber(Set):
    """Meant to be a little bit like a Real number."""

    ...


def test_equiv_mixed_comm(reset_state):
    # NOTE: See also chain_helpers_test.py

    # BinaryRule.register(
    #     MyOp, args=(MyVec(), MyVec()), out=MyVec(), associative=True, commutative=False
    # )
    # BinaryRule.register(
    #     MyOp, args=(MyVec(), MyNumber()), out=MyVec(), associative=True, commutative=True
    # )
    # BinaryRule.register(
    #     MyOp, args=(MyNumber(), MyNumber()), out=MyNumber(), associative=True, commutative=True
    # )

    x = Variable("x", MyVec())
    y = Variable("y", MyVec())
    u = Variable("u", MyNumber())
    v = Variable("v", MyNumber())

    def associative_perms(items: Sequence[Expression], op: type[BinaryOperator]):
        if len(items) == 2:
            yield op.apply(items[0], items[1])
        else:
            # Create new lists that combines each consecutive pair using op.
            # Recursively pass that the associative_perms.
            items = list(items)
            for i in range(len(items) - 1):
                new_items = items[:i] + [op.apply(items[i], items[i + 1])] + items[i + 2 :]
                for perm in associative_perms(new_items, op):
                    yield perm

    target_perm = (u + v) + (x + y)
    target_chain: list[Expression] = [u, v, x, y]

    for chain_perm in itertools.permutations(target_chain, r=len(target_chain)):
        first_vec = next(v for v in chain_perm if v.output_set is MyVec())
        # When the relative order of x and y is flipped, we expect the equivalence to be False
        # since the lack of MyVec commutivity gets in the way.
        xy_flipped = first_vec.name == "y"

        for item_perm in associative_perms(chain_perm, Add):
            # Also check that the equivalene is symmetric.
            if xy_flipped:
                assert item_perm != equiv(target_perm)
                assert target_perm != equiv(item_perm)
            else:
                assert item_perm == equiv(target_perm)
                assert target_perm == equiv(item_perm)
