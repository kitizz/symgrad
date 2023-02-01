# Copyright, Christopher Ham, 2022

import itertools

from ..expression import Expression
from ..exact import exact, exact_seq
from ..constant import Constant
from ..variable import Variable
from ..set import Set
from .chain_helpers import combine_constants, equivalent_chains


class MyVec(Set):
    ...


class MyNumber(Set):
    ...


class FakeConstant:
    """FakeConstant class gives us full control of its behavior when manipulating
    Constants in the test below.
    """

    def __init__(self, *names: str):
        self.names = list(names)

    def combine(self, other: "FakeConstant") -> "FakeConstant":
        return FakeConstant(*(self.names + other.names))

    def __eq__(self, other):
        if isinstance(other, FakeConstant):
            return self.names == other.names
        return NotImplemented

    def __hash__(self):
        return hash((type(self).__name__,) + tuple(self.names))

    def __repr__(self):
        return f"FakeConstant({self.names})"


def vars(**kwargs: Set) -> tuple[Variable, ...]:
    return tuple(Variable(name, _set) for name, _set in kwargs.items())


def test_fake_constant_eq():
    assert FakeConstant("a") == exact(FakeConstant("a"))


def test_combine_constants_all():
    x, y = vars(x=MyVec(), y=MyVec())
    u, v = vars(u=MyNumber(), v=MyNumber())
    a = Constant(FakeConstant("a"), MyNumber())
    b = Constant(FakeConstant("b"), MyNumber())

    def can_swap(a: Expression, b: Expression):
        return a.output_set is not MyVec() or b.output_set is not MyVec()

    def reduce_constants(a: Constant, b: Constant):
        assert isinstance(a.value, FakeConstant) and isinstance(b.value, FakeConstant)
        return Constant(a.value.combine(b.value), MyNumber())

    seed_chain = [x, y, u, v, a, b]
    for chain_perm in itertools.permutations(seed_chain, r=len(seed_chain)):
        # The expected result is a and b combined, and all the way to the left.
        expected = list(chain_perm)
        order = [v for v in expected if isinstance(v, Constant)]
        expected.remove(order[0])
        expected.remove(order[1])
        expected.insert(0, reduce_constants(*order))

        assert combine_constants(chain_perm, can_swap, reduce_constants) == exact_seq(expected)


def test_combine_constants_noncommutative():
    x, y = vars(x=MyVec(), y=MyVec())
    u, v = vars(u=MyNumber(), v=MyNumber())
    a = Constant(FakeConstant("a"), MyVec())
    b = Constant(FakeConstant("b"), MyVec())

    def can_swap(a: Expression, b: Expression):
        return a.output_set is not MyVec() or b.output_set is not MyVec()

    def reduce_constants(a: Constant, b: Constant):
        assert isinstance(a.value, FakeConstant) and isinstance(b.value, FakeConstant)
        return Constant(a.value.combine(b.value), MyVec())

    a_b = reduce_constants(a, b)
    b_a = reduce_constants(b, a)

    def combine(chain):
        return combine_constants(chain, can_swap, reduce_constants)

    assert combine([x, y, u, v, a, b]) == exact_seq([x, y, a_b, u, v])
    assert combine([x, a, y, u, v, b]) == exact_seq([x, a, y, b, u, v])
    assert combine([a, x, y, u, v, b]) == exact_seq([a, x, y, b, u, v])
    assert combine([a, x, y, b, u, v]) == exact_seq([a, x, y, b, u, v])
    assert combine([x, a, b, y, u, v]) == exact_seq([x, a_b, y, u, v])
    assert combine([x, y, b, u, v, a]) == exact_seq([x, y, b_a, u, v])
    assert combine([x, b, u, v, y, a]) == exact_seq([x, b, u, v, y, a])
    assert combine([x, b, y, u, v, a]) == exact_seq([x, b, y, a, u, v])


def test_equivalent_chains():
    x, y = vars(x=MyVec(), y=MyVec())
    u, v = vars(u=MyNumber(), v=MyNumber())

    target_chain = [u, v, x, y]

    def is_equal(a: Expression, b: Expression):
        return a == exact(b)

    def can_swap(a: Expression, b: Expression):
        return a.output_set is not MyVec() or b.output_set is not MyVec()

    for chain_perm in itertools.permutations(target_chain, r=len(target_chain)):
        first_vec = next(v for v in chain_perm if v.output_set is MyVec())
        xy_flipped = first_vec.name == "y"

        # These chains should only be equivalent if x and y are in the same order.
        # When their order is flipped, their lack of commutivity stops the chains
        # from being able to be rearranged to match.
        # Further check that the function is symmetric.
        if xy_flipped:
            assert not equivalent_chains(chain_perm, target_chain, is_equal, can_swap)
            assert not equivalent_chains(target_chain, chain_perm, is_equal, can_swap)
        else:
            assert equivalent_chains(chain_perm, target_chain, is_equal, can_swap)
            assert equivalent_chains(target_chain, chain_perm, is_equal, can_swap)
