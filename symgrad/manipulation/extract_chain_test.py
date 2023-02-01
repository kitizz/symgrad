import logging
import random

import pytest

from symgrad.exact import exact
from symgrad.manipulation.extract_chain import (
    extract_associative_chain,
    extract_sortable_chain,
    is_commutative,
)
from symgrad.operators import Add, Multiply
from symgrad.set import Set
from symgrad.sets import Ints, Reals
from symgrad.shared_context import define_rules, no_auto_simplify
from symgrad.variable import Variable

from .test_helpers import random_reduce


class NonCommutative(Set):
    ...


@pytest.fixture(scope="module")
def sort_chain_kbase(basic_knowledgebase):
    x = Variable("x", Reals())
    nc1 = Variable("nc1", NonCommutative())
    nc2 = Variable("nc2", NonCommutative())
    nc3 = Variable("nc3", NonCommutative())

    with define_rules():
        (nc1 + nc2) in NonCommutative()
        (x + nc1) in Reals()
        (nc1 + x) in Reals()

        (nc1 + nc2) + nc3 == nc1 + (nc2 + nc3)

    with no_auto_simplify():
        yield basic_knowledgebase


def test_commutative(sort_chain_kbase):
    assert is_commutative(Add, Reals())
    assert is_commutative(Multiply, Reals())
    assert is_commutative(Add, Ints())
    assert is_commutative(Multiply, Ints())

def test_non_commutative(sort_chain_kbase):
    assert is_commutative(Add, NonCommutative()) == False


def test_commutative_cache(sort_chain_kbase, benchmark):
    def run():
        return is_commutative(Add, Reals())

    benchmark(run)


def test_sortable_chain(sort_chain_kbase):
    a = Variable("a", Reals())
    b = Variable("b", Reals())
    c = Variable("c", Ints())
    d = Variable("d", Reals())
    e = Variable("e", Reals())
    nc = Variable("nc", NonCommutative())

    # Mix Ints, Reals, and an element, "d + nc" which is in Reals, but can't be expanded further.
    elements = [a, b, c, d + nc, e]

    random.seed(4)
    for i in range(20):
        expr = random_reduce(Add, elements)
        assert extract_sortable_chain(expr) == elements


def test_not_sortable(sort_chain_kbase):
    a = Variable("a", Reals())
    nc = Variable("nc", NonCommutative())

    assert extract_sortable_chain(a + nc) == [exact(a + nc)]
    assert extract_sortable_chain(nc + a) == [exact(nc + a)]
    assert extract_sortable_chain(nc + nc) == [exact(nc + nc)]


def test_shallow(sort_chain_kbase):
    a = Variable("a", Reals())
    i = Variable("i", Ints())
    assert extract_sortable_chain(a + a) == [a, a]
    assert extract_sortable_chain(a + i) == [a, i]
    assert extract_sortable_chain(i + a) == [i, a]
    assert extract_sortable_chain(i + i) == [i, i]
