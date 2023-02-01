import logging
import random

from symgrad.constant import Constant
from symgrad.exact import exact
from symgrad.expression import Expression
from symgrad.manipulation.extract_chain import extract_sortable_chain
from symgrad.manipulation.sort_chain import sort_chain
from symgrad.operators import Add, Multiply
from symgrad.operators.binary_operator import BinaryOperator
from symgrad.sets import Ints, Reals
from symgrad.variable import Variable

from .test_helpers import random_reduce


def test_sort_chain_variables(basic_knowledgebase):
    a = Variable("a", Reals())
    b = Variable("b", Reals())
    c = Variable("c", Ints())
    d = Variable("d", Reals())

    expected = [a, b, c, d]

    random.seed(9)
    for op in (Add, Multiply):
        for i in range(10):
            expr = random_reduce(op, expected, shuffle=True)
            assert extract_sortable_chain(sort_chain(expr)) == expected


def test_sort_chain_simple_powers(basic_knowledgebase):
    a = Variable("a", Reals())
    b = Variable("b", Reals())
    c = Variable("c", Ints())
    d = Variable("d", Reals())

    source_chain = [a**2, a, b**4, b**2, c**3, d**1.5]
    expected = [exact(v) for v in source_chain]

    random.seed(6)
    for op in (Add, Multiply):
        for i in range(10):
            expr = random_reduce(op, source_chain, shuffle=True)
            assert extract_sortable_chain(sort_chain(expr)) == expected


def test_sort_chain_polynomial(basic_knowledgebase):
    a = Variable("a", Reals())
    b = Variable("b", Reals())
    c = Variable("c", Ints())
    d = Variable("d", Reals())

    source_chain = [(a**2) * (b**3), a * c, b**4, b**2, (c**3) * (d**c), d**1.5]
    expected = [exact(v) for v in source_chain]

    random.seed(77)
    for i in range(10):
        args = []
        for arg in source_chain:
            if isinstance(arg, Multiply):
                arg_chain = extract_sortable_chain(arg)
                arg = random_reduce(Multiply, arg_chain, shuffle=True)
            args.append(arg)
        expr = random_reduce(Add, args, shuffle=True)

        assert extract_sortable_chain(sort_chain(expr)) == expected


def test_sort_constants_add(basic_knowledgebase):
    # Constants to right under addition.
    a = Variable("a", Reals())
    b = Variable("b", Reals())
    c = Variable("c", Ints())
    d = Variable("d", Reals())

    const4 = Expression.wrap(4.0)
    const_7 = Expression.wrap(-7)

    source_chain = [a, b, c, d, const4, const_7]
    expected = [exact(v) for v in source_chain]

    random.seed(11)
    for i in range(10):
        args = source_chain.copy()
        random.shuffle(args)
        expr = random_reduce(Add, args)

        sorted_chain = extract_sortable_chain(sort_chain(expr))

        assert sorted_chain[:4] == expected[:4]
        # Constants can be in any order, so just make sure the sorting is stable.
        expected_consts = [exact(arg) for arg in args if isinstance(arg, Constant)]
        assert sorted_chain[4:] == expected_consts


def test_sort_constants_multiply(basic_knowledgebase):
    # Constants to the left under multiplication.
    a = Variable("a", Reals())
    b = Variable("b", Reals())
    c = Variable("c", Ints())
    d = Variable("d", Reals())

    const4 = Expression.wrap(4.0)
    const_7 = Expression.wrap(-7)

    source_chain = [const4, const_7, a, b, c, d]
    expected = [exact(v) for v in source_chain]

    random.seed(11)
    for i in range(10):
        args = source_chain.copy()
        random.shuffle(args)
        expr = random_reduce(Multiply, args)

        sorted_chain = extract_sortable_chain(sort_chain(expr))

        assert sorted_chain[2:] == expected[2:]
        # Constants can be in any order, so just make sure the sorting is stable.
        expected_consts = [exact(arg) for arg in args if isinstance(arg, Constant)]
        assert sorted_chain[:2] == expected_consts


def deep_shuffle(expr: Expression) -> Expression:
    """Shuffle an Expression recursively all the way down the tree."""
    arg_chain = extract_sortable_chain(expr)
    if len(arg_chain) == 1:
        return expr
    assert isinstance(expr, BinaryOperator)

    arg_chain = [deep_shuffle(arg) for arg in arg_chain]
    consts = [arg for arg in arg_chain if arg.is_constant]
    random.shuffle(arg_chain)

    # Replace the constants in their original order.
    const_iter = iter(consts)
    arg_chain = [next(const_iter) if arg.is_constant else arg for arg in arg_chain]

    return random_reduce(type(expr), arg_chain)


def test_sort_chain_nested_neg(basic_knowledgebase):
    a = Variable("a", Reals())
    b = Variable("b", Reals())
    c = Variable("c", Ints())
    d = Variable("d", Reals())

    expected = a + (-b) + c + (-d)
    expected_chain = [exact(term) for term in extract_sortable_chain(expected)]
    assert extract_sortable_chain(sort_chain(a + c - (b + d))) == expected_chain


def test_sort_chain_roots(basic_knowledgebase):
    a = Variable("a", Reals())
    b = Variable("b", Reals())
    c = Variable("c", Ints())
    d = Variable("d", Reals())

    expected = (a - 5) * (b * b + 3) * (2 * b + 3) * (4 * b - 6) * (c * d + 5) * (2 * d - 1)
    # expected = (a - 5) * (2 * b * b + b) * (4 * b - 6) * (c + 5)
    expected_chain = [exact(term) for term in extract_sortable_chain(expected)]

    random.seed(99)
    for i in range(10):
        shuffled = deep_shuffle(expected)
        assert extract_sortable_chain(sort_chain(shuffled)) == expected_chain
