# Copyright, Christopher Ham, 2022

import logging
import random

from ..operators import BinaryOperator, UnaryOperator
from ..operators import Multiply, Add, Power
from .expand import expand
from ..exact import exact, exact_unordered
from ..expression import Expression
from ..constant import Constant
from ..variable import Variable
from ..sets import Reals, Ints


def test_expand_pos():
    x = Variable("x", Reals())

    result = expand((x + 1) * (2 * x + 4))
    assert isinstance(result, Add)
    assert result._extract_chain() == exact_unordered((2 * x**2, 6 * x, 4))


def test_expand_neg():
    x = Variable("x", Reals())

    assert -(x + 10) == exact(-x - 10)


def test_expand_binomial_neg():
    x = Variable("x", Reals())

    result = expand((x + 1) * (-x - 4))
    assert isinstance(result, Add)
    assert result._extract_chain() == exact_unordered((-(x**2), -5 * x, -4))


def test_expand_double_quadratic():
    x = Variable("x", Reals())
    result = expand((x**2 + x + 1) * (0.5 * x**2 + 2 * x - 4))
    expected_chain = [
        0.5 * x**4,
        2.5 * x**3,
        -1.5 * x**2,
        -2 * x,
        -4,
    ]

    assert isinstance(result, Add)
    assert result._extract_chain() == exact_unordered(expected_chain)


def test_expand_power_addition():
    x = Variable("x", Reals())
    result = expand((x + 1) ** 2)

    assert isinstance(result, Add)
    assert result._extract_chain() == exact_unordered((x**2, 2 * x, 1))


def test_nested_powers():
    x = Variable("x", Reals())
    result = expand((-5 * ((9 * x) ** 4)) ** 2)

    assert isinstance(result, Multiply)
    assert result._extract_chain() == exact_unordered((x**8, 1076168025))


def test_expand_binomial_with_coefficients():
    x = Variable("x", Reals())
    result = expand((4 * x + 2 * x**3) ** 2)

    assert isinstance(result, Add)
    assert result._extract_chain() == exact_unordered((16 * x**2, 16 * x**4, 4 * x**6))


def test_no_auto_rearrangement():
    # As this expression is depth-first, recursively expanded, we make sure it doesn't
    # rearrange chains along the way. In this case, that behavior can hide the chain
    # from an outer multiplication. Example:
    #    (a + b) * ((a * 5)**2)
    #  = (a + b) * (a**2 * 25)
    #  = ((a + b) * a**2) * 25  # Unexpected (but equivalent) re-arranging of order
    #  = Multiply((a + b) * a, 5)
    # When the two terms of the last Multiply operation are inspected, it violates
    # the assumption of the algorithm that they are each fully expanded.
    a = Variable("a", Reals())
    b = Variable("b", Reals())
    result = expand((((a + b) * a**3) * b) * (((b * -3) ** 4) * 11))

    assert isinstance(result, Add)
    assert result._extract_chain() == exact_unordered(
        (891 * a**4 * b**5, 891 * a**3 * b**6)
    )


def test_expand_post_conditions():
    # Given a random expression, we expect the expanded version to have:
    # - No Addition ops in any children of Multiply ops
    # - No Multiply ops in children of Power ops.
    OPS = (Add, Multiply, Power)

    random.seed(1337)

    def verify(expr: Expression, ops: tuple[type[BinaryOperator], ...] = OPS):
        if isinstance(expr, UnaryOperator):
            return verify(expr.a, ops)
        elif isinstance(expr, BinaryOperator):
            if type(expr) not in ops:
                logging.warning(f"{type(expr)} not in {ops}")
                logging.warning(f"Expr: {expr}")
                return False
            subops = ops[ops.index(type(expr)) :]
            return verify(expr.a, subops) and verify(expr.b, subops)
        else:
            return True

    for i in range(100):
        expr = _generate_random_expression()
        expanded = expand(expr)
        assert verify(expanded)


def _generate_random_expression(sets=(Reals(), Ints()), max_depth=4):
    from ..operators import Multiply, Add, Power

    OPS: list[type[BinaryOperator]] = [Multiply, Add, Power]
    if max_depth > 0 and random.random() > 0.1:
        # Generate op
        op = random.choice(OPS)
        rules = [rule for rule in op._rules if any(rule.out in set for set in sets)]
        if rules:
            rule = random.choice(rules)
            # TODO: Consider subsets and various combinations that achieve allowed return types...
            a = _generate_random_expression(sets=(rule.args[0],), max_depth=max_depth - 1)
            if op is Power:
                b = Constant(random.randrange(2, 5))
            else:
                b = _generate_random_expression(sets=(rule.args[1],), max_depth=max_depth - 1)
            return op.apply(a, b)

    # Generate constant or variable
    if random.random() > 0.5:
        set = random.choice(sets)
        if set is Ints():
            name = random.choice(("a", "b", "c"))
        else:
            name = random.choice(("x", "y", "z"))
        return Variable(name, set)
    else:
        value = random.randrange(-40, 40) / 4
        set = random.choice(sets)
        if set is Ints():
            value = int(value)
        return Constant(value)
