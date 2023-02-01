# Copyright, Christopher Ham, 2022

import functools
import itertools
import logging
import math
from typing import Sequence

from ..expression import Expression
from ..operators import BinaryOperator, UnaryOperator, DistributiveSide
from ..operators import BinaryRuleNotFoundError
from ..sets import Ints
from ..constant import Constant

__all__ = ["expand"]


class ExpandFailed(Exception):
    ...


def expand(expr: Expression) -> Expression:
    match expr:
        case BinaryOperator(a=a, b=b) as op:
            a_expanded = expand(a)
            b_expanded = expand(b)
            expr = type(op).apply(a_expanded, b_expanded, condense=False)
        case UnaryOperator(a=a) as op:
            expr = type(op).apply(expand(a))

    # In the process of expanding, its possible that Constant terms were combined.
    # So we separate the above and below match-cases.
    match expr:
        case BinaryOperator(a=a, b=b) as op:
            return expand_binary(expr)
        case UnaryOperator(a=a) as op:
            return expand_unary(expr)

    return expr


def _extract_inner_chain(
    outer_expr: BinaryOperator, inner_expr: Expression, direction: DistributiveSide
) -> list[Expression]:
    """Expand inner_expr's chain if the operation rules allow distribution from `direction`.

    Ensures:
     - The operator chain is extracted from inner_expr when the following conditions are met:
       - inner_expr is the BinaryOperator declared by outer_expr._rule.distributive_over
       - outer_expr is the BinaryOperator declared by inner_expr._rule.power
       - The distribution side of outer_expr._rule.distributive_over is "both" or
         matches the `direction` argument provided.
    """
    assert outer_expr._rule.distributive_over is not None
    inner_op, distr_direction = outer_expr._rule.distributive_over

    logging.debug("Extracting chain: %s", inner_expr)
    if not isinstance(inner_expr, BinaryOperator) or type(inner_expr) is not inner_op:
        # Only consider inner expressions that the outer expression can distribute over.
        logging.debug("Inner expr wrong type: %s", type(inner_expr))
        return [inner_expr]

    if inner_expr._rule.power is not type(outer_expr):
        # Avoid distributive rules where the power of the inner expression isn't the outer op.
        # Example: AND/OR are distributive over each other. But we only want to expand powers.
        logging.debug(
            "Inner rule's power not right: %s vs %s", inner_expr._rule.power, type(outer_expr)
        )
        return [inner_expr]

    if distr_direction != "both" and distr_direction != direction:
        # If the distribution side/direction isn't "both", make sure inner_expr is being
        # distributed from the same direction.
        logging.debug("Distr direction not matching: %s vs %s", distr_direction, direction)
        return [inner_expr]

    return inner_expr._extract_chain()


def expand_binary(expr: BinaryOperator) -> Expression:
    for expand_func in (expand_one_layer, expand_two_layer):
        try:
            return expand_func(expr)
        except ExpandFailed as e:
            logging.debug("Failed: %s", e)

    return expr


def expand_one_layer(expr: BinaryOperator) -> Expression:
    if expr._rule.distributive_over is None:
        raise ExpandFailed("No distribution rules")

    a_chain = _extract_inner_chain(expr, inner_expr=expr.a, direction="right")
    b_chain = _extract_inner_chain(expr, inner_expr=expr.b, direction="left")

    if len(a_chain) == 1 and len(b_chain) == 1:
        raise ExpandFailed("No chains found that satisfy distribution rules.")

    inner_op, _ = expr._rule.distributive_over
    outer_op = type(expr)
    try:
        return expand_chains(outer_op, inner_op, a_chain, b_chain)
    except BinaryRuleNotFoundError:
        logging.debug("Failed full expand.")
        ...

    # Failed to expand across both sides. Let's fall back to just expanding one
    # side. First try the right side, then the left side.
    # Consider: choosing the shortest one first (that's longer than 1).

    if len(a_chain) > 1 and len(b_chain) > 1:
        try:
            return expand_chains(outer_op, inner_op, [expr.a], b_chain)
        except BinaryRuleNotFoundError:
            logging.debug("Failed right-only")
            ...

        try:
            return expand_chains(outer_op, inner_op, a_chain, [expr.b])
        except BinaryRuleNotFoundError:
            logging.debug("Failed left-only")
            ...

    raise ExpandFailed("No expansions found that operate under valid BinaryRules.")


def multinomial_coefficient(n: int, ks: Sequence[int]):
    """See: https://en.wikipedia.org/wiki/Multinomial_theorem"""
    return math.factorial(n) // math.prod(math.factorial(k) for k in ks)


def iterate_valid_ks(n, m):
    """Where n is the power, and m is the number of terms in the multinomial."""
    for lst in itertools.product(range(n + 1), repeat=m):
        if sum(lst) == n:
            yield lst


def expand_two_layer(expr: BinaryOperator, *, term_limit=6, power_limit=10) -> Expression:
    if expr._rule.distributive_over is None:
        raise ExpandFailed("No distribution rules")

    middle_op, distr_direction = expr._rule.distributive_over

    # For now, this only supports integer powers operating from the right hand side.
    if distr_direction == "left":
        raise ExpandFailed("Only consider distributing to an exponential on the right.")
    if not isinstance(expr.b, Constant) or expr.b not in Ints():
        raise ExpandFailed("Exponent must be int")

    # The left hand side must be a BinaryOperator whose rule's power operator
    # matches the outer rule's distribute_over operator. Example:
    #   (x + y)**3, where x, y in Reals
    # The outer op (Power) with a Real LHS, can be distributed over Multiply.
    # And the LHS is Addition over Reals, whose BinaryRule's power operator is also Multiply.
    # Therefore, we can perform expansion on this expression.
    if not isinstance(expr.a, BinaryOperator) or expr.a._rule.power is not middle_op:
        raise ExpandFailed("LHS cannot by distributed over RHS exponent.")

    a_chain = expr.a._extract_chain()
    exponent = int(expr.b.value)
    assert len(a_chain) > 1
    assert exponent == expr.b.value

    power_op = type(expr)
    inner_op = type(expr.a)

    expanded_terms = []
    for ks in iterate_valid_ks(n=exponent, m=len(a_chain)):
        coeff = multinomial_coefficient(n=exponent, ks=ks)
        term = Expression.wrap(coeff)
        for a_term, k in zip(a_chain, ks):
            term = middle_op.apply(term, power_op.apply(a_term, k))
        expanded_terms.append(expand(term))

    return inner_op.reduce(expanded_terms)


def expand_chains(
    outer_op: type[BinaryOperator],
    inner_op: type[BinaryOperator],
    chain_a: Sequence[Expression],
    chain_b: Sequence[Expression],
) -> Expression:
    new_chain = []
    for a in chain_a:
        for b in chain_b:
            new_chain.append(outer_op.apply(a, b))
    assert len(new_chain) > 1
    return inner_op.reduce(new_chain)


def expand_unary(expr: UnaryOperator) -> Expression:
    raise NotImplementedError("Oh no!")
