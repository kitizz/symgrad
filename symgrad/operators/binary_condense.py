# TODO: Delete this file.

from __future__ import annotations

from typing import Any, TypeAlias
from collections.abc import Callable, Sequence
import functools
import logging

from .chain_helpers import CanSwap
from ..exact import exact
from ..expression import Expression
from ..constant import Constant

from . import binary_operator

BinaryRule: TypeAlias = "binary_operator.BinaryRule"
BinaryCondenseFunc: TypeAlias = Callable[
    [BinaryRule, Expression, Expression], Expression | tuple[Expression, Expression] | None
]

__all__ = [
    "condense_terms",
    "register_binary_condenser",
]


_condense_functions = []


def register_binary_condenser(func: BinaryCondenseFunc):
    """Register a function that tries to condense (collect) two terms under a BinaryOperator.

    Example:

        @register_binary_condenser
        def collect_constants(rule: BinaryRule, a: Expression, b: Expression) -> Expression | None:
            if isinstance(a, Constant) and isinstance(b, Constant):
                return Expression.wrap(rule.operator._apply(a.value, b.value))
    """
    if func.__qualname__ in [f.__qualname__ for f in _condense_functions]:
        raise RuntimeError(f"Unexpected re-registration of condenser, {func}")

    _condense_functions.append(func)


def condense_terms(
    chain: Sequence[Expression],
    can_swap: CanSwap,
    find_rule: Callable[[Expression, Expression], BinaryRule | None],
) -> list[Expression]:
    """Try to condense all pair-wise chain terms while respecting the rules of
    associativity and commutivity.

    Ensures:
     - Condense functions are applied in the order they were registered.
     - Each registered condense function is applied across the whole chain
       before considering the next one.
    """
    chain = list(chain)
    for condense_func in _condense_functions:
        chain = _condense_terms_with(chain, can_swap, find_rule, condense_func)
    return chain


def _condense_terms_with(
    chain: Sequence[Expression],
    can_swap: CanSwap,
    find_rule: Callable[[Expression, Expression], BinaryRule | None],
    condense: BinaryCondenseFunc,
) -> list[Expression]:
    """Try to condense all pair-wise chain terms while respecting the rules of
    associativity and commutivity.

    All combinations of the reordered chain are considered - respecting
    commutative rules. Each consecutive pair is passed to the condense function
    to see if they can be condensed/collected.
    """

    def condense_next(chain: Sequence[Expression]):
        logging.debug("condense_next: %s", chain)
        for i in range(len(chain)):
            for j in range(i + 1, len(chain)):
                # TODO: Do we need to test the condense function in reverse order?
                rule = find_rule(chain[i].output_set, chain[j].output_set)
                if rule is None:
                    continue
                result = condense(rule, chain[i], chain[j])
                if result is None:
                    continue

                shifted = _move_element(chain, can_swap, from_=j, to=i + 1)
                if shifted is None:
                    continue

                if isinstance(result, Expression):
                    shifted[i] = result
                    del shifted[i + 1]
                else:
                    shifted[i], shifted[i + 1] = result
                return shifted

        return None

    new_chain = list(chain)
    while True:
        next_chain = condense_next(new_chain)
        if next_chain is None:
            return new_chain
        new_chain = next_chain


def _move_element(
    orig_chain: Sequence[Expression], can_swap: CanSwap, from_: int, to: int
) -> list[Expression] | None:
    """
    Ensures:
     - If allowed by can_swap, the element as chain[from_] will be swapped with neighbors
       until it is at index "to"; the new list is returned.
     - If it cannot be done, None is returned.
    """
    chain = list(orig_chain)
    while from_ != to:
        if from_ > to:
            # Shift a_ind to the left.
            i = from_ - 1
            j = from_
            from_ -= 1
        else:
            # Shift a_ind to the right.
            i = from_
            j = from_ + 1
            from_ += 1

        if can_swap(chain[i], chain[j]):
            chain[i], chain[j] = chain[j], chain[i]
        else:
            return None
    return chain


@register_binary_condenser
def condense_constants(rule: BinaryRule, a: Expression, b: Expression) -> Expression | None:
    """TODO: Doc"""
    if isinstance(a, Constant) and isinstance(b, Constant):
        return Expression.wrap(rule.operator._apply(a.value, b.value))


def _disect_power(
    rule: BinaryRule, expr: Expression
) -> None | tuple[Expression, Constant, binary_operator.DistributiveSide]:
    """TODO: Doc"""
    assert rule and rule.power  # Programming names can be funny sometimes...

    if type(expr) is not rule.power:
        one = Expression.wrap(1)
        assert isinstance(one, Constant)
        return expr, one, "both"

    chain = expr._extract_chain()
    if isinstance(chain[0], Constant):
        base = rule.power.reduce(chain[1:])
        exp = chain[0]
        arg_types = (exp.output_set, base.output_set)
        exp_side = "left"
    elif isinstance(chain[-1], Constant):
        base = rule.power.reduce(chain[:-1])
        exp = chain[-1]
        arg_types = (base.output_set, exp.output_set)
        exp_side = "right"
    else:
        return None

    pow_rule = rule.power.find_rule(*arg_types)
    if pow_rule is None or pow_rule.distributive_over is None:
        return None

    if pow_rule.distributive_over[1] == "both":
        exp_side = "both"
    elif pow_rule.distributive_over[1] != exp_side:
        return None

    return base, exp, exp_side


@register_binary_condenser
def condense_powers(rule: BinaryRule, a: Expression, b: Expression) -> Expression | None:
    """When possible condense things like: a^2 * a^3 -> a^5"""
    if not rule.power:
        return None

    a_disect = _disect_power(rule, a)
    b_disect = _disect_power(rule, b)
    if a_disect is None or b_disect is None:
        return None

    a_base, a_exp, a_exp_side = a_disect
    b_base, b_exp, b_exp_side = b_disect

    if a_base != exact(b_base):
        return None
    if a_exp_side != "both" and b_exp_side != "both" and a_exp_side != b_exp_side:
        return None

    return rule.power.apply(a_base, a_exp + b_exp)


@register_binary_condenser
def condense_identity(rule: BinaryRule, a: Expression, b: Expression) -> Expression | None:
    """When possible, remove redundant Identity terms."""
    left_id_cls = rule.left_identity()
    if left_id_cls is not None and a == exact(left_id_cls(a.output_set)):
        return b

    right_id_cls = rule.right_identity()
    if right_id_cls is not None and b == exact(right_id_cls(b.output_set)):
        return a


@register_binary_condenser
def condense_zero_power(rule: BinaryRule, a: Expression, b: Expression) -> Expression | None:
    """If the rule operation is a valid power op for another rule, check for zeros.
    The result is the Identity of the other rule.
    """
    # While subtle, the valid sides of the Identity dictate where a power exponent
    # can exist. Therefore the other side must be the base arg type of the base rule.
    if a == exact(0) and rule.left_identity() is not None:
        input_set = b.output_set
    elif b == exact(0) and rule.right_identity() is not None:
        input_set = a.output_set
    else:
        return None

    # Try to find the base rule for which "rule" describes the power operator.
    for base_rule in rule.operator._power_rules:
        # Example: For input x**0, find the rule matching: x * x
        var = match_arg_patterns(base_rule.args, (input_set, input_set))
        if var is not None:
            # Found the base rule and operator!
            assert isinstance(base_rule.identity, type)
            return base_rule.identity(input_set)


@register_binary_condenser
def condense_neg_one_power(rule: BinaryRule, a: Expression, b: Expression) -> Expression | None:
    """If the rule operation is a valid power op for another rule, check for zeros.
    The result is the Identity of the other rule.
    """
    # While subtle, the valid sides of the Identity dictate where a power exponent
    # can exist. Therefore the other side must be the base arg type of the base rule.
    if a == exact(-1) and rule.left_identity() is not None:
        base_term = b
    elif b == exact(-1) and rule.right_identity() is not None:
        base_term = a
    else:
        return None

    # Try to find the base rule for which "rule" describes the power operator.
    for base_rule in rule.operator._power_rules:
        if base_rule.inverse is None:
            continue
        # Example: For input x**-1, find the rule matching: x * x
        var = match_arg_patterns(base_rule.args, (base_term.output_set, base_term.output_set))
        if var is not None:
            # Found the base rule and operator!
            return base_rule.inverse.apply(base_term)


@register_binary_condenser
def condense_nested_powers(rule: BinaryRule, a: Expression, b: Expression) -> Expression | None:
    """TODO: Doc"""
    if rule.distributive_over is None:
        return None

    # Look for multiple nested constants under the same operator.
    # We assume that at least a or b are not Constant terms by this point. Example:
    # ((x**2)**3)**4 -> x**24
    distr_op, distr_side = rule.distributive_over

    if distr_side != "left" and isinstance(b, Constant):
        base = a
        reduced_exponent = b
        right_exp = True
    elif distr_side != "right" and isinstance(a, Constant):
        base = b
        reduced_exponent = a
        right_exp = False
    else:
        return None

    # Dig down, collecting all repeated power operations.
    while type(base) is rule.operator:
        exp = base.b if right_exp else base.a
        if not isinstance(exp, Constant):
            break

        try:
            reduced_exponent = reduced_exponent * exp if right_exp else exp * reduced_exponent
        except binary_operator.BinaryRuleNotFoundError:
            # Incompatible types/sets found for multiply. Let's stop here.
            break

        base = base.a if right_exp else base.b

    return rule.operator.apply(base, reduced_exponent, condense=False)
