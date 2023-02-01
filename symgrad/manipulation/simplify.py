# Copyright, Christopher Ham, 2022

import functools

from ..binary_operator import *
from ..expression import Expression
from ..symbol import *
from ..unary_operator import *


def simplify(gen: Expression):
    """Rearrange order of ops where possible and look for simplificatons based on special_cases."""
    return gen

    # Before we continue, we try to recursively simplify the child generators.
    match gen:
        case BinaryOperator(a, b):
            op = gen.__class__
            gen = op(simplify(a), simplify(b))
        case UnaryOperator(a):
            op = gen.__class__
            gen = op(simplify(a))
        case Symbol():
            # Can't simplify any further.
            return gen
        case _:
            raise ValueError(f"Not sure what to do with type {op}")

    match gen:
        case Multiply():
            num_chain, den_chain = extract_associative_chain(gen, Multiply)
            num_chain, den_chain = _cancel_terms(num_chain, den_chain, Multiply, Divide, One())
            return _reconstitute(num_chain, den_chain, Multiply, Divide, One())

        case Add():
            num_chain, den_chain = extract_associative_chain(gen, Add)
            num_chain, den_chain = _cancel_terms(num_chain, den_chain, Add, Subtract, Zero())
            return _reconstitute(num_chain, den_chain, Add, Subtract, Zero())

        case _:
            # print(f"No rule to simplify: {str(gen)}")
            return gen


# Define how a non-associative BinaryOperator can be converted into associative one.
# Given a key-value pair like: BinOpAss: (BinOpNonAss, UnaryOp), then:
#   BinOpAss(a, b) = BinOpNonAss(a, UnaryOp(b))
ASSOCIATIVE_MAP = {}
#     Add: (Subtract, Neg),
#     Multiply: (Divide, ElementWiseInverse),
# }


def extract_associative_chain(
    g: Expression, associative_operator: BinaryOperator | None = None
) -> tuple[list[Expression], list[Expression]]:
    """Recursively process nested operations to extract lists of the numerator and denominator.

    NOTE: Currently only works for Generators returning a Real result type.

    Requires:
     - associative_operator is None or a valid key in ASSOCIATIVE_MAP (Add, Multiply)

    Some examples:
        (a * b) / (c * d) -> ([a, b], [c, d])
        (a * b) / (c / d) -> ([a, b, d], [c])
        (a / b) * (c / d) -> ([a, c], [b, d])
    """
    return ([g], [])
    if associative_operator is None:
        for ass_op, (nonass_op, unary) in ASSOCIATIVE_MAP.items():
            if isinstance(g, ass_op) or isinstance(g, nonass_op):
                associative_operator = ass_op
                break
        else:
            # No chain can be determined for the input generator.
            return ([g], [])

    inv_operator = ASSOCIATIVE_MAP[associative_operator][0]

    match g:
        case Operator(result=Expression() as result) if not isinstance(result, Real):
            # This is the safe bet for now to avoid incorrect re-associations with Matrices.
            return [g], []
        case associative_operator(a, b):
            num_a, den_a = extract_associative_chain(a, associative_operator)
            num_b, den_b = extract_associative_chain(b, associative_operator)
            return num_a + num_b, den_a + den_b
        case inv_operator(a, b):
            num_a, den_a = extract_associative_chain(a, associative_operator)
            den_b, num_b = extract_associative_chain(b, associative_operator)
            return num_a + num_b, den_a + den_b
        case _:
            return [g], []


def _reconstitute(
    numerator: list[Expression],
    denominator: list[Expression],
    operator: BinaryOperator,
    inv_operator: BinaryOperator,
    identity: Constant,
) -> Expression:
    """Reconstitue a chain of Generators as nested operations of `operator`.

    Example:
        [a, b, c, d], Multiply -> (((a * b) * c) * d)
    """
    return inv_operator.from_args(
        functools.reduce(operator.from_args, numerator, identity),
        functools.reduce(operator.from_args, denominator, identity),
    )


def _filter_split(filt, values):
    """Like filter, but store the unfiltered results in a separate list."""
    in_ = []
    out_ = []
    for v in values:
        in_.append(v) if filt(v) else out_.append(v)
    return in_, out_


def _cancel_terms(
    numerator: list[Expression],
    denominator: list[Expression],
    operator: BinaryOperator,
    inv_operator: BinaryOperator,
    identity: Constant,
) -> tuple[list[Expression], list[Expression]]:
    """Cancel out pairs of generators with the same name in the numerators and denominators.

    Requires:
     - operator and inv_operator are the inverse operations of each other. Examples:
       (Multiply, Divide) and (Add, Subtract)

    Ensures:
     - Constant terms are collected under the provided `operator`, `inv_operator` rules.

    Returns tuple (new_numerator, new_denominator)
    """
    # Collect all the constant terms together into (likely) a single Generator.
    num_const, num_var = _filter_split(lambda x: isinstance(x, Constant), numerator)
    den_const, den_var = _filter_split(lambda x: isinstance(x, Constant), denominator)
    const = _reconstitute(num_const, den_const, operator, inv_operator, identity)

    for i, a in enumerate(num_var):
        for j in range(i + 1, len(num_var)):
            b = num_var[j]
            if (a is None) or (b is None):
                continue
            res = operator.from_args(a, b)
            # print(f"Try: {a} {operator.op_chr} {b} -> {res}")
            if not isinstance(res, operator):
                # print(f"Maybe simplify {a} {operator.op_chr} {b} -> {res}")
                num_var[i] = res
                num_var[j] = None

    # Mark entries for removal by setting them to None to avoid changing the list layout.
    for i, a in enumerate(num_var):
        for j, b in enumerate(den_var):
            if (a is None) or (b is None):
                continue

            if a.name == b.name:
                num_var[i] = None
                den_var[j] = None
            else:
                res = inv_operator.from_args(a, b)
                if not isinstance(res, inv_operator):
                    print(f"Maybe simplify {a, b} -> {res}")
                    num_var[i] = res
                    den_var[j] = None

    # Remove the Nones, and make sure the "compressed" const result is in the numerator.
    num_var = [const] + list(filter(lambda x: x is not None, num_var))
    den_var = list(filter(lambda x: x is not None, den_var))
    return num_var, den_var
