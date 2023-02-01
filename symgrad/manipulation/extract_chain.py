import logging

from symgrad.cache import rule_cache
from symgrad.constant import Constant
from symgrad.expression import Expression
from symgrad.operators import BinaryOperator
from symgrad.rules.knowledgebase import the_knowledgebase
from symgrad.set import Set
from symgrad.variable import Variable

logger = logging.getLogger(__name__)
logger.setLevel("WARNING")

__all__ = [
    "extract_sortable_chain",
    "extract_associative_chain",
]


def extract_sortable_chain(expr: Expression) -> list[Expression]:
    """
    Ensures:
    - chain is list of sub-expressions in expr such that:
      - an equivalent of expr can be reconstructed from any permutation of chain with
        `reduce(type(expr), shuffle(chain)) == equiv(expr)`
      - every element has the same output_set
    """
    if not isinstance(expr, BinaryOperator):
        logger.debug("Not a BinaryOperator")
        return [expr]

    if expr.a.output_set not in expr.output_set or expr.b.output_set not in expr.output_set:
        logger.debug("One arg is not a subset of result Set")
        return [expr]

    if not is_commutative(type(expr), expr.output_set):
        logger.debug("Not commutative")
        return [expr]
    if not is_associative(type(expr), expr.output_set):
        logger.debug("Not associative: (%s, %s)", type(expr).__name__, expr.output_set)
        return [expr]

    return _operator_chain(type(expr), expr.a) + _operator_chain(type(expr), expr.b)


def extract_associative_chain(
    expr: Expression, op: type[BinaryOperator] = BinaryOperator
) -> list[Expression]:
    """Extract a chain of operands whose association can be changed under the root operator.

    Ensures:
    - chain is list of sub-expressions in expr such that:
      - an equivalent of expr can be reconstructed from the chain with
        `reduce(type(expr), chain) == equiv(expr)`
      - every element has the same output_set
    - a chain is only extract when expr is an instance of op.
    """
    if not isinstance(expr, op):
        logger.debug("Operator is not %s", op.__name__)
        return [expr]

    if expr.a.output_set not in expr.output_set or expr.b.output_set not in expr.output_set:
        logger.debug("One arg is not a subset of result Set")
        return [expr.a, expr.b]

    if not is_associative(type(expr), expr.output_set):
        logger.debug("Not associative: (%s, %s)", type(expr).__name__, expr.output_set)
        return [expr.a, expr.b]

    return _operator_chain(type(expr), expr.a) + _operator_chain(type(expr), expr.b)


def _operator_chain(op: type[BinaryOperator], expr: Expression) -> list[Expression]:
    if not isinstance(expr, op):
        return [expr]
    if expr.a.output_set not in expr.output_set or expr.b.output_set not in expr.output_set:
        return [expr]
    return _operator_chain(op, expr.a) + _operator_chain(op, expr.b)


@rule_cache
def is_commutative(op: type[BinaryOperator], set_: Set) -> bool:
    """Check if the elements of set_ are commutative under op."""
    x1 = Variable("x1", set_)
    x2 = Variable("x2", set_)

    kbase = the_knowledgebase()
    for match in kbase.query(op(x1, x2), rhs=op(x2, x1)):
        return True
    return False


@rule_cache
def is_associative(op: type[BinaryOperator], set_: Set) -> bool:
    """Check if the elements of set_ are associative under op."""
    x1 = Variable("x1", set_)
    x2 = Variable("x2", set_)
    x3 = Variable("x3", set_)

    kbase = the_knowledgebase()
    for match in kbase.query(op(op(x1, x2), x3), rhs=op(x1, op(x2, x3))):
        return True
    return False
