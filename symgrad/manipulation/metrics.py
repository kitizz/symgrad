import logging
import typing
from typing import TypeVar, Generic

from symgrad.set import Set
from ..expression import Expression
from ..rules.rule_matcher import RuleMatcher
from ..operators import BinaryOperator, UnaryOperator
from ..constant import Constant
from ..variable import Variable
from ..rules.matcher import MatchResult, Matcher
from ..sets import NonNegativeInts
from ..rules.knowledgebase import the_knowledgebase
from ..cache import cache, cache_single_expr
from ..rules.matcher_cache import matcher_cache

__all__ = [
    "complexity",
    "factorization",
    "constant_grouping",
]


def complexity(expr: Expression) -> tuple[int, int]:
    """TODO: Doc"""
    return (expr.operator_count, expr.operand_count)


@cache_single_expr
def variable_complexity(expr: Expression) -> tuple[int, int, int, int]:
    var_operator_count, var_operand_count = _var_complexity(expr)[:2]
    return (expr.operator_count, expr.operand_count, var_operator_count, var_operand_count)


def _var_complexity(expr: Expression) -> tuple[int, int, bool]:
    """Returns (non-const operator count, non-const operand count, is constant term)"""
    match expr:
        case BinaryOperator():
            a_operator_count, a_operand_count, a_is_constant = _var_complexity(expr.a)
            b_operator_count, b_operand_count, b_is_constant = _var_complexity(expr.b)
            operator_count = a_operator_count + b_operator_count
            operand_count = a_operand_count + b_operand_count
            is_constant = a_is_constant and b_is_constant
            if not is_constant:
                operator_count += 1
            return operator_count, operand_count, is_constant
        case UnaryOperator():
            # TODO: Figure out unary op setup
            operator_count, operand_count, is_constant = _var_complexity(expr.a)
            if not is_constant:
                operator_count += 1
            return operator_count, operand_count, is_constant
        case Variable():
            return (0, 2, False)
        case Constant():
            return (0, 1, True)
        case _:
            assert False


@cache_single_expr
def factorization(expr: Expression) -> int:
    """TODO: Doc"""
    match expr:
        case BinaryOperator():
            a_fact = factorization(expr.a)
            b_fact = factorization(expr.b)
            score = binary_factorization(expr)
            # is_const = a_is_const and b_is_const
            # logging.warning(f"fact({expr}) -> ({score + a_fact + b_fact}, {is_const})")
            return score + a_fact + b_fact
        case UnaryOperator():
            a_fact = factorization(expr.a)
            score = unary_factorization(expr)
            # logging.warning(f"fact({expr}) -> ({score + a_fact}, {a_is_const})")
            return score + a_fact
        case _:
            return 0


@cache_single_expr
def constant_grouping(expr: Expression) -> int:
    """Scores highest when more constant terms are grouped together.

    Examples:
        constant_grouping(5) > constant_grouping(a)
        constant_grouping((5 + 1) + a) > constant_grouping(5 + (1 + a))
        constant_grouping(5 + a) == constant_grouping(a + 5)
    """
    return _constant_depths(expr)


def _constant_depths(expr: Expression):
    match expr:
        case BinaryOperator():
            c_a = _constant_depths(expr.a)
            c_b = _constant_depths(expr.b)
            return 2 * (c_a + c_b)
        case UnaryOperator():
            return 2 * _constant_depths(expr.a)
        case Constant():
            return 1
        case _:
            return 0


def binary_factorization(expr: BinaryOperator) -> int:
    factorization = count_left_expandable_terms(expr) + count_right_expandable_terms(expr)
    if factorization > 0:
        return factorization
    else:
        return multinomial_factorization(expr)


def count_right_expandable_terms(expr: Expression) -> int:
    """For a Binary operation, how many terms on the right can be expanded out
    to the term on the left.

    Ensures:
     - Returns 0 when expr is not a BinaryOperator or if the right term can't be split.
     - Returns 0 when the RHS term is constant.
     - If the right term can be split, returns the number of terms that can be expanded.

    Examples:
        x * (a) -> 0
        x * (a + b) -> 2
        x * (a + (b + c)) -> 3
        (x + y) * (a + b) -> 2
        x * (1 + 5) -> 0
    """
    if not isinstance(expr, BinaryOperator):
        return 0
    if expr.b.is_constant:
        return 0
    num_terms = _count_right_terms(type(expr), expr.a, expr.b)
    return num_terms if num_terms > 1 else 0


def _count_right_terms(op: type[BinaryOperator], a: Expression, b: Expression) -> int:
    if can_distribute_from_left.from_parts(outer_op=op, x=a, a_b=b):
        assert isinstance(b, BinaryOperator)
        return _count_right_terms(op, a, b.a) + _count_right_terms(op, a, b.b)
    else:
        return 1


def count_left_expandable_terms(expr: Expression) -> int:
    """For a Binary operation, how many terms on the left can be expanded out
    to the term on the right.

    Ensures:
     - Returns 0 when expr is not a BinaryOperator or if the left term can't be split.
     - Returns 0 when the LHS term is constant.
     - If the left term can be split, returns the number of terms that can be expanded.

    Examples:
        (a) * x -> 0
        (a + b) * x -> 2
        (a + (b + c)) * x -> 3
        (x + y) * (a + b) -> 2
        (1 + 5) * x -> 0
    """
    if not isinstance(expr, BinaryOperator):
        return 0
    if expr.a.is_constant:
        return 0
    num_terms = _count_left_terms(type(expr), expr.a, expr.b)
    return num_terms if num_terms > 1 else 0


def _count_left_terms(op: type[BinaryOperator], a: Expression, b: Expression) -> int:
    if can_distribute_from_right.from_parts(outer_op=op, a_b=a, x=b):
        assert isinstance(a, BinaryOperator)
        return _count_left_terms(op, a.a, b) + _count_left_terms(op, a.b, b)
    else:
        return 1


@matcher_cache(
    matcher=RuleMatcher("x ?? (a ? b)").constrain_equal("(x ?? a) ? (x ?? b)"),
    failed_result=False,
    part_names={
        "outer_op": "??",
        "x": "x",
        "a_b": "a ? b",
    },
)
def can_distribute_from_left(match: MatchResult) -> bool:
    return True


@matcher_cache(
    matcher=RuleMatcher("(a ? b) ?? x").constrain_equal("(a ?? x) ? (b ?? x)"),
    failed_result=False,
    part_names={
        "outer_op": "??",
        "a_b": "a ? b",
        "x": "x",
    },
)
def can_distribute_from_right(match: MatchResult) -> bool:
    return True


@matcher_cache(pattern="a ?? n", failed_result=None)
def depower_operator(match: MatchResult) -> type[BinaryOperator] | None:
    maybe_power = match.binary_operator("??")
    a = Variable("a", match.set("a"))
    n = Variable("n", NonNegativeInts())
    m = Variable("m", NonNegativeInts())
    lhs = maybe_power(a, n + m)
    rhs_matcher = Matcher("(a ?? n) ? (a ?? m)")
    rhs_matcher.constrain_binary_operator("??", maybe_power)

    kbase = the_knowledgebase()
    for kb_match in kbase.query(lhs):
        depower_result = rhs_matcher.match(kb_match.rhs)
        if not depower_result:
            continue
        return depower_result.binary_operator("?")
    return None


def multinomial_factorization(expr: Expression) -> int:
    """For an expression matching (a + b + ...) ** n, return the number of
    terms on the right to the power of n.

    Notet that "n" must be a Constant, non-negative integer.
    Though + and ** can be any operators for which ** is the power-power
    operator of +.
    """
    if not is_multinomial(expr):
        return 0

    # TODO: Use key from previous result as shortcut here.
    depower = depower_operator(expr)
    if depower is None:
        return 0

    assert isinstance(expr, BinaryOperator)
    n = expr.b
    assert isinstance(n, Constant)
    num_left_terms = count_left_expandable_terms(depower(expr.a, expr.a))
    return num_left_terms**n.value


@matcher_cache(
    matcher=Matcher("(a ? b) ?? n")
    .constrain_set("n", NonNegativeInts())
    .constrain_type("n", Constant),
    failed_result=False,
)
def is_multinomial(match: MatchResult) -> bool:
    return True


def unary_factorization(expr: UnaryOperator, is_root=True) -> int:
    """Return the number of terms inside a UnaryOperator that could be expanded out.

    Examples:
      -(x + y) == (-x) + (-y) -> 2
      -(x + y + z) -> 3
      -(x) -> 0
    """
    if not can_distribute_unary(expr):
        # is_root=True means we're at the beginning of this recursion,
        # and so there's been no successful distribution.
        # If is_root=False then we're at a "leaf" in the recursion, so we want to
        # score this part as one factorized term.
        return 0 if is_root else 1

    assert isinstance(expr, UnaryOperator)
    assert isinstance(expr.a, BinaryOperator)

    a_score = _unary_factorization(type(expr), expr.a.a)
    b_score = _unary_factorization(type(expr), expr.a.b)
    return a_score + b_score


def _unary_factorization(op: type[UnaryOperator], arg: Expression) -> int:
    if not can_distribute_unary.from_parts(op=op, arg=arg):
        return 1
    assert isinstance(arg, BinaryOperator)
    a_score = _unary_factorization(op, arg.a)
    b_score = _unary_factorization(op, arg.b)
    return a_score + b_score


@matcher_cache(
    matcher=RuleMatcher("??(a ? b)").constrain_equal("(??a) ? (??b)"),
    failed_result=False,
    part_names={"op": "??", "arg": "a ? b"},
)
def can_distribute_unary(match: MatchResult) -> bool:
    return True
