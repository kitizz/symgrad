from __future__ import annotations

import logging
import pprint
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Self

from ..constant import Constant
from ..exact import exact
from ..expression import Expression
from ..operators import Add, BinaryOperator, Inverse, Multiply, Neg, Operator, Power, UnaryOperator
from ..set import Set
from ..variable import Variable
from . import abstract_tree as tree
from .abstract_tree_from_expression import abstract_tree_from_expression
from .lexer import tokenize
from .match_sets import match_sets
from .parse_tokens import parse_tokens

logger = logging.getLogger(__name__)
logger.setLevel("WARNING")


class MatcherError(Exception):
    ...


class FailedMatch(Exception):
    ...


@dataclass
class BinaryConstraint:
    operator: type[BinaryOperator]
    a_operator: type[UnaryOperator] | None = None
    b_operator: type[UnaryOperator] | None = None

    def match(self, expr: Expression) -> None | tuple[Expression, Expression]:
        """Returns whether expr is a BinaryOperator with arguments that follow this BinaryConstraint."""
        if not isinstance(expr, self.operator):
            return None

        if self.a_operator is None:
            a = expr.a
        elif isinstance(expr.a, self.a_operator):
            a = expr.a.a
        else:
            return None

        if self.b_operator is None:
            b = expr.b
        elif isinstance(expr.b, self.b_operator):
            b = expr.b.a
        else:
            return None

        return a, b

    def apply(self, a_value, b_value):
        if self.a_operator:
            a_value = self.a_operator._apply(a_value)
        if self.b_operator:
            b_value = self.b_operator._apply(b_value)
        return self.operator._apply(a_value, b_value)


def _default_unary_operators() -> dict[str, type[UnaryOperator]]:
    return {
        "-": Neg,
        "Neg": Neg,
        "Inverse": Inverse,
    }


def _default_binary_constraints() -> dict[str, BinaryConstraint]:
    return {
        "Add": BinaryConstraint(Add),
        "Multiply": BinaryConstraint(Multiply),
        "Power": BinaryConstraint(Power),
        "+": BinaryConstraint(Add),
        "*": BinaryConstraint(Multiply),
        "-": BinaryConstraint(Add, None, Neg),
        "/": BinaryConstraint(Multiply, None, Inverse),
        "**": BinaryConstraint(Power),
    }


@dataclass
class MatchResult:
    _expressions: dict[str, Expression] = field(default_factory=dict)

    _unary_operators: dict[str, type[UnaryOperator]] = field(default_factory=dict)
    _unary_subexprs: defaultdict[type[UnaryOperator], list[Expression]] = field(
        default_factory=lambda: defaultdict(list)
    )

    _binary_operators: dict[str, type[BinaryOperator]] = field(default_factory=dict)
    _binary_subexprs: defaultdict[type[BinaryOperator], list[Expression]] = field(
        default_factory=lambda: defaultdict(list)
    )

    #: Maps any Variables used as Set parameters to their matched values.
    _set_params: dict[Variable, Any] = field(default_factory=dict)

    def expression(self, label_or_pattern: str) -> Expression:
        if expr := self._expressions.get(label_or_pattern):
            return expr
        # Consider caching the result in the original sub-tree if this is too slow.
        return expression_from_tree(_tree_from_pattern(label_or_pattern), result=self, full=True)

    def constant(self, label: str) -> Constant:
        expr = self.expression(label)
        if not isinstance(expr, Constant):
            raise TypeError(f"Captured Expression for '{label}' is not Constant")
        return expr

    def unary_operator(self, label: str) -> type[UnaryOperator]:
        return self._unary_operators[label]

    def binary_operator(self, label: str) -> type[BinaryOperator]:
        return self._binary_operators[label]

    def set(self, label: str) -> Set:
        return self.expression(label).output_set

    def variable_map(self) -> dict[str, Expression]:
        """Mapping of variable names in the pattern to their matched Expressions."""
        return self._expressions.copy()

    def __eq__(self, other):
        if not isinstance(other, MatchResult):
            return NotImplemented
        self_expr = {key: exact(expr) for key, expr in self._expressions.items()}
        other_expr = {key: exact(expr) for key, expr in other._expressions.items()}
        if self_expr != other_expr:
            return False

        if self._unary_operators != other._unary_operators:
            return False

        if self._binary_operators != other._binary_operators:
            return False

        return True


@dataclass(kw_only=True)
class Constraints:
    #: Constrain terms in the pattern to match only when the candidate expression's
    #: output Set is a subset of the specified Set here.
    sets: dict[str, Set] = field(default_factory=dict)

    #: Each term that appears in this mapping is only matched if the captured
    #: sub-expression is of the same class (eg. Variable, Constant, Expression).
    types: dict[str, type[Expression] | tuple[type[Expression], ...]] = field(default_factory=dict)

    unary: dict[str, type[UnaryOperator]] = field(default_factory=_default_unary_operators)

    binary: dict[str, BinaryConstraint] = field(default_factory=_default_binary_constraints)


class Matcher:
    """

    Matching rules:
     - Brackets must be used to enclose all binary operators (no implied left-to-right order)
    """

    constraints: Constraints
    pattern_tree: tree.TreeNode
    constrained_tree: tree.TreeNode

    def __init__(self, pattern: Expression | str | tree.TreeNode):
        self.constraints = Constraints()

        if isinstance(pattern, str):
            self.pattern_tree = _tree_from_pattern(pattern)
        elif isinstance(pattern, Expression):
            self.pattern_tree = abstract_tree_from_expression(pattern)
            for var in pattern.variables.values():
                self.constrain_set(var.name, var.output_set)
        else:
            self.pattern_tree = pattern

        self.constrained_tree = evaluate_tree_constants(self.pattern_tree, self.constraints)

    def constrain_set(self, name: str, set_: Set) -> Self:
        self.constraints.sets[name] = set_
        return self

    def constrain_type(
        self, name: str, type_: type[Expression] | tuple[type[Expression], ...]
    ) -> Self:
        self.constraints.types[name] = type_
        return self

    def constrain_binary_operator(
        self, name: str, op: type[BinaryOperator] | BinaryConstraint | None
    ) -> Self:
        if op is None:
            try:
                self.constraints.binary.pop(name)
            except KeyError:
                ...
        elif isinstance(op, BinaryConstraint):
            self.constraints.binary[name] = op
        else:
            self.constraints.binary[name] = BinaryConstraint(op)

        self.constrained_tree = evaluate_tree_constants(self.pattern_tree, self.constraints)
        return self

    def constrain_unary_operator(self, name: str, op: type[UnaryOperator] | None) -> Self:
        if op is None:
            try:
                self.constraints.unary.pop(name)
            except KeyError:
                ...
        else:
            self.constraints.unary[name] = op

        self.constrained_tree = evaluate_tree_constants(self.pattern_tree, self.constraints)
        return self

    def pattern_captures(self) -> PatternCaptures:
        """Returns the tokens in the Matcher's pattern that will attempt to match and capture
        expressions and operators in an Expression"""
        if not hasattr(self, "_pattern_captures"):
            variables = set()
            unary_ops = set()
            binary_ops = set()

            for node in tree.depth_first(self.constrained_tree):
                if isinstance(node, tree.Term):
                    variables.add(node.name)
                elif isinstance(node, tree.Function):
                    if len(node.children) == 1:
                        unary_ops.add(node.name)
                    elif len(node.children) == 2:
                        binary_ops.add(node.name)
                    else:
                        raise NotImplementedError("Multi arg functions not yet supported")

            self._pattern_captures = PatternCaptures(
                list(variables), list(unary_ops), list(binary_ops)
            )

        return self._pattern_captures

    def match(self, expr: Expression, *, apply_constraints: bool = True) -> MatchResult | None:
        result = MatchResult()

        try:
            assert _match(self.constrained_tree, expr, result, self.constraints)
        except FailedMatch as e:
            logger.debug("Pattern: %s", self.constrained_tree)
            logger.debug(e)
            return None

        if apply_constraints and not self.passes_constraints(result):
            return None
        return result

    def match_parts(
        self, parts: dict[str, Expression | type[Operator]], *, apply_constraints: bool = True
    ) -> MatchResult | None:
        result = MatchResult()

        for sub_pattern_str, sub_expr in parts.items():
            if isinstance(sub_expr, Expression):
                sub_pattern = evaluate_tree_constants(
                    _tree_from_pattern(sub_pattern_str), self.constraints
                )
                for node in tree.depth_first(self.constrained_tree):
                    if sub_pattern != node:
                        continue
                    try:
                        assert _match(sub_pattern, sub_expr, result, self.constraints)
                    except FailedMatch as e:
                        logger.debug("Pattern: %s", self.constrained_tree)
                        logger.debug(e)
                        return None
                    break
                else:
                    raise KeyError(f"Unable to find associated sub-pattern for {sub_pattern_str}")
            elif issubclass(sub_expr, UnaryOperator):
                existing = result._unary_operators.setdefault(sub_pattern_str, sub_expr)
                if existing is not sub_expr:
                    return None
            elif issubclass(sub_expr, BinaryOperator):
                existing = result._binary_operators.setdefault(sub_pattern_str, sub_expr)
                if existing is not sub_expr:
                    return None
            else:
                assert False, "Unreachable"

        self._update_match_from_constraints(result)

        if apply_constraints and not self.passes_constraints(result):
            return None
        return result

    def passes_constraints(self, result: MatchResult) -> bool:
        if not check_set_constraints(result, self.constraints):
            return False
        if not check_type_constraints(result, self.constraints):
            return False
        if not check_operator_constraints(result, self.constraints):
            return False

        return True

    def reverse_match(self, expr: Expression) -> ReverseMatchResult | None:
        """
        Requires:
         - All terms in pattern have a Set constraint (see Matcher.constrain_set())

        Ensures:
         - All result.expression() will be Variables created from the Set
           constraints whose names will be unique from Variables in expr.
        """
        result = ReverseMatchResult()

        if not _reverse_match(expr, self.constrained_tree, result):
            logger.debug("Match failed.")
            return None

        self._update_match_from_constraints(result)

        # Generate result._expressions from the constrained Set info.
        for term in tree.unique_terms(self.constrained_tree):
            set_ = self.constraints.sets.get(term.name)
            if not set_:
                raise MatcherError(
                    f"No Set constraint found for pattern term '{term.name}'. "
                    "Required for reverse_match()."
                )

            # Avoid name collisions of the created pattern Variables and those already in expr.
            var_name = f"{term.name}_pattern"
            while var_name in expr.variables:
                var_name = f"{term.name}_{uuid.uuid4().hex[:4]}"
            result._expressions[term.name] = Variable(var_name, set_)

        if not check_reverse_set_constraints(result):
            logger.warning("Set constraints failed")
            return None
        if not check_operator_constraints(result, self.constraints):
            return None
        # TODO: These don't make sense until terms can be pinned down to specific values
        # with constraints.
        # if not check_type_constraints(result, self.constraints):
        #     logging.warning("Type constraints failed")
        #     return None

        return result

    def _update_match_from_constraints(self, result: MatchResult):
        """Make the MatchResult reflect any constraints that affect any operators in match pattern.

        This is most needed when MatchResult was created from partial matching
        of the match pattern, and hasn't been able to match up everything. This
        happens in match_reverse and match_parts.
        """
        pattern_captures = self.pattern_captures()

        for name in pattern_captures.unary_ops:
            unary_constraint = self.constraints.unary.get(name)
            if unary_constraint:
                result._unary_operators.setdefault(name, unary_constraint)

        for name in pattern_captures.binary_ops:
            binary_constraint = self.constraints.binary.get(name)
            if binary_constraint:
                result._binary_operators.setdefault(name, binary_constraint.operator)

        # TODO: pattern_captures.variables?


@dataclass
class PatternCaptures:
    variables: list[str]
    unary_ops: list[str]
    binary_ops: list[str]


@cache
def _tree_from_pattern(pattern: str) -> tree.TreeNode:
    tokens = tokenize(pattern)
    return parse_tokens(tokens, pattern)


def check_type_constraints(result: MatchResult, constraints: Constraints) -> bool:
    """Check that types of matched sub-expressions satisfy the constraints."""
    for name, expected_type in constraints.types.items():
        captured_expr = result.expression(name)
        if not isinstance(captured_expr, expected_type):
            logger.warning(
                "Type mismatch for '%s'. Expected %s got %s",
                name,
                expected_type,
                type(captured_expr),
            )
            return False
    return True


def check_set_constraints(result: MatchResult, constraints: Constraints) -> bool:
    """Check that the output_sets for match sub-expressions satisfy the constraints.
    
    Ensures:
     - result._set_params is updated in-place to capture any matched Set parameters.
    """
    for name, expr in result._expressions.items():
        pattern_set = constraints.sets.get(name)
        if not pattern_set:
            continue

        if not match_sets(pattern_set, expr.output_set, result._set_params):
            return False

    return True


def check_operator_constraints(result: MatchResult, constraints: Constraints) -> bool:
    """Check that matched BinaryOperator Expressions satisfy their respective
    constraints in unary_operators and binary_patterns.
    """
    for op_str, unary_op in result._unary_operators.items():
        for expr in result._unary_subexprs[unary_op]:
            constrained_op = constraints.unary.get(op_str)
            if constrained_op and type(expr) is not constrained_op:
                return False

    for op_str, binary_op in result._binary_operators.items():
        for expr in result._binary_subexprs[binary_op]:
            constrained_pattern = constraints.binary.get(op_str)
            if constrained_pattern and not constrained_pattern.match(expr):
                return False

    return True


def check_reverse_set_constraints(result: ReverseMatchResult) -> bool:
    for var, node in result._nodes.items():
        try:
            # Since result._expressions has been built from the initial constraints,
            # evaluating the full Expression from this sub-tree will give us
            # the true Set against which the Variable should be checked.
            expr = expression_from_tree(node, result)
        except KeyError as k:
            logger.debug("Unable to convert tree to expression (%s -> %s): %s", var, node, k)
            return False
        if not match_sets(expr.output_set, var.output_set, result._set_params):
            logger.debug("match_sets failed.")
            return False

    return True


#
# Ordinary Matching Logic
#


def _match(
    pattern_node: tree.TreeNode, expr: Expression, result: MatchResult, constraints: Constraints
) -> bool:
    """TODO: Doc"""
    match pattern_node:
        case tree.Function() as func:
            if len(func.children) == 1:
                return _match_unary(func, expr, result, constraints)
            elif len(func.children) == 2:
                return _match_binary(func, expr, result, constraints)
            else:
                raise MatcherError("Matcher currently only supports unary and binary functions.")

        case tree.Term() as term:
            # Check consistency of what the pattern's variable captures.
            if result._expressions.setdefault(term.name, expr) != exact(expr):
                raise FailedMatch(
                    f"Inconsistent matches for pattern variable, {term.name}. "
                    f"{expr} vs. {result._expressions[term.name]}"
                )
            return True

        case tree.Number() as number:
            if number.value != expr.eval():
                raise FailedMatch(f"Failed to match pattern constant, {number}, with expr, {expr}")
            return True

        case _:
            raise TypeError(f"Unhandled CompleteNode type, {type(pattern_node)}")


def _match_unary(
    func: tree.Function, expr: Expression, result: MatchResult, constraints: Constraints
) -> bool:
    assert len(func.children) == 1

    if isinstance(expr, Constant) and func._internal.constant is not None:
        if expr.value != func._internal.constant:
            raise FailedMatch(
                "Expression Constant != Pattern Unary's constant; "
                f"{expr.value} != {func._internal.constant}"
            )
        return True

    if not isinstance(expr, UnaryOperator):
        raise FailedMatch(f"Expected UnaryOp for {func}. Got {expr}")

    if not _match(func.children[0], expr.a, result, constraints):
        return False

    # Inspect the unary ops for consistency.
    if result._unary_operators.setdefault(func.name, type(expr)) is not type(expr):
        raise FailedMatch(
            f"Inconsistent UnaryOperator for {func.name}. "
            f"{type(expr)} vs {result._unary_operators[func.name]}"
        )
    result._unary_subexprs[type(expr)].append(expr)

    return True


def _match_binary(
    func: tree.Function, expr: Expression, result: MatchResult, constraints: Constraints
) -> bool:
    assert len(func.children) == 2

    if isinstance(expr, Constant) and func._internal.constant is not None:
        if expr.value != func._internal.constant:
            raise FailedMatch(
                "Expression Constant != Pattern Binary's constant; "
                f"{expr.value} != {func._internal.constant}"
            )
        return True

    if not isinstance(expr, BinaryOperator):
        raise FailedMatch(f"Expected BinaryOp for {func}. Got {expr}")

    # Unwrap expr based on how the operator is constrained.
    # For example, when "a - b" should be interpreted as (a + Neg(b))
    binary_constraint = constraints.binary.get(func.name)
    if binary_constraint and (binary_match := binary_constraint.match(expr)):
        expr_a, expr_b = binary_match
    else:
        expr_a = expr.a
        expr_b = expr.b

    f_a, f_b = func.children
    if not _match(f_a, expr_a, result, constraints) or not _match(f_b, expr_b, result, constraints):
        return False

    # Inspect the binary patterns for consistency.
    existing = result._binary_operators.setdefault(func.name, type(expr))
    if existing is not type(expr):
        raise FailedMatch(
            f"Inconsistent BinaryOperator for {func.name}. "
            f"{type(expr)} vs {result._binary_operators[func.name]}"
        )
    result._binary_subexprs[type(expr)].append(expr)

    return True


#
# Reverse Matching Logic.
#


@dataclass
class ReverseMatchResult(MatchResult):
    _nodes: dict[Variable, tree.TreeNode] = field(default_factory=dict)


def _reverse_match(expr: Expression, node: tree.TreeNode, result: ReverseMatchResult) -> bool:
    """TODO: Discuss why this is kept separate from _match logic for now."""
    match expr:
        case UnaryOperator():
            return _reverse_match_unary(expr, node, result)
        case BinaryOperator():
            return _reverse_match_binary(expr, node, result)
        case Variable():
            # Check consistency of what the pattern's variable captures.
            existing = result._nodes.setdefault(expr, node)
            if existing != node:
                logger.warning(f"Sub-tree not matched. {hash(node)} vs {hash(existing)}")
            return existing == node
        case Constant():
            return isinstance(node, tree.Number) and expr.value == node.value
        case _:
            assert False, f"Unsupported Expression type! {type(expr)}"


def _reverse_match_unary(
    expr: UnaryOperator, node: tree.TreeNode, result: ReverseMatchResult
) -> bool:
    if not isinstance(node, tree.Function) or len(node.children) != 1:
        return False

    # Inspect the unary ops for consistency.
    if result._unary_operators.setdefault(node.name, type(expr)) is not type(expr):
        logger.warning(f"Inconsistent unary op {node.name}")
        return False

    return _reverse_match(expr.a, node.children[0], result)


def _reverse_match_binary(
    expr: BinaryOperator, node: tree.TreeNode, result: ReverseMatchResult
) -> bool:
    if not isinstance(node, tree.Function) or len(node.children) != 2:
        return False

    # Inspect the binary patterns for consistency.
    existing = result._binary_operators.setdefault(node.name, type(expr))
    if existing is not type(expr):
        logger.warning(f"Inconsistent binary op {node.name}. {type(expr)} vs {existing}")
        return False

    n_a, n_b = node.children
    return _reverse_match(expr.a, n_a, result) and _reverse_match(expr.b, n_b, result)


#
# Converters
#


def expression_from_tree(
    pattern_node: tree.TreeNode, result: MatchResult, full=False
) -> Expression:
    """Given a MatchResult, reconstruct an Expression given a pattern tree.

    Ensures:
     - When full is True, all Expressions from result are substituted for
       terms in pattern_node; otherwise, Variables are created that match the
       term name and its Expression's output_set.
    """
    match pattern_node:
        case tree.Function() as func:
            if len(func.children) == 1:
                op_class = result.unary_operator(func.name)
                a = expression_from_tree(func.children[0], result, full)
                return op_class(a)
            elif len(func.children) == 2:
                op_class = result.binary_operator(func.name)
                a = expression_from_tree(func.children[0], result, full)
                b = expression_from_tree(func.children[1], result, full)
                return op_class(a, b)
            else:
                raise MatcherError("Matcher currently only supports unary and binary functions.")

        case tree.Term() as term:
            if full:
                return result.expression(term.name)
            else:
                return Variable(term.name, result.expression(term.name).output_set)

        case tree.Number() as number:
            return Expression.wrap(number.value)

        case _:
            raise TypeError(f"Unhandled CompleteNode type, {type(pattern_node)}")


def evaluate_tree_constants(pattern_node: tree.TreeNode, constraints: Constraints) -> tree.TreeNode:
    """Use the declared operators in constraints to evaluate possible node constant value.

    For example, consider the following abstract tree:
        -(a, -(5))
    If constraints declares "-" as a unary Neg operator, then the Function node
        -(5)
    can have its ._internal.constant set to -5.

    Further, if constraints declares "-" as a binary Add operator, with a Neg
    applied to the RHS, then we can restructure the tree as:
        +(a, -(-(5)))
    """
    match pattern_node:
        case tree.Function() as func:
            if len(func.children) == 1:
                a = evaluate_tree_constants(func.children[0], constraints)
                func = tree.Function(name=func.name, children=(a,))
                a_val = a._internal.constant
                if (a_val is not None) and (op := constraints.unary.get(func.name)):
                    func._internal.constant = op._apply(a_val)
                return func

            elif len(func.children) == 2:
                a, b = func.children
                func_name = func.name
                bin_constraint = constraints.binary.get(func.name)
                if bin_constraint:
                    # Restructure the tree to match the binary contraint pattern.
                    func_name = bin_constraint.operator.__name__
                    if bin_constraint.a_operator:
                        a = tree.Function(name=bin_constraint.a_operator.__name__, children=(a,))
                    if bin_constraint.b_operator:
                        b = tree.Function(name=bin_constraint.b_operator.__name__, children=(b,))

                a = evaluate_tree_constants(a, constraints)
                b = evaluate_tree_constants(b, constraints)
                func = tree.Function(name=func_name, children=(a, b))

                a_val = a._internal.constant
                b_val = b._internal.constant
                if (a_val is not None) and (b_val is not None) and bin_constraint:
                    func._internal.constant = bin_constraint.operator._apply(a_val, b_val)
                return func

            else:
                raise MatcherError("Matcher currently only supports unary and binary functions.")

        case tree.Term() as term:
            return term

        case tree.Number() as number:
            return number

        case _:
            raise TypeError(f"Unhandled CompleteNode type, {type(pattern_node)}")
