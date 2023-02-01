from __future__ import annotations

import logging

from symgrad.expression import Expression
from symgrad.manipulation.expression_optimizer import ExpressionOptimizer
from symgrad.manipulation.metrics import factorization, variable_complexity
from symgrad.operators.binary_operator import BinaryOperator
from symgrad.operators.unary_operator import UnaryOperator
from symgrad.rules.knowledgebase import the_knowledgebase
from symgrad.shared_context import no_auto_simplify


def auto_simplify(expr: Expression) -> Expression:
    # Goals:
    #  - Maintain factorization
    #  - Minimize complexity
    #  - Minimize constant_depths

    # Disable operators from trying to run this routine while we're already in it,
    # since we're pulling apart and reconstructing operators here.
    kbase = the_knowledgebase()

    def minimize(expr: Expression):
        return variable_complexity(expr)

    clusterer = ExpressionOptimizer(kbase, minimize, cluster_func=factorization)
    with no_auto_simplify(), kbase.block_writes():
        return clusterer.minimize(expr)


def _auto_simplify(expr: Expression, clusterer: ExpressionOptimizer) -> Expression:
    """
    The Exact class is used as a way to cache the results of Expressions without
    needing to muddying the implementations of __eq__ or __hash__ for the
    Expression type.
    """
    # Try to simplify the deeper branchs first.
    match expr:
        case BinaryOperator():
            op = type(expr)
            a_simp = _auto_simplify(expr.a, clusterer)
            b_simp = _auto_simplify(expr.b, clusterer)
            expr = op(a_simp, b_simp)
        case UnaryOperator():
            op = type(expr)
            expr = op(_auto_simplify(expr.a, clusterer))

    # .eval with no args will combine Constants where possible.
    return clusterer.minimize(expr)
