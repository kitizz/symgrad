import copy
from dataclasses import dataclass, field
from functools import cache
import logging
from typing import Self

from ..expression import Expression
from ..variable import Variable
from . import abstract_tree as tree
from . import matcher
from .equiv import is_equivalent
from .knowledgebase import Knowledgebase, the_knowledgebase
from .matcher import Matcher, MatchResult, expression_from_tree
from ..operators import BinaryOperator, UnaryOperator, BinaryRuleNotFoundError
from ..variable import Variable
from ..constant import Constant


@dataclass
class RuleConstraints:
    #: The match is constrained to Expressions that can be re-arranged from the
    #: original matching pattern to these equalities.
    equalities: list[tree.TreeNode] = field(default_factory=list)


class RuleMatcher(Matcher):
    """Introduces constraints to Matcher that make use of the Knowledgebase"""

    rule_constraints: RuleConstraints

    def __init__(self, pattern: Expression | str):
        super().__init__(pattern)
        self.rule_constraints = RuleConstraints()

    def constrain_equal(self, pattern: str) -> Self:
        self.rule_constraints.equalities.append(matcher._tree_from_pattern(pattern))
        return self

    def passes_constraints(self, result: MatchResult) -> bool:
        if not super().passes_constraints(result):
            return False

        # kbase = the_knowledgebase()

        # # TODO: Sanction this in the Matcher API somewhere...
        # eq_matcher = Matcher(tree.Number(value=1))
        # eq_matcher.constraints = copy.deepcopy(self.constraints)
        # for term_name, expr in result._expressions.items():
        #     eq_matcher.constrain_set(term_name, expr.output_set)
        # for op_name, pattern in result._binary_patterns.items():
        #     eq_matcher.constrain_binary_operator(op_name, pattern)

        # Enfore constrain_equal() here.
        # Reconstruct the Expression form of the source pattern.
        source_expr = expression_from_tree(self.pattern_tree, result)
        for equality in self.rule_constraints.equalities:
            # TODO: Check if the knowledgebase has this relationship.
            # eq_matcher.pattern_tree = equality

            # logging.warning(f"Matching: {equality} in KB...")
            # for match in kbase.query_matcher(eq_matcher):
            #     logging.warning(f"Found match: {match}")
            #     # Oooof, ok, what if we want to match:
            #     # self.pattern_tree = "a ? (2 * n_pattern)"
            #     # match.rhs = "a ** (n + m)"
            #     # Going to need some semblance of "partial matching"
            #     eq_result = super().match(match.rhs)
            #     if eq_result is None:
            #         continue
            #     break
            # else:
            #     return None

            try:
                eq_expr = expression_from_tree(equality, result)
            except BinaryRuleNotFoundError:
                return False
            if not is_equivalent(source_expr, eq_expr):
                return False

        return True
