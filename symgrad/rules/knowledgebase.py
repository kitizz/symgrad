from __future__ import annotations

import dataclasses
import logging
import time
import typing
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar

from prettytable import PrettyTable

from ..constant import Constant
from ..exact import exact
from ..expression import Expression
from ..operators import BinaryOperator, Operator, UnaryOperator
from ..set import Set
from ..shared_context import define_rules, thread_local_context
from ..util.read_write_lock import ReadWriteLock
from ..variable import Variable
from .match_sets import match_sets
from .matcher import Matcher

if typing.TYPE_CHECKING:
    # Annoyingly, cache currently suppresses proper type checking of the underlying function.
    # This gets around it for now. Consider an internal cache wrapper if this comes up a lot.
    # This was suggested by the benevolent dictator, himself:
    # https://github.com/python/mypy/issues/5107#issuecomment-529372406
    F = TypeVar("F", bound=Callable)

    def cache(f: F) -> F:
        ...

else:
    from functools import cache

__all__ = ["Knowledgebase", "Match", "the_knowledgebase"]

logger = logging.getLogger(__name__)
logger.setLevel("WARNING")


def the_knowledgebase() -> Knowledgebase:
    """Access the Knowledgebase for the current context.

    Ensures:
     - If inside a shared_context.use_knowledgebase(kbase) context, then
       kbase will be returned (mainly for testing).
     - Otherwise, the default global Knowledgebase will be returned.
    """
    kbase = thread_local_context().knowledgebase
    if kbase is None:
        return _default_knowledgebase()
    elif not isinstance(kbase, Knowledgebase):
        raise TypeError(
            "Something other than a Knowledgebase object was given to "
            "the use_knowledgebase() context provider."
        )
    return kbase


Expression._add_spaghetti("the_knowledgebase", the_knowledgebase)


@cache
def _default_knowledgebase() -> Knowledgebase:
    return Knowledgebase()


@dataclass
class Match:
    lhs: Expression
    rhs: Expression
    mapping: dict[str, Expression]


class KnowledgebaseError(Exception):
    ...


class Knowledgebase:
    """

    Interesting resources:
    https://ris.utwente.nl/ws/files/6405767/274_acta.pdf
    """

    # Map hashes to expressions
    _patterns: dict[int, Expression]

    _matchers: dict[int, Matcher]
    _matchers_index_by_op: defaultdict[type[Operator], list[int]]

    _matchers_index_by_struchash: defaultdict[int, list[int]]

    # Map bidirectional equalities of the pattern hashes.
    _equalities: defaultdict[int, list[int]]

    _binary_set_rules: dict[BinaryRuleKey, Set]
    _binary_rule_index_by_op: defaultdict[type[BinaryOperator], list[BinaryRuleKey]]

    _unary_set_rules: dict[UnaryRuleKey, Set]
    _unary_rule_index_by_op: defaultdict[type[UnaryOperator], list[UnaryRuleKey]]

    _rw_lock: ReadWriteLock

    #: This version number is incremented whenever a change is made.
    version: int

    def __init__(self):
        self._patterns = {}
        self._matchers = {}
        self._matchers_index_by_op = defaultdict(list)
        self._matchers_index_by_struchash = defaultdict(list)
        self._equalities = defaultdict(list)
        self._binary_set_rules = {}
        self._binary_rule_index_by_op = defaultdict(list)
        self._unary_set_rules = {}
        self._unary_rule_index_by_op = defaultdict(list)
        self._rw_lock = ReadWriteLock()
        self.version = 0
        self._stats = QueryStats()

    @contextmanager
    def _make_changes(self):
        with self._rw_lock.write():
            try:
                yield
            finally:
                self.version += 1

    #
    # Operator Rules
    #

    def add_binary_set_rule(
        self, op: type[BinaryOperator], a_set: Set, b_set: Set, return_set: Set
    ):
        with self._make_changes():
            key = BinaryRuleKey(op, a_set, b_set)
            previous = self._binary_set_rules.setdefault(key, return_set)
            if previous != return_set:
                raise KnowledgebaseError("Cannot change already defined Set rules for operators")
            self._binary_rule_index_by_op[op].append(key)

            # Clear the cache since things have shifted around.
            self.query_binary_set_rule.cache_clear()

    def add_unary_set_rule(self, op: type[UnaryOperator], a_set: Set, return_set: Set):
        with self._make_changes():
            key = UnaryRuleKey(op, a_set)
            previous = self._unary_set_rules.setdefault(key, return_set)
            if previous != return_set:
                raise KnowledgebaseError("Cannot change already defined Set rules for operators")
            self._unary_rule_index_by_op[op].append(key)

            # Clear the cache since things have shifted around.
            self.query_unary_set_rule.cache_clear()

    @cache
    def query_binary_set_rule(self, op: type[BinaryOperator], a_set: Set, b_set: Set) -> Set | None:
        with self._rw_lock.read():
            best_key = _query_set_rule(self._binary_rule_index_by_op[op], a_set, b_set)
            if not best_key:
                return None

            var = best_key.match(a_set, b_set)
            assert var is not None
            output_set = self._binary_set_rules[best_key]
            return output_set.eval_parameters(var)

    @cache
    def query_unary_set_rule(self, op: type[UnaryOperator], a_set: Set) -> Set | None:
        with self._rw_lock.read():
            best_key = _query_set_rule(self._unary_rule_index_by_op[op], a_set)
            if not best_key:
                return None

            var = best_key.match(a_set)
            assert var is not None
            output_set = self._unary_set_rules[best_key]
            return output_set.eval_parameters(var)

    #
    # Relation Rules
    #

    def add(self, lhs: Expression, rhs: Expression):
        # TODO: Consider a way to avoid double querying symmetric rules.
        # These are rules for which going from LHS -> RHS is functionally equivalent
        # to going from RHS -> LHS.
        with self._make_changes():
            lhs_id = lhs._hash_()
            rhs_id = rhs._hash_()
            self._patterns[lhs_id] = lhs
            self._patterns[rhs_id] = rhs
            self._equalities[lhs_id].append(rhs_id)
            self._equalities[rhs_id].append(lhs_id)
            self._matchers[lhs_id] = Matcher(lhs)
            self._matchers[rhs_id] = Matcher(rhs)

            if isinstance(lhs, Operator):
                self._matchers_index_by_op[type(lhs)].append(lhs_id)
            if isinstance(rhs, Operator):
                self._matchers_index_by_op[type(rhs)].append(rhs_id)

    def query(
        self,
        expr: Expression,
        rhs: Expression | None = None,
        verbose=False,
    ) -> Generator[Match, None, None]:
        if verbose:
            logger = logging.getLogger(f"{__name__}.query")
            logger.setLevel("INFO")
        else:
            logger = logging.getLogger(__name__)

        expr_id = expr.hash()
        self._stats.queried.setdefault(expr_id, expr)
        self._stats.query_counts[expr_id] += 1
        t0 = time.time()

        # TODO: Explore why stuchash didn't work for matching pattern like "-1 * x"
        struchash = structure_hash(expr)

        with self._rw_lock.read():
            logger.info("Query with: %s", expr)

            # Record some stats.

            # Potentially speed up queries by using indexes.
            matchers: Iterable[tuple[int, Matcher]]
            if struchash in self._matchers_index_by_struchash:
                matchers = (
                    (pattern_id, self._matchers[pattern_id])
                    for pattern_id in self._matchers_index_by_struchash[struchash]
                )
            elif isinstance(expr, Operator):
                matchers = (
                    (pattern_id, self._matchers[pattern_id])
                    for pattern_id in self._matchers_index_by_op[type(expr)]
                )
            else:
                matchers = self._matchers.items()

            compare_count = 0
            match_count = 0
            matched_ids = []
            # Perform the query.
            for pattern_id, matcher in matchers:
                compare_count += 1
                result = matcher.match(expr)
                if result is None:
                    continue

                matched_ids.append(pattern_id)
                lhs = self._patterns[pattern_id]
                mapping = result.variable_map()
                match_count += 1
                for other_id in self._equalities[pattern_id]:
                    logger.info("Match: %s", lhs)
                    match_rhs = self._patterns[other_id]
                    if not rhs or match_rhs.sub(mapping) == exact(rhs):
                        yield Match(lhs=lhs, rhs=match_rhs, mapping=mapping)

            if struchash not in self._matchers_index_by_struchash:
                self._matchers_index_by_struchash[struchash] = matched_ids

            self._stats.query_comparisons[expr_id] += compare_count
            self._stats.query_matches[expr_id] += match_count
            self._stats.query_times[expr_id] += (time.time() - t0) * 1000

    def query_matcher(self, matcher: Matcher) -> Generator[Match, None, None]:
        with self._rw_lock.read():
            for pattern_id, expr in self._patterns.items():
                result = matcher.reverse_match(expr)
                if result is None:
                    continue

                lhs = self._patterns[pattern_id]
                mapping = result.variable_map()
                for other_id in self._equalities[pattern_id]:
                    yield Match(
                        lhs=lhs,
                        rhs=self._patterns[other_id],
                        mapping=mapping,
                    )

    @contextmanager
    def block_writes(self):
        """Block any writes to this Knowledgebase while inside the block_writes context."""
        with self._rw_lock.read():
            yield

    def __hash__(self) -> int:
        return hash((id(self), self.version))

    def __eq__(self, other):
        return self is other


@dataclass
class UnaryRuleKey:
    op: type[UnaryOperator]
    a: Set

    def sets(self):
        return (self.a,)

    def match(self, a_set: Set) -> dict[Variable, Any] | None:
        var = {}
        if not match_sets(self.a, a_set, var):
            return None
        return var

    def __hash__(self):
        return hash((self.op, self.a))

    def __lt__(self, other):
        """Ensures that the BinaryRule with clearly more specific rules comes second."""
        if not isinstance(other, UnaryRuleKey):
            return NotImplemented

        return (other.a in self.a) and (other.a is not self.a)


@dataclass
class BinaryRuleKey:
    op: type[BinaryOperator]
    a: Set
    b: Set

    def sets(self):
        return (self.a, self.b)

    def match(self, a_set: Set, b_set: Set) -> dict[Variable, Any] | None:
        var = {}
        if not (match_sets(self.a, a_set, var) and match_sets(self.b, b_set, var)):
            return None
        return var

    def __hash__(self):
        return hash((self.op, self.a, self.b))

    def __lt__(self, other):
        """Ensures that the BinaryRule with clearly more specific rules comes second."""
        if not isinstance(other, BinaryRuleKey):
            return NotImplemented

        a_subset = (other.a in self.a) and (other.a is not self.a)
        b_subset = (other.b in self.b) and (other.b is not self.b)
        return a_subset and b_subset


Key = TypeVar("Key", "UnaryRuleKey", "BinaryRuleKey")


def _query_set_rule(keys: list[Key], *sets: Set) -> Key | None:
    scored_keys = []
    for key in keys:
        if key.match(*sets) is None:
            continue

        # Matching exactly scores the best, subsets score a little less well.
        score = 0
        for rule_arg, in_arg in zip(key.sets(), sets):
            distance = in_arg.superset_distance(rule_arg)
            if distance < 0:
                continue
            score += 1.0 / (distance + 1)

        scored_keys.append((score, key))

    if len(scored_keys) == 0:
        return None

    # Note that BinaryRuleKey's __lt__ ensures that the more specific .args comes
    # after others when tie-breaking the score.
    best_score, best_key = sorted(scored_keys)[-1]
    return best_key


@dataclass
class QueryStats:
    queried: dict[int, Expression] = dataclasses.field(default_factory=dict)
    query_counts: Counter[int] = dataclasses.field(default_factory=Counter)
    query_comparisons: Counter[int] = dataclasses.field(default_factory=Counter)
    query_matches: Counter[int] = dataclasses.field(default_factory=Counter)
    query_times: defaultdict[int, float] = dataclasses.field(
        default_factory=lambda: defaultdict(float)
    )

    def print(self):
        table = PrettyTable()
        table.field_names = (
            "Queried",
            "Compares",
            "Matches",
            "Ratio",
            "Total time (ms)",
            "Ave time (ms)",
            "Expr",
        )
        for expr_id, count in self.query_counts.most_common(30):
            # print(f"{count:>5}: {self.queried[expr_id]}")
            matches = self.query_matches[expr_id]
            compares = self.query_comparisons[expr_id]
            times = self.query_times[expr_id]
            table.add_row(
                (
                    count,
                    compares,
                    matches,
                    f"{matches / (compares or 1):.2}",
                    f"{int(times)}",
                    f"{times / count:.2}",
                    str(self.queried[expr_id]),
                )
            )
        print("Knowledgebase query stats...")
        print(table)


def structure_hash(expr: Expression) -> int:
    match expr:
        case BinaryOperator():
            hash_a = structure_hash(expr.a)
            hash_b = structure_hash(expr.b)
            # TODO: Make two tuples of the expr.a.variables in expr.b and visa-versa
            a_in_b = tuple(expr.b.has_variable(a_var) for a_var in expr.a.variables)
            b_in_a = tuple(expr.a.has_variable(b_var) for b_var in expr.b.variables)
            return hash((type(expr).__name__, hash_a, hash_b, a_in_b, b_in_a))

        case UnaryOperator():
            return hash((type(expr).__name__, structure_hash(expr.a)))

        case Variable():
            return hash(("Variable", expr.output_set))

        case Constant():
            return expr.hash()

        case _:
            raise TypeError(f"Unsupported type {type(expr)}")
