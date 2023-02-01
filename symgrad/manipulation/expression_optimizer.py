from __future__ import annotations
from dataclasses import dataclass
import dataclasses
import heapq

import logging
from collections import defaultdict, deque
from collections.abc import Callable
import sys
from typing import Any, Generator, Generic, Protocol, TypeAlias, TypeVar

from symgrad.constant import Constant
from symgrad.expression import Expression
from symgrad.operators.binary_operator import BinaryOperator
from symgrad.operators.unary_operator import UnaryOperator
from symgrad.rules.knowledgebase import Knowledgebase
from symgrad.variable import Variable
from symgrad.set_element import SetElement

__all__ = ["ExpressionOptimizer"]

logger = logging.getLogger(__name__)
logger.setLevel("WARNING")


class Comparable(Protocol):
    def __lt__(self, other, /) -> bool:
        ...

    def __eq__(self, other, /) -> bool:
        ...


CostType = TypeVar("CostType", bound=Comparable)
ClusterType = TypeVar("ClusterType")

ExprID: TypeAlias = Expression.Hash


@dataclass(kw_only=True)
class Node(Generic[CostType]):
    expr: Expression
    cost: CostType

    expr_id: ExprID = dataclasses.field(init=False)

    def __post_init__(self):
        self.expr_id = self.expr.hash()


class Cluster(Generic[CostType]):
    """TODO: Doc"""

    index: int
    expressions: set[ExprID]

    best_exprs: set[ExprID]
    best_cost: CostType

    def __init__(self, index: int, node: Node):
        self.index = index
        self.expressions = set()
        self.best_cost = node.cost
        self.best_exprs = {node.expr_id}
        self.add(node)

    def is_best(self, expr: Expression) -> bool:
        """Return if expr is currently one of the best options in the Cluster."""
        return expr.hash() in self.best_exprs

    def add(self, node: Node[CostType]):
        """Add this expression to the Cluster along with its metric to minimize."""
        self.expressions.add(node.expr_id)

        if node.cost < self.best_cost:
            self.best_cost = node.cost
            self.best_exprs = {node.expr_id}
        elif node.cost == self.best_cost:
            self.best_exprs.add(node.expr_id)

    def merge_from(self, other: Cluster[CostType]):
        """Merge another cluster into this one."""
        self.expressions.update(other.expressions)

        if self.best_cost == other.best_cost:
            self.best_exprs.update(other.best_exprs)
        elif other.best_cost < self.best_cost:
            self.best_cost = other.best_cost
            self.best_exprs = other.best_exprs

    def __contains__(self, expr: Expression) -> bool:
        return expr.hash() in self.expressions


class ExpressionOptimizer(Generic[CostType, ClusterType]):
    """Performs a search for an equivalent Expression that minimizes some cost function."""

    kbase: Knowledgebase
    kbase_version: int

    cost_func: Callable[[Expression], CostType]
    cluster_func: Callable[[Expression], ClusterType]

    nodes: dict[ExprID, Node]

    clusters: dict[int, Cluster[CostType]]
    next_cluster_index: int = 0
    node_to_cluster: dict[ExprID, int]

    # Stores an edge weight, which is the number of "moves" to go from one node to the other.
    equality_edges: defaultdict[ExprID, dict[ExprID, int]]

    explored: set[ExprID]

    def __init__(
        self,
        kbase: Knowledgebase,
        cost_func: Callable[[Expression], CostType],
        cluster_func: Callable[[Expression], ClusterType],
    ):
        self.clusters = {}
        self.node_to_cluster = {}
        self.nodes = {}
        self.equality_edges = defaultdict(dict)

        self.kbase = kbase
        self.kbase_version = kbase.version
        self.cost_func = cost_func
        self.cluster_func = cluster_func
        self.explored = set()

    def cluster(self, expr: Expression) -> Cluster[CostType]:
        """TODO: Doc"""
        cluster_id = self.node_to_cluster.get(expr.hash())
        if cluster_id is None:
            raise KeyError("Expression has no record in this optimizer")
        return self.clusters[cluster_id]

    def add(self, expr: Expression) -> Cluster[CostType]:
        """TODO: Doc"""
        cluster_id = self.node_to_cluster.get(expr.hash())
        if cluster_id is not None:
            return self.clusters[cluster_id]

        node = Node(expr=expr, cost=self.cost_func(expr))
        cluster = Cluster(self.next_cluster_index, node)
        self.next_cluster_index += 1

        self.nodes[node.expr_id] = node
        self.clusters[cluster.index] = cluster
        self.node_to_cluster[node.expr_id] = cluster.index

        return cluster

    def set_equal(
        self, lhs: Expression, rhs: Expression, num_moves: int
    ) -> Cluster[CostType] | None:
        """TODO: Doc

        Ensures:
         - lhs and rhs are put in the same Cluster if self.cluster_func() returns
           the same value for them both.
         - If lhs and rhs can exist in the same Cluster, then that Cluster is returned,
           otherwise None is returned.
        """
        lhs_cluster = self.add(lhs)
        rhs_cluster = self.add(rhs)
        if self.cluster_func(lhs) != self.cluster_func(rhs):
            # cluster_func must be equal in order to merge these clusters.
            logger.info("Different clusters: %s != %s", lhs, rhs)
            return

        lhs_id = lhs.hash()
        rhs_id = rhs.hash()
        if lhs_id == rhs_id:
            return lhs_cluster

        lhs_edges = self.equality_edges[lhs_id]
        if rhs_id in lhs_edges:
            logger.debug("already equal: %s == %s", lhs, rhs)
            logger.debug("%s vs %s", lhs_edges[rhs_id], num_moves)
            assert lhs_edges[rhs_id] == num_moves
            assert self.equality_edges[rhs_id][lhs_id] == num_moves
            return lhs_cluster

        logger.debug("set_equal(%s, %s, %s)", lhs, rhs, num_moves)
        rhs_edges = self.equality_edges[rhs_id]
        lhs_edges[rhs_id] = num_moves
        rhs_edges[lhs_id] = num_moves

        if lhs_cluster is rhs_cluster:
            return lhs_cluster

        lhs_cluster.merge_from(rhs_cluster)
        self.clusters.pop(rhs_cluster.index)
        for expr_id in rhs_cluster.expressions:
            self.node_to_cluster[expr_id] = lhs_cluster.index

        return lhs_cluster

    def _connected_nodes(self, node_id: ExprID) -> Generator[tuple[ExprID, int], None, None]:
        assert node_id in self.equality_edges
        for next_id, num_moves in self.equality_edges[node_id].items():
            yield next_id, num_moves

    def _distance(self, lhs: Expression, rhs: Expression):
        """Returns the minimum number of "equality hops" needed to go from lhs to rhs"""
        lhs_id = lhs.hash()
        rhs_id = rhs.hash()
        assert lhs_id in self.nodes and rhs_id in self.nodes

        if lhs_id == rhs_id:
            return 0

        # Just your good ol' fashioned Dijkstra's algorithm.
        # Heapqueue uses (distance, insertion index, Expr ID) as tuple.
        queue: list[tuple[int, int, ExprID]] = []
        heapq.heappush(queue, (0, 0, lhs_id))
        queue_index = 1
        final_distances: dict[ExprID, int] = {}
        seen_distances: defaultdict[ExprID, int] = defaultdict(lambda: 100000000)

        while queue:
            (curr_dist, _, curr_id) = heapq.heappop(queue)
            if curr_id == rhs_id:
                return curr_dist
            if curr_id in final_distances:
                continue
            # Once we get to this point, we're sure this is the shortest distance.
            final_distances[curr_id] = curr_dist

            for next_id, num_moves in self._connected_nodes(curr_id):
                next_final_dist = final_distances.get(next_id)
                if next_final_dist is not None:
                    assert next_final_dist <= curr_dist, "Should not happen with positive weights"
                    continue

                next_dist = curr_dist + num_moves
                if next_dist >= seen_distances[next_id]:
                    continue
                seen_distances[next_id] = next_dist
                heapq.heappush(queue, (next_dist, queue_index, next_id))
                queue_index += 1

        assert False, "LHS and RHS should never be unconnected in this graph..."

    def minimize(self, expr: Expression) -> Expression:
        """

        Ensures:
         - If there are multiple equally best options, then a subset of the equally
           nearest* options will be returned. Otherwise all the best options are returned.
           * See self.distance(prefer_expr, other)
        """
        if self.kbase.version != self.kbase_version:
            raise NotImplementedError(
                "Need to reset this object appropriately when changes are made to the Knowledgebase"
            )

        # If there are multiple bests, weed out the nearest ones in the graph.
        nearest_dist: int | None = None
        nearest: list[Expression] = []
        for best_expr, distance in self._best_alternatives(expr.sub()):
            if nearest_dist is None or distance < nearest_dist:
                nearest_dist = distance
                nearest = [best_expr]
            elif distance == nearest_dist:
                nearest.append(best_expr)

        # logger.info(f"Minimizing: {expr}")
        # logger.info(f"Best chosen: {nearest} from\n" + self._pretty_cluster_str(expr))
        if len(nearest) > 1:
            logger.warning("Multiple best options available! Need to widdle it down some more")
            # raise RuntimeError("Multiple best options available! Need to widdle it down some more")

        return nearest[0].sub()

    def _best_alternatives(self, expr: Expression) -> list[tuple[Expression, int]]:
        if isinstance(expr, Variable) or isinstance(expr, Constant):
            self.add(expr)
            return [(expr, 0)]

        with self.kbase.block_writes():
            self._explore_equalities(expr)

        cluster = self.cluster(expr)
        exprs_and_distance = []
        for best_id in cluster.best_exprs:
            best_expr = self.nodes[best_id].expr
            exprs_and_distance.append((best_expr, self._distance(expr, best_expr)))
        return exprs_and_distance

    def _explore_equalities(self, expr: Expression):
        """Grow the internal graph, seeded from expr, that potentially minimize the cost function.

        Each exploration path will terminate when it finds any of:
         - An expression that has a different ClusterFunction value
         - An expression that's already been explored
         - An expression that returns a worse CostFunction than the current cluster's best.
        """

        # Try to simplify the deeper branchs first.
        match expr:
            case BinaryOperator():
                op = type(expr)
                a_best = self._best_alternatives(expr.a)
                b_best = self._best_alternatives(expr.b)
                for a, a_num_moves in a_best:
                    for b, b_num_moves in b_best:
                        alt_expr = op(a, b)
                        self.set_equal(expr, alt_expr, a_num_moves + b_num_moves)
            case UnaryOperator():
                op = type(expr)
                a_best = self._best_alternatives(expr.a)
                for a, a_num_moves in a_best:
                    alt_expr = op(a)
                    self.set_equal(expr, alt_expr, a_num_moves)
            case Variable():
                self.add(expr)
                return
            case Constant():
                self.add(expr)
                return
            case _:
                assert False

        # Breadth-first search that only continues to explores from nodes that are
        # so far among the best options.

        q: deque[Expression] = deque(
            self.nodes[best_id].expr for best_id in self.cluster(expr).best_exprs
        )

        self.explored.add(expr.hash())
        # TODO: Need to consider a bigger-picture search that considers alterations to sub-expressions.
        logger.debug("Find alternatives for %s", expr)
        while q:
            leaf_expr = q.popleft()

            logger.debug("  Leaf: %s", leaf_expr)
            for match in self.kbase.query(leaf_expr):
                alt_expr = match.rhs.sub(match.mapping)
                logger.debug("    Match: %s", alt_expr)
                alt_cluster = self.set_equal(leaf_expr, alt_expr, num_moves=1)

                if not alt_cluster or not alt_cluster.is_best(alt_expr):
                    continue
                if alt_expr.hash() in self.explored:
                    continue

                q.append(alt_expr)

                # Hijacking explored to avoid adding the same expr twice in this loop.
                self.explored.add(alt_expr.hash())

    def _pretty_cluster_str(self, expr: Expression):
        """Helpful debugging string of the Expressions in this cluster.

        If expr_distance is provided, then the distance of each expression to it
        are also printed.
        """
        cluster = self.cluster(expr)

        def format_line(node: Node, padding: int) -> str:
            dist = self._distance(expr, node.expr)
            best = "*" if cluster.is_best(node.expr) else ""
            return f"  {str(node.expr):>{padding}} : {node.cost} [{dist}] {best}"

        nodes = [self.nodes[expr_id] for expr_id in cluster.expressions]
        cost_and_id = [(node.cost, node.expr_id, node) for node in nodes]
        cost_and_id.sort()

        padding = max(len(str(node.expr)) for _, _, node in cost_and_id)

        return "\n".join(format_line(node, padding) for _, _, node in cost_and_id)


# @cache
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
            name_type = "Constant" if expr.is_constant else "Variable"
            return hash((name_type, expr.output_set))

        case SetElement():
            return expr.hash()

        case Constant():
            return hash(("Constant", expr.output_set))

        case _:
            raise TypeError(f"Unsupported type {type(expr)}")
