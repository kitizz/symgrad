from __future__ import annotations
from dataclasses import dataclass, field
from functools import cache
import logging
from typing import Any, Generator
from ..set_element import SetElement
from ..expression import Expression


@dataclass
class InternalData:
    hash: int | None = None

    # The Node's constant evaluated result when it is available.
    constant: Any | None = None


@dataclass(frozen=True, init=False, kw_only=True)
class TreeNode:
    #: Point to the window of text in the original pattern that this node comes from.
    range_in_pattern: tuple[int, int] = (-1, -1)

    #: All nodes have a children tuple, though note that LeafNode enforces it remains empty.
    children: tuple[TreeNode, ...] = field(default_factory=tuple)

    _internal: InternalData = field(default_factory=InternalData)

    def __hash__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return NotImplemented
        return hash(self) == hash(other)


@dataclass(frozen=True, init=False, eq=False)
class LeafNode(TreeNode):
    #: LeafNodes may not point to any children.
    children: list[TreeNode] = field(default_factory=list, init=False)


@dataclass(frozen=True, eq=False)
class Function(TreeNode):
    """General node for functions and operators.

    The name field will reflect the function identifier or operator string.
    Examples:
        "a + b" -> Function(name="+", ...)
        sin(a) -> Function(name="sin", ...)
    """

    name: str

    def __str__(self):
        return f"{self.name}({', '.join(str(c) for c in self.children)})"

    def __hash__(self):
        if not self._internal.hash:
            values: list[Any] = ["Function", self.name]
            for c in self.children:
                values.append(hash(c))
            self._internal.hash = hash(tuple(values))
        return self._internal.hash


@dataclass(frozen=True, eq=False)
class Term(LeafNode):
    """AKA a Variable. "Term" was used to avoid colliding with symgrad's Variable type."""

    name: str

    def __str__(self):
        return self.name

    def __hash__(self):
        if not self._internal.hash:
            self._internal.hash = hash(("Term", self.name))
        return self._internal.hash


@dataclass(frozen=True, eq=False)
class Number(LeafNode):
    """Constant numbers parsed in expressions."""

    value: float | SetElement = float("nan")

    def __post_init__(self):
        self._internal.constant = self.value

    def __str__(self):
        return str(self.value)

    def __hash__(self):
        if not self._internal.hash:
            if isinstance(self.value, SetElement):
                self._internal.hash = hash(("Term", self.value._hash_()))
            else:
                self._internal.hash = hash(("Term", self.value))
        return self._internal.hash


def depth_first(tree_node: TreeNode) -> Generator[TreeNode, None, None]:
    for child in tree_node.children:
        for n in depth_first(child):
            yield n
    yield tree_node


def unique_terms(tree_node: TreeNode) -> Generator[Term, None, None]:
    seen: set[str] = set()
    for node in depth_first(tree_node):
        if not isinstance(node, Term):
            continue
        if node.name in seen:
            continue

        seen.add(node.name)
        yield node
