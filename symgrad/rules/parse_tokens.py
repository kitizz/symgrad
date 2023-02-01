from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypeVar, overload

from . import abstract_tree as tree
from . import lexer

__all__ = [
    "parse_tokens",
    "ParseError",
]

# FAQ: Why is this written by hand?!
#
# A few off-the-shelf PEG (Parsing Expression Grammer) libraries* were considered
# when putting this together. At the time, they either didn't provide the desired
# features or weren't very actively maintained. Since the parsing required here
# wasn't terribly complex, and expected to be stable, I chose to roll my own.
#
# Considered libraries*:
#  - Arpeggio
#  - Parsimonious
#  - Pegen
#
# Pegen was the most promising one, claiming to be a subset of the official
# Python parser. But documentation and examples were minimal.
#
# In rolling my own, I can output directly to my own TreeNode datastructure
# without extra visiting logic. Plus, I can take full control of error catching.
#
# Check out these great resources for manually converting a grammar into a
# Recursive Descent Parser:
# [1] https://craftinginterpreters.com/parsing-expressions.html
# [2] https://medium.com/@gvanrossum_83706/peg-parsing-series-de5d41b2ed60


class ParseError(Exception):
    ...


def parse_tokens(tokens: Sequence[lexer.Token], source_pattern: str) -> tree.TreeNode:
    """Parse a sequence of tokens into an expression tree."""
    parser = Parser(tokens)
    try:
        return parser.expression()
    except TokenParserError as e:
        arrows = [" "] * max(e.range_in_source[1], e.position + 1)
        for i in range(*e.range_in_source):
            arrows[i] = "-"
        if e.position >= 0:
            arrows[e.position] = "^"
        message = f"{e.message}:\n  {source_pattern}\n  {''.join(arrows)}"
        raise ParseError(message)


class TokenParserError(Exception):
    range_in_source: tuple[int, int]
    position: int

    def __init__(
        self,
        message: str,
        *token_or_node: lexer.Token | tree.TreeNode,
        position=-1,
    ):
        if token_or_node:
            self.range_in_source = _calculate_range(*token_or_node)
        else:
            self.range_in_source = (position, position)
        self.position = position
        self.message = message
        super().__init__(message)


T1 = TypeVar("T1")
T2 = TypeVar("T2")


class Parser:
    """The Parsing Expression Grammar (PEG) for expressions:

    expression <- binary_op EOF
    binary_op <- unary_op (Operator unary_op)?
    unary_op <- Operator value / value
    value <- function / Label / Number / OpenBracket binary_op OpenBracket
    function <- Label OpenBracket binary_op (Comma binary_op)* CloseBracket

    Note: CamelCase identifiers refer to their respective lexer ParseToken
    """

    tokens: list[lexer.Token]
    position: int = 0

    def __init__(self, tokens: Sequence[lexer.Token]):
        self.tokens = list(tokens)

    def source_position(self) -> int:
        token = self._at(self.position)
        if token is None:
            return -1
        return token.position

    def _at(self, position) -> lexer.Token | None:
        if position < 0 or position >= len(self.tokens):
            return None
        return self.tokens[position]

    def at_end(self) -> bool:
        return self.position >= len(self.tokens)

    def match(self, lexer_type: type[T1]) -> T1 | None:
        result = self.match_many((lexer_type,))
        if not result:
            return None
        return result[0]

    # Some overload f**ery to make the type analyser happy.
    @overload
    def match_many(self, lexer_types: tuple[type[T1]]) -> tuple[T1] | None:
        ...

    @overload
    def match_many(self, lexer_types: tuple[type[T1], type[T2]]) -> tuple[T1, T2] | None:
        ...

    def match_many(self, lexer_types: tuple[type, ...]) -> tuple | None:
        matched = []
        for offset, lexer_type in enumerate(lexer_types):
            current = self._at(self.position + offset)
            if not isinstance(current, lexer_type):
                return None
            matched.append(current)

        self.position += len(lexer_types)
        return tuple(matched)

    def expression(self) -> tree.TreeNode:
        """expression <- binary_op"""
        if binary := self.binary_op():
            if not self.at_end():
                raise TokenParserError(
                    "May be missing operator or brackets", position=self.source_position() - 1
                )
            return binary
        else:
            raise RuntimeError("Unable to parse...")

    def binary_op(self) -> tree.TreeNode:
        """binary_op <- unary_op (Operator unary_op)?"""
        a = self.unary_op()
        if not (op := self.match(lexer.Operator)):
            return a
        b = self.unary_op()
        if b is None:
            raise RuntimeError("Unable to parse second part of binary op?")

        return tree.Function(
            name=op.string, children=[a, b], range_in_pattern=_calculate_range(a, b)
        )

    def unary_op(self) -> tree.TreeNode:
        """unary_op <- Operator value / value"""
        if op := self.match(lexer.Operator):
            value = self.value()
            return tree.Function(
                name=op.string, children=[value], range_in_pattern=_calculate_range(op, value)
            )
        return self.value()

    def value(self) -> tree.TreeNode:
        """value <- function / Label / Number / OpenBracket binary_op OpenBracket"""
        if func := self.function():
            return func

        if label := self.match(lexer.Label):
            return tree.Term(name=label.label, range_in_pattern=_calculate_range(label))

        if number := self.match(lexer.Number):
            return tree.Number(
                value=number.value,
                range_in_pattern=_calculate_range(number),
            )

        if open_bracket := self.match(lexer.OpenBracket):
            expr = self.binary_op()
            if not self.match(lexer.CloseBracket):
                position = self.source_position()
                raise TokenParserError("Missing ')'", open_bracket, expr, position=position)
            return expr

        if self.at_end():
            raise TokenParserError(
                "Unexpected end of expression", position=self.tokens[-1].position
            )

        raise TokenParserError("Unexpected character or token", position=self.source_position())

    def function(self) -> tree.Function | None:
        """function <- Label OpenBracket binary_op (Comma binary_op)* CloseBracket"""
        label_bracket = self.match_many((lexer.Label, lexer.OpenBracket))
        if not label_bracket:
            return None
        label = label_bracket[0]

        args = [self.binary_op()]
        while self.match(lexer.Comma):
            args.append(self.binary_op())

        close_bracket = self.match(lexer.CloseBracket)
        if not close_bracket:
            position = self.source_position() - 1
            if not self.at_end() and (expr := self.binary_op()):
                raise TokenParserError(
                    "May be missing ',' or operator", label, expr, position=position
                )
            else:
                raise TokenParserError(
                    "May be missing ')' for function", label, *args, position=position
                )

        return tree.Function(
            name=label.label,
            children=args,
            range_in_pattern=_calculate_range(label, close_bracket),
        )


def _calculate_range(*tokens_or_nodes: lexer.Token | tree.TreeNode):
    """Calculates the total span of a sequence of tokens or nodes in the original pattern."""
    min_pos = float("inf")
    max_pos = 0
    for token_or_node in tokens_or_nodes:
        if isinstance(token_or_node, lexer.Token):
            min_pos = min(min_pos, token_or_node.position)
            max_pos = max(max_pos, token_or_node.position + 1)
        else:
            min_pos = min(min_pos, token_or_node.range_in_pattern[0])
            max_pos = max(max_pos, token_or_node.range_in_pattern[1])
    assert isinstance(min_pos, int)
    return (min_pos, max_pos)
