from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable

import typing

__all__ = [
    "tokenize",
    "TokenizeError",
    "Token",
    "Label",
    "Number",
    "Operator",
    "OpenBracket",
    "CloseBracket",
    "Comma",
]


def tokenize(pattern: str) -> list[Token]:
    try:
        lexer = Lexer(pattern)
        return lexer.tokens()

    except LexerError as e:
        message = f"{e.message}:\n  {pattern}\n  {'^':>{e.position + 1}}"
        raise TokenizeError(message)


class TokenizeError(Exception):
    ...


class LexerError(Exception):
    def __init__(self, message: str, position) -> None:
        self.position = position
        self.message = message
        super().__init__(message)


#
# Token classes
#


@dataclass
class Token:
    """Tokens are the pre-cursors to parsing a full expression tree from a pattern string.

    They "smoosh" together groups of characters that belong to the same low-level
    concept and make the higher level parser easier to reason about and manage.
    """

    # Columner position of the token in the expression (0-indexed).
    position: int


@dataclass
class Label(Token):
    """Any contiguous run of characters of [a-zA-Z0-9_] that start with [a-zA-Z_]"""

    label: str


class OpenBracket(Token):
    """The character "(" """

    ...


class CloseBracket(Token):
    """The character ")" """


class Comma(Token):
    """The character "," """

    ...


@dataclass
class Number(Token):
    """Any continguous run of characters of [0-9]"""

    value: float


@dataclass
class Operator(Token):
    """Any contiguous run of characters of [?*+-/%^~!&|@] (unescaped).
    These are all the possible python operator characters.
    """

    string: str


#
# Lexer that does the hard lifting
#


class Lexer:
    """Pre-clump pattern characters to tokens to help ease the load on a higher level parser.

    PEG for the tokens:

        fragment ALPHA = [a-zA-Z];
        fragment NUMERIC = [0-9];
        fragment OP_CHAR = ('?' | '*' | '+' | '-' | '/' | '%' | '^' | '~' | '!' | '&' | '|');
        fragment UNDERSCORE = '_';
        fragment DOT = '.';

        Label = (ALPHA | UNDERSCORE) (ALPHA | NUMERIC | UNDERSCORE)*;
        Number = (NUMERIC)* (DOT (NUMERIC)*)?;
        Operator = (OP_CHAR)+;
        OpenBracket = '(';
        CloseBracket = ')';
        Comma = ','

    Some notes:
     - This uses ANTLR's "fragment" syntax for uncaptured helpers.
     - Whitespace is ignored.

    """

    pattern: str
    position: int

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.position = 0

    def at_end(self):
        return self.position >= len(self.pattern)

    def tokens(self) -> list[Token]:
        tokens = []
        while not self.at_end():
            if self.match(str.isspace):
                continue
            tokens.append(self.parse_next())
        return tokens

    def match(self, *char_or_predicates: str | Callable[[str], bool]) -> str | None:
        if self.at_end():
            return None

        current_char = self.pattern[self.position]

        for char_or_pred in char_or_predicates:
            if isinstance(char_or_pred, str):
                if char_or_pred != current_char:
                    continue

            elif isinstance(char_or_pred, Callable):
                if not char_or_pred(current_char):
                    continue
            else:
                typing.assert_never(char_or_pred)

            # Success! Advance the position.
            self.position += 1
            return current_char

        return None

    def parse_next(self) -> Token:
        assert not self.at_end()

        if label := self.label():
            return label
        if number := self.number():
            return number
        if operator := self.operator():
            return operator

        first_pos = self.position
        if self.match("("):
            return OpenBracket(first_pos)
        if self.match(")"):
            return CloseBracket(first_pos)
        if self.match(","):
            return Comma(first_pos)

        char = self.pattern[self.position]
        raise LexerError(f"Encountered unexpected character in expression '{char}'", self.position)

    def label(self) -> Label | None:
        """Label = (ALPHA | UNDERSCORE) (ALPHA | NUMERIC | UNDERSCORE)*;"""
        first_pos = self.position
        if not self.match(str.isalpha, "_"):
            return None
        while self.match(str.isalpha, str.isnumeric, "_"):
            ...

        return Label(label=self.pattern[first_pos : self.position], position=first_pos)

    def number(self) -> Number | None:
        """Number = (NUMERIC)* (DOT (NUMERIC)*)?;"""
        first_pos = self.position
        while self.match(str.isnumeric, "."):
            ...

        if self.position == first_pos:
            return None

        captured = self.pattern[first_pos : self.position]
        parts = captured.split(".", maxsplit=2)
        if len(parts) == 3:
            # Point pos to the second decimal that occurs.
            pos = self.position - len(parts[-1])
            raise LexerError("Found more than one decimal in number while parsing", pos)

        return Number(value=float(captured), position=first_pos)

    def operator(self) -> Operator | None:
        first_pos = self.position
        while self.match(_is_operator_char):
            ...

        if self.position == first_pos:
            return None

        return Operator(string=self.pattern[first_pos : self.position], position=first_pos)


def _is_operator_char(char: str):
    return char in "?*+-/%^~!&|"
