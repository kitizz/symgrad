from __future__ import annotations

import functools
import logging
import typing
from collections.abc import Callable, Hashable
from typing import Any, Generic, ParamSpec, TypeVar

from symgrad.expression import Expression
from symgrad.operators.operator import Operator
from symgrad.rules.matcher import Matcher, MatchResult, PatternCaptures

Return = TypeVar("Return")

Args = ParamSpec("Args")


class MatcherCache(Generic[Return]):
    matcher: Matcher
    failed_result: Return
    calculate: Callable[[MatchResult], Return]

    results: dict[Hashable, Return]

    # Used to create deterministic keys for the Matcher.
    key_inputs: PatternCaptures

    _overloads: dict[str, Callable[..., Return]]
    part_names: dict[str, str]

    def __init__(
        self,
        matcher: Matcher,
        calculate: Callable[[MatchResult], Return],
        failed_result,
        part_names: dict[str, str],
    ):
        self.matcher = matcher
        self.calculate = calculate
        self.results = {}
        self.key_inputs = matcher.pattern_captures()
        self.failed_result = failed_result
        self.part_names = part_names

    def __call__(self, expr: Expression) -> Return:
        match = self.matcher.match(expr, apply_constraints=False)
        return self._handle_match(match)

    def from_parts(self, **kwargs: Expression | type[Operator]) -> Return:
        parts = {self.part_names[name]: value for name, value in kwargs.items()}
        return self._parts(parts)

    def _parts(self, parts: dict[str, Expression | type[Operator]]) -> Return:
        match = self.matcher.match_parts(parts, apply_constraints=False)
        return self._handle_match(match)

    def _handle_match(self, match: MatchResult | None) -> Return:
        # We apply constraints later so that we can make a key, and cache any constraint failures.
        if match is None:
            return self.failed_result

        key = self.key(match)
        if key is None:
            return self.failed_result

        if key in self.results:
            return self.results[key]

        if not self.matcher.passes_constraints(match):
            self.results[key] = self.failed_result
            return self.failed_result

        result = self.calculate(match)
        self.results[key] = result
        return result

    def key(self, match: MatchResult):
        key = []
        for name in self.key_inputs.variables:
            key.append(match.set(name))
        for name in self.key_inputs.unary_ops:
            key.append(match.unary_operator(name))
        for name in self.key_inputs.binary_ops:
            key.append(match.binary_operator(name))
        return tuple(key)

    def make_key(self, mappings: dict[str, Any]):
        key = []
        for name in self.key_inputs.variables:
            key.append(mappings[name])
        for name in self.key_inputs.unary_ops:
            key.append(mappings[name])
        for name in self.key_inputs.binary_ops:
            key.append(mappings[name])
        return tuple(key)

    # TODO: Delete overload logic?
    # def __getattr__(self, name: str) -> Any:
    #     return self._overloads[name]

    # def overload(self, func: Callable[..., Return]):
    #     self._overloads[func.__name__] = func


def matcher_cache(
    *,
    pattern: str | None = None,
    matcher: Matcher | None = None,
    failed_result,
    part_names: dict[str, str] | None = None,
):
    if pattern is None and matcher is None:
        raise TypeError("Must define a Matcher or a pattern to construct a Matcher")
    if not matcher:
        assert pattern
        matcher = Matcher(pattern)

    def wrapper(func: Callable[[MatchResult], Return]) -> MatcherCache[Return]:
        cache = MatcherCache(matcher, func, failed_result, part_names or {})
        return functools.update_wrapper(cache, func)

    return wrapper
