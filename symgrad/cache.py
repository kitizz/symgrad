import functools
from typing import TypeVar
from collections.abc import Callable, Hashable
import typing

from symgrad.rules.knowledgebase import Knowledgebase, the_knowledgebase
from .expression import Expression
from .exact import exact, Exact
from .shared_context import thread_local_context

T = TypeVar("T", bound=Callable)
Return = TypeVar("Return")


def cache(func: T) -> T:
    @functools.cache
    def unwrapper(args, **kwargs):
        args = tuple(_unwrap(arg) for arg in args)
        kwargs = {key: _unwrap(value) for key, value in kwargs.items()}
        return func(*args, **kwargs)

    def wrapper(*args, **kwargs):
        if thread_local_context().use_cache:
            args = tuple(_wrap(arg) for arg in args)
            kwargs = {key: _wrap(value) for key, value in kwargs.items()}
            return unwrapper(args, **kwargs)
        else:
            return func(*args, **kwargs)

    wrapper = functools.update_wrapper(wrapper, func)
    return typing.cast(type(func), wrapper)


def cache_single_expr(func: Callable[[Expression], Return]) -> Callable[[Expression], Return]:
    @functools.cache
    def unwrapper(expr_exact: Exact):
        return func(expr_exact.expr)

    def wrapper(expr: Expression):
        if thread_local_context().use_cache:
            return unwrapper(Exact(expr))
        else:
            return func(expr)

    wrapper = functools.update_wrapper(wrapper, func)
    return typing.cast(type(func), wrapper)


def _wrap(value) -> Hashable:
    if isinstance(value, Expression):
        return exact(value)
    return value


def _unwrap(value):
    if isinstance(value, Exact):
        return value.expr
    return value


def rule_cache(func: T) -> T:
    """Cache is attached to the active KnowledgeBase, and is invalidated when updates are made."""

    @functools.cache
    def unwrapper(_kbase: Knowledgebase, _kbase_version: int, *args, **kwargs):
        return func(*args, **kwargs)

    def wrapper(*args, **kwargs):
        kbase = the_knowledgebase()
        return unwrapper(kbase, kbase.version, *args, **kwargs)

    return typing.cast(type(func), wrapper)
