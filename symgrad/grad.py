# Copyright 2018, Christopher Ham.

from collections.abc import Iterable
from functools import cache
import inspect
import logging

from .core import format_key, Lookup
from .expression import Expression
from .symbol import symbol


logger = logging.Logger(__name__)


# Store the registered grad rules.
try:
    _grad_lookup
except NameError:
    _grad_lookup = Lookup(Expression)


def register_grad(func, gen_cls=None, lookup=None):
    lookup = lookup or _grad_lookup
    sig = inspect.signature(func)

    if sig.return_annotation != inspect.Signature.empty:
        logger.debug("Grad function returns scalar value.")

    params = sig.parameters
    names = list(params.keys())
    types = [params[name].annotation for name in names]

    # TODO: Clean this up...
    gen_cls = gen_cls or params[names[0]].annotation
    key = gen_cls.grad_key_from_types(types)
    # _grad_lookup[key] = func
    # print('Register:', format_key(key))
    lookup.add(key, func)

    return func


def register_grad_rule(gen_cls):
    def reg(func):
        return register_grad(func, gen_cls)

    return reg


def grad(gen: Expression, wrt: str, lookup: Lookup = None) -> Expression:
    """Return the derivative of a generator with respect to Symbol, `wrt`."""
    if wrt not in gen.variables:
        return symbol(0)

    lookup = lookup or _grad_lookup
    # Lookup best function for this Generator.
    grad_func = lookup.lookup(gen.grad_key())
    if not grad_func:
        raise ValueError("Could not find grad for key, {}".format(format_key(gen.grad_key())))

    sig = inspect.signature(grad_func)
    kwargs = {}
    if "wrt" in sig.parameters:
        kwargs["wrt"] = wrt
    if "result" in sig.parameters:
        kwargs["result"] = gen

    out_cls = grad_func.__annotations__.get("out", None)
    if out_cls:
        out = out_cls()
        grad_func(*gen.args(), out=out, **kwargs)
    else:
        out = grad_func(*gen.args(), **kwargs)

    # TODO: Run a simplification algorithm to reduce the number of ops.
    return out


def multigrad(gen: Expression, wrt_list: Iterable[str]) -> list[Expression]:
    """Return a list of generators, each differentiated wrt an element from wrt_list."""
    return [grad(gen, wrt) for wrt in wrt_list]
