# Copyright, Christopher Ham, 2022

from collections import OrderedDict
from typing import Any, Sequence


__all__ = [
    "collect_args",
]


def collect_args(
    arg_names: Sequence[str], args: tuple, kwargs: dict[str, Any]
) -> OrderedDict[str, Any]:
    """Combine the args and kwargs to match the annotations dict.

    Requires:
     - arg_names is ordered list of argument names.
     - args are provided positional arguments.
     - kwargs are provided keyword argments.
     - len(args) + len(kwargs) == len(annotations)
     - kwargs does not overlap with positional args.

    Ensures:
     - result.keys() == arg_names.
     - result.values() are filled first by args, then by kwargs.

    """
    if len(args) > len(arg_names):
        raise TypeError(
            f"Takes {len(arg_names)} arguments but {len(args)} positional args were given."
        )

    extra_kw = set(kwargs) - set(arg_names)
    if extra_kw:
        raise TypeError(f"Got unexpected keyword arguments, {tuple(extra_kw)}")

    result = OrderedDict()
    for i, key in enumerate(arg_names):
        if i < len(args):
            if key in kwargs:
                raise TypeError(f"Multiple values for argument, '{key}'")
            result[key] = args[i]
        elif key in kwargs:
            result[key] = kwargs[key]

    # Check that we didn't miss any required args.
    missing = set(arg_names) - set(result)
    if missing:
        raise TypeError(f"Missing {len(missing)} required args: {tuple(missing)}")

    return result
