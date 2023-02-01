# Copyright, Christopher Ham, 2022

from ..set import Set, register_set_mapping
from ..expression import Expression

__all__ = ["SetType"]


class SetType(Set):
    """Used to represent set (symbolic type) classes symbolically."""

    #: Constrain the possible sets to (inclusive) subsets of bound.
    upper_bound: Set | None = None

    def __contains__(self, value) -> bool:
        if isinstance(value, SetType):
            if self.upper_bound is None:
                return True
            if value.upper_bound is None:
                return False
            return value.upper_bound in self.upper_bound

        if isinstance(value, Expression):
            return value.output_set in self

        if not isinstance(value, Set):
            return False

        if self.upper_bound is not None:
            return value in self.upper_bound

        # When bound is None, all Sets are a subset of SetType.
        return True


@register_set_mapping
def set_to_settype(s: Set) -> SetType:
    return SetType(upper_bound=s)
