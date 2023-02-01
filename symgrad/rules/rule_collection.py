from __future__ import annotations

import abc

from symgrad.shared_context import define_rules


class RuleCollection(abc.ABC):
    """TODO: Doc"""

    _collections: list[type[RuleCollection]] = []

    def __init_subclass__(cls):
        cls._collections.append(cls)

        with define_rules():
            # Execute the definitions as they come in.
            cls.axioms()
            cls.theorems()

        super().__init_subclass__()

    @abc.abstractclassmethod
    def axioms(cls):
        """TODO: Doc"""
        ...
        
    @abc.abstractclassmethod
    def theorems(cls):
        """TODO: Doc"""
        ...
