# Copyright, Christopher Ham, 2022

from collections.abc import Sequence

__all__ = ["is_unordered_equal"]


def is_unordered_equal(seq_a: Sequence, seq_b: Sequence) -> bool:
    """Compare two unordered sets of objects using only equality comparisons.

    This avoids hashes, which may be more strict than equailty comparions."""
    list_b = list(seq_b)

    if len(seq_a) != len(list_b):
        return False

    for a in seq_a:
        for i_b, b in enumerate(list_b):
            if a == b:
                del list_b[i_b]
                break
        else:
            # No values in list_b match a
            return False

    return True
