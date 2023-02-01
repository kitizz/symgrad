# Copyright, Christopher Ham, 2022

"""Helper function around manipulating and comparing "chains" of Expressions.

A chain, in this context, refers to an ordered list of Expressions that have
been extracted from a BinaryOperator and are associative under that operator.

The functions here make use of predicates to allow for:
1. Unit testing independent of the behavior of other library operators and sets.
2. Separation from the binary_operators module without creating circular dependencies.
"""

import logging
from typing import Any, Callable, Sequence, TypeAlias
from ..expression import Expression
from ..constant import Constant

__all__ = [
    "combine_constants",
    "equivalent_chains",
]

CanSwap: TypeAlias = Callable[[Expression, Expression], bool]


# def collect_terms(
#     chain: Sequence[Expression],
#     can_swap: CanSwap,
#     find_rule: Callable[[Expression, Expression], Any | None],
#     collect: Callable[[Any, Expression, Expression], Expression | None],
# ) -> list[Expression]:
#     """TODO: Doc"""

#     def collect_next(chain):
#         logging.debug("collect_next: %s", chain)
#         for i in range(len(chain)):
#             for j in range(i + 1, len(chain)):
#                 # TODO: Do we need to test the collect function in reverse order?
#                 rule = find_rule(chain[i].output_set, chain[j].output_set)
#                 if rule is None:
#                     continue
#                 result = collect(rule, chain[i], chain[j])
#                 if result is None:
#                     continue

#                 shifted = move_element(chain, can_swap, from_=j, to=i + 1)
#                 if shifted is None:
#                     continue

#                 shifted[i] = result
#                 del shifted[i + 1]
#                 return shifted

#         return None

#     new_chain = list(chain)
#     while True:
#         next_chain = collect_next(new_chain)
#         if next_chain is None:
#             return new_chain
#         new_chain = next_chain


def combine_constants(
    chain: Sequence[Expression],
    can_swap: CanSwap,
    reduce_constants: Callable[[Constant, Constant], Constant],
) -> list[Expression]:
    """Combine and reduce all Constants in a chain allowed by the can_swap predicate.

    Requires:
     - can_swap returns whether any consecutive pair in a chain is allowed to be swapped
     - reduce_constants combines two constants into one (usually with some BinaryOperator)

    Ensures:
     - All Constant terms are shifted as far to the left (zero index) as possible,
       when allowed by can_swap.
     - Any consecutive pairs of Constants encountered are reduced to one with reduce_constants.
    """
    chain = list(chain)
    for i in reversed(range(len(chain) - 1)):
        x, y = chain[i], chain[i + 1]
        if not isinstance(y, Constant):
            continue

        if isinstance(x, Constant):
            # Eval two constants into one.
            chain[i] = reduce_constants(x, y)
            del chain[i + 1]
        elif can_swap(x, y):
            # Swap to bring the Constant term more to the left.
            chain[i], chain[i + 1] = y, x
    return chain


def equivalent_chains(
    chain_a: Sequence[Expression],
    chain_b: Sequence[Expression],
    is_equal: Callable[[Expression, Expression], bool],
    can_swap: CanSwap,
) -> bool:
    """Checks if chain_a can be rearranged to match chain_b.

    Requires:
     - is_equal returns whether any two Expressions are equal (or equivalent)
     - can_swap returns whether any consecutive pair in a chain is allowed to be swapped

    Ensures:
     - All valid permutations of chain_a are checked for equality against chain_b,
       element-wise with is_equal.
     - A valid permutation is any obtained by any series of consecutive pair swaps with can_swap.
     - Function is symmetric: (equivalent_chains(a, b, ...) == equivalent_chain(b, a, ...)
    """
    chain_a = list(chain_a)
    chain_b = list(chain_b)

    if len(chain_a) != len(chain_b):
        logging.debug("Chains different length after combine_constants")
        return False

    num_el = len(chain_a)
    # Map the possible associations for each element in chain_a to chain_b
    a_to_bs: list[list[int]] = [[] for _ in range(num_el)]
    for i in range(num_el):
        for j in range(num_el):
            if is_equal(chain_a[i], chain_b[j]):
                a_to_bs[i].append(j)

    onto_indices = set(range(num_el))
    for i, mapping in enumerate(a_to_bs):
        if len(mapping) == 0:
            logging.debug(f"Unable to find an equiv mapping from chain a to b. Index: {i}")
            return False
        elif len(mapping) > 1:
            raise NotImplementedError("Yet to handle multiple equivalent terms")

        onto_indices.remove(mapping[0])

    if onto_indices:
        logging.debug(f"Unable to find a mapping onto chain b from a. Indices: {onto_indices}")
        return False

    # Finally need to check if we can match up the args, while satisfying commutivity.
    a_to_b: list[int] = [v[0] for v in a_to_bs]
    for target_ind in range(num_el):
        # This isn't the necessarilly the fastest approach. But is easy for now.
        a_ind = next(i for i, mapping in enumerate(a_to_b) if target_ind == mapping)

        while a_ind != a_to_b[a_ind]:
            if a_ind > a_to_b[a_ind]:
                # Shift a_ind to the left.
                i = a_ind - 1
                j = a_ind
                a_ind -= 1
            else:
                # Shift a_ind to the right.
                i = a_ind
                j = a_ind + 1
                a_ind += 1

            if can_swap(chain_a[i], chain_a[j]):
                a_to_b[i], a_to_b[j] = a_to_b[j], a_to_b[i]
                chain_a[i], chain_a[j] = chain_a[j], chain_a[i]
            else:
                logging.debug(
                    f"Lack of commutivity between chain elements: ({chain_a[i]}, {chain_a[j]})"
                )
                return False

    return True


def sort_chain(
    chain: list[Expression], can_swap: CanSwap, key: Callable[[Expression], Any]
) -> list[Expression]:
    """Sort the chain according to key, up to what is allowed by can_swap.

    Ensures:
     - chain is sorted by swaps of neighboring elements.
     - Only pairs that return true for can_swap are swapped.
     - Input chain is unmodified.
    """
    pairs = [(key(expr), expr) for expr in chain]

    # Hah, looks like BubbleSort has a place in the world afterall?
    # Interestingly, no literature could be found around problems like this.
    for start in range(len(pairs) - 1):
        # Starting at the end of the list, attempt swap down the minimum key element
        # as far down to index, start, as possible.
        for i in reversed(range(start, len(pairs) - 1)):
            k_i, expr_i = pairs[i]
            k_j, expr_j = pairs[i + 1]
            if can_swap(expr_i, expr_j) and k_j < k_i:
                # Swap i + 1 back down the chain
                pairs[i], pairs[i + 1] = pairs[i + 1], pairs[i]

    return [expr for k, expr in pairs]


# def move_element(
#     orig_chain: Sequence[Expression], can_swap: CanSwap, from_: int, to: int
# ) -> list[Expression] | None:
#     """
#     Ensures:
#      - If allowed by can_swap, the element as chain[from_] will be swapped with neighbors
#        until it is at index "to"; the new list is returned.
#      - If it cannot be done, None is returned.
#     """
#     chain = list(orig_chain)
#     while from_ != to:
#         if from_ > to:
#             # Shift a_ind to the left.
#             i = from_ - 1
#             j = from_
#             from_ -= 1
#         else:
#             # Shift a_ind to the right.
#             i = from_
#             j = from_ + 1
#             from_ += 1

#         if can_swap(chain[i], chain[j]):
#             chain[i], chain[j] = chain[j], chain[i]
#         else:
#             return None
#     return chain
