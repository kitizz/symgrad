import logging
from typing import Any

from symgrad.exact import exact
from symgrad.expression import Expression
from symgrad.set import Set
from symgrad.variable import Variable

__all__ = ["match_sets"]

logger = logging.getLogger(__name__)
logger.setLevel("WARNING")


def match_sets(pattern_set: Set, target_set: Set, set_params: dict[Variable, Any]) -> bool:
    """Check if target_set is a non-strict subset of pattern_set, and that
    their parameters are consistent.

    Parameter consistency can checked across multiple Set pairs by passing
    the same set_params dict into sequential calles to match_sets. That is
    because set_params is updated in-place to map an Variables found in
    pattern_set to their respective parameters in target_set.

    Ensures:
     - Returns true when all following condition met:
       - pattern_set's class is a superset class of target_set's
       - each pattern_set's parameter matches the target_set's; OR
         if a pattern_set parameter is a Variable, then the associated
         target_set parameter is consistent with the Variable's mapped value
         in the set_params dict.
     - set_params is updated to map any Variables in pattern_set's parameters
       to their associated target_set paramater values.

    Requires:
     - pattern_set parameters may only be constants or single Variables.
       More complex Expressions are not yet supported.
     - target_set may not define a Variable when pattern_set defines a constant.

    Examples:
    ```
    match_sets(Ints(), Reals(), {}) == True
    ```
    ```
    set_params = {}
    match_sets(Matrices(N, N), Matrices(3, 3), set_params) == True
    set_params == {N: 3}
    ```
    ```
    set_params = {}
    match_sets(Matrices(N, N), Matrices(3, 4), set_params) == False
    ```
    ```
    set_params = {}
    match_sets(Matrices(N, 1), Vectors(3), set_params) == True
    set_params == {N: 3}
    ```
    """
    target_superset = target_set.superset_of(type(pattern_set))
    if target_superset is None:
        logger.debug("target_superset is None")
        return False

    pattern_params = pattern_set._set_params
    target_params = target_superset._set_params
    for pattern_param, target_param in zip(pattern_params, target_params):
        if isinstance(target_param, Variable) and not isinstance(pattern_param, Variable):
            raise TypeError(
                f"Variable found in target_set ({target_set}) "
                f"where pattern_set defines constant ({pattern_set})"
            )

        if isinstance(pattern_param, Variable):
            # Try to match the rule variable its concrete counterpart.
            if target_param not in pattern_param.output_set:
                # The concrete params must be elements of the variable rule params.
                logger.debug(
                    f"Concrete param not a subset of rule param: "
                    f"{target_param} not in {pattern_param.output_set}"
                )
                return False

            if pattern_param not in set_params:
                # Variable not yet pinned down. Let's do so.
                set_params[pattern_param] = target_param
                logger.debug(f"Assigned {pattern_param} = {target_param}")
                continue
            elif exact(set_params[pattern_param]) != target_param:
                # Mismatch in args. Try the next rule.
                logger.debug(
                    f"Mismatch! {pattern_param}: {set_params[pattern_param]} != {target_param}"
                )
                return False

        elif pattern_param == target_param:
            # Constant/literal values match! Keep checking the others.
            logger.debug(f"Direct match {pattern_param} == {target_param}")
            continue

        elif isinstance(pattern_param, Expression):
            raise NotImplementedError("Cannot yet handle expressions for Set params.")

        else:
            logger.debug(f"No equality: {pattern_param} != {target_param}")
            return False

    return True
