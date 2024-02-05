"""
The :mod:`teacher.metrics` module provides methods to measure
the performance of the different agnostic explainers from the
:mod:`teacher.explanation` module. These metrics take into account
factual explanations composed of multiple rules.

Functions
---------

The following functions are provided to compute different metrics:

:meth:`coverage`
    Returns the coverage that a single rule has over a dataset. In fuzzy
    context, this means that an instance is covered by a rule if the minimum
    activation degree of any antecedent of the rule is higher than a given
    threshold. Then, the coverage will get all the instances that are covered
    at least by one rule of the ones composing the factual.

:meth:`precision`
    Returns the percentage of the covered instances that match a given
    class value.

:meth:`fidelity`
    This metric specific for agnostic classifiers measures the degree of similarity
    of the blackbox classifier and the whitebox classifier, by using the blackbox
    as ground truth and getting the score of the whitebox against that ground truth.

:meth:`rule_fidelity`
    This metric checks the fidelity of the covered part of a dataset given a set
    of rules.

--------------------
"""
# =============================================================================
# Imports
# =============================================================================

# Local application
from ._rule import coverage, precision, fidelity, rule_fidelity
from ._counterfactual import implausibility, instability, proximity_dissimilarity, sparsity_dissimilarity


# =============================================================================
# Public objects
# =============================================================================
__all__ = [
    "coverage",
    "precision",
    "fidelity",
    "rule_fidelity",
    "implausibility",
    "instability",
    "proximity_dissimilarity",
    "sparsity_dissimilarity"
]
