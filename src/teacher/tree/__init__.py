"""

.. _ID3 : https://link.springer.com/content/pdf/10.1007/BF00116251.pdf
.. _here : https://doi.org/10.1109/TFUZZ.2016.2646746

The :mod:`teacher.tree` implements the Decision Trees used as a white-box
for the different explainers. Currently, there are two trees implemented as
well as a representation for the rules.

The classes are the following ones:

:class:`.BaseDecisionTree`
    Base class with the common methods that will be used for
    the rest of the decision trees.

:class:`.ID3`
    Fuzzy adaptation of the ID3_ algorithm that implements an inference that allows
    for an instance to traverse multiple branches of the tree. Used mainly as a baseline
    for comparison against other algorithms.

:class:`.FDT`
    Multiway Fuzzy Decision Tree implemented as described here_. This tree takes a set
    of fuzzy variables and a dataset and generates a decision tree, to get later
    a classification based on different types of inference.

-------------------------
"""
# =============================================================================
# Imports
# =============================================================================

# Local application
from .id3_tree import ID3
from .fdt_tree import FDT
from .fdt_binary_tree import FBDT
from .base_decision_tree import BaseDecisionTree
from .rule import Rule


# =============================================================================
# Public objects
# =============================================================================
__all__ = [
    "BaseDecisionTree",
    "ID3",
    "FDT",
    "FBDT",
    "Rule"
]
