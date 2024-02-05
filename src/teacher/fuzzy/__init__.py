"""

.. _here: https://doi.org/10.1109/TFUZZ.2016.2646746

The :mod:`teacher.fuzzy` module provides methods and classes
to represent fuzzy elements. This includes fuzzification procedures
to transform regular datasets into fuzzy datasets, as well as
classes to represent the Fuzzy Sets and Fuzzy Variables.

Fuzzification Functions
-----------------------

These functions provide an interface to fuzzify a dataset that is
preloaded into a :class:`pandas.DataFrame`. These functions are:

:meth:`get_fuzzy_points`
    This function provides three methods of discretization to divide
    a :class:`pandas.DataFrame` into triangular fuzzy sets. These three
    methods are:
        * Equal width: Divides each variable into triangular sets of equal width.

        * Equal frequency: Divides each variable into triangular sets of equal frequency.

        * Entropy: Divides each variable using the fuzzy partitioning based on fuzzy entropy \
        used here_

:meth:`get_fuzzy_variables`
    This function takes a list of points that define triangular fuzzy sets and returns
    the adequate :class:`.FuzzyVariable` objects. These points can be
    extracted via the :meth:`get_fuzzy_points` method or can be introduced manually
    given they are obtained through other means (i.e.: an expert).

:meth:`dataset_membership`
    This function takes a dataset and a set of fuzzy variables and returns a dictionary
    with the pertenence of each instance of the dataset to the different fuzzy sets of
    each variable.

Classes
-------

:class:`.FuzzyVariable`
    This class represents a fuzzy variable that has a name, a membership function and
    some Fuzzy Sets that can be either discrete or continuous.
:class:`.FuzzyDiscreteSet`
    This class represents a discrete fuzzy set, whose membership function is either 1 if
    the variable takes this value or 0 if it does not.
:class:`.FuzzyContinuousSet`
    This class represents a triangular continuous fuzzy set, whose membership function varies depending
    on how close the variable is from the peak of the set.

-------------------
"""
# =============================================================================
# Imports
# =============================================================================

# Local application
from ._base import (get_fuzzy_points,
                    dataset_membership,
                    get_fuzzy_variables)
from .fuzzy_set import FuzzyDiscreteSet, FuzzyContinuousSet
from .fuzzy_variable import FuzzyVariable

# =============================================================================
# Public objects
# =============================================================================

__all__ = [
    "get_fuzzy_points",
    "get_fuzzy_variables",
    "dataset_membership",
    "FuzzyVariable",
    "FuzzyDiscreteSet",
    "FuzzyContinuousSet",
]
