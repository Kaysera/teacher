"""

.. _LORE : https://doi.org/10.1109/MIS.2019.2957223

The :mod:`teacher.neighbors` module provides classes to generate neighborhoods
from an instance. These neighborhoods are used in agnostic explainers, so that
a blackbox can be explained in the locality of an instance. They are decoupled
from the explainers so that different neighborhoods can be used or created ad-hoc
for the different explainers.

Classes
-------

These are the different neighborhoods implemented in the tool. They match the Estimator
guidelines of scikit-learn and can be inherited and expanded to create a more
adequate neighborhood for the different explainers.

:class:`.BaseNeighborhood`
    Abstract base class from which all the neighbors must extend. Provided as baseline
    for new neighborhood creations.
:class:`.FuzzyNeighborhood`
    Abstract base class that extends from :class:`.BaseNeighborhood` and provides methods
    to fuzzify a the neighborhood. This is a baseline to create neighborhoods to be used
    with fuzzy models such as :class:`.FDT` and used in *Explainers* based on those models
    such as :class:`.FDTExplainer`.
:class:`.SimpleNeighborhood`
    Baseline functional neighborhood that replicates an instance as many times as required
    without modifying it.
:class:`.LoreNeighborhood`
    Fuzzy adaptation of the neighborhood defined for the LORE_ algorithm, which uses a genetic
    generation to obtain sufficient modifications of the original instance to train a white-box
    classifier

--------------------
"""
# =============================================================================
# Imports
# =============================================================================

# Legacy application
from .neighbor_generator import genetic_neighborhood
from .gpdatagenerator import calculate_feature_values, generate_data
from .genetic_algorithm import (get_feature_values, random_init, informed_init,
                                uniform_crossover, tournament_selection, replacement, genetic_algorithm)

# Local application
from ._base_neighborhood import BaseNeighborhood
from ._simple_neighborhood import SimpleNeighborhood
from ._fuzzy_neighborhood import FuzzyNeighborhood
from ._lore_neighborhood import LoreNeighborhood
from ._sampling_neighborhood import SamplingNeighborhood
from ._exceptions import NotFittedError, NotFuzzifiedError


# =============================================================================
# Public objects
# =============================================================================

__all__ = [
    "genetic_neighborhood",
    "calculate_feature_values",
    "generate_data",
    "get_feature_values",
    "random_init",
    "informed_init",
    "uniform_crossover",
    "tournament_selection",
    "replacement",
    "genetic_algorithm",
    "BaseNeighborhood",
    "FuzzyNeighborhood",
    "SimpleNeighborhood",
    "LoreNeighborhood",
    "SamplingNeighborhood",
    "NotFittedError",
    "NotFuzzifiedError"
]
