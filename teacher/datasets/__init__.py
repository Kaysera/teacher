"""
The :mod:`teacher.datasets` module includes the different databases
used to run experiments with the `Teacher` package
"""

# =============================================================================
# Imports
# =============================================================================

# Local application
from ._base import load_german, load_adult, load_compas, load_heloc, load_beer, load_pima, load_breast


# =============================================================================
# Public objects
# =============================================================================

# Set the classes that are accessible
# from the module teacher.datasets
__all__ = [
    "load_german",
    "load_adult",
    "load_compas",
    "load_heloc",
    "load_beer",
    "load_pima",
    "load_breast"
]
