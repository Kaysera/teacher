"""Base Explainer"""

# =============================================================================
# Imports
# =============================================================================

# Standard library
from abc import ABC, abstractmethod

# Local application
from teacher.neighbors import NotFittedError


# =============================================================================
# Classes
# =============================================================================
class BaseExplainer(ABC):
    def __init__(self):
        self.explanation = None

    @abstractmethod
    def fit(self):
        """
        Perform the operations to obtain the explanation
        """

    def explain(self):
        if self.explanation is None:
            raise(NotFittedError)
        return self.explanation
