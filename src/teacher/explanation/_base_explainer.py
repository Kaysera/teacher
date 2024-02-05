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
        """Builds the explainer with the necessary parameters"""

    def explain(self):
        """Return the explanation from a fitted Explainer

        Returns
        -------
        tuple, (factual, counterfactual)
            Tuple with factual and counterfactual explanations.

        Raises
        ------
        NotFittedError
            When the Explainer is not fitted
        """
        if self.explanation is None:
            raise NotFittedError
        return self.explanation
