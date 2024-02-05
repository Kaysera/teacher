"""Factual Local Explainer"""

# =============================================================================
# Imports
# =============================================================================

# Local application
from ._base_explainer import BaseExplainer
from teacher.neighbors import NotFittedError


# =============================================================================
# Classes
# =============================================================================

class FactualLocalExplainer(BaseExplainer):
    def __init__(self):
        self.target = None
        self.exp_value = None
        super().__init__()

    def write_explanation(self):
        """Writes the explanation in a human readable manner

        Returns
        -------
        str
            The NLP representation of the explanation
        """
        nlp_exp = f'The element is {self.exp_value[0]} because {self.explanation[0]}\n'
        nlp_exp += 'Otherwise, you would need '
        nlp_exp += ' and '.join([f'{cf[0]} = {cf[1]}' for cf in self.explanation[1]])

        return nlp_exp

    def hit(self):
        """Returns `True` if the explanation corresponds with the
        expected prediction

        Returns
        -------
        bool
            The explanation corresponds with the expected prediction

        Raises
        ------
        NotFittedError
            When the Explainer is not fitted
        """
        if self.exp_value is None or self.target is None:
            raise NotFittedError
        return self.exp_value == self.target
