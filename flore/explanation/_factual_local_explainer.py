from ._base_explainer import BaseExplainer
from flore.neighbors import NotFittedError


class FactualLocalExplainer(BaseExplainer):
    def __init__(self):
        self.target = None
        self.exp_value = None
        super().__init__()

    def write_explanation(self):
        """TODO"""

    def hit(self):
        if self.exp_value is None or self.target is None:
            raise(NotFittedError)
        return self.exp_value == self.target
