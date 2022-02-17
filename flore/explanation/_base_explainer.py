from abc import ABC, abstractmethod
from flore.neighbors import NotFittedError


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
