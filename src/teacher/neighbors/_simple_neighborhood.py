# =============================================================================
# Imports
# =============================================================================

# Third party
import pandas as pd

# Local application
from ._base_neighborhood import BaseNeighborhood

# =============================================================================
# Classes
# =============================================================================


class SimpleNeighborhood(BaseNeighborhood):
    """
    Simple neighborhood that is formed by
    copying the instance as many times as the
    size of the neighborhood without modifying it
    """

    def __init__(self, instance, size, class_name, bb):
        super().__init__(instance, size, class_name, bb)

    def fit(self):
        self._X = pd.DataFrame([self.instance] * self.size)
        self._y = self.bb.predict(self._X)
