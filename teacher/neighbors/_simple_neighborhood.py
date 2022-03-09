from ._base_neighborhood import BaseNeighborhood
import pandas as pd


class SimpleNeighborhood(BaseNeighborhood):

    def __init__(self, instance, size, class_name, bb):
        super().__init__(instance, size, class_name, bb)

    def fit(self):
        self._X = pd.DataFrame([self.instance] * self.size)
        self._y = self.bb.predict(self._X)
