# =============================================================================
# Imports
# =============================================================================

# Local application
from ._fuzzy_neighborhood import FuzzyNeighborhood
from teacher.neighbors import genetic_neighborhood, calculate_feature_values
from teacher.utils import dataframe2explain

# =============================================================================
# Classes
# =============================================================================


class LoreNeighborhood(FuzzyNeighborhood):
    """
    Fuzzy adaptation of the neighborhood used by LORE, which
    generates the different elements by modifying the instance
    using a genetic algorithm in order to obtain elements
    for all the different possible class values.
    """

    def __init__(self, instance, size, class_name, bb, dataset, X2E, idx_record_to_explain):
        """
        Parameters
        ----------
        instance : array-like, of shape (n_features)
            Instance to generate the neighborhood from
        size : int
            Size of the neighborhood
        class_name : str
            Name of the feature that is the class value
        bb : scikit-learn compatible predictor
            Black-box already fitted with the input data.
        X2E : LORE styled dataset - Legacy
            Necessary dataset for the LORE genetic algorithm to work
        idx_record_to_explain : int - Legacy
            Index of the instance to explain in X2E
        """
        self.X2E = X2E
        self.dataset = dataset
        self.idx_record_to_explain = idx_record_to_explain
        super().__init__(instance, size, class_name, bb)

    def fit(self):
        # Dataset Preprocessing
        self.dataset['feature_values'] = calculate_feature_values(self.X2E,
                                                                  self.dataset['columns'],
                                                                  self.class_name,
                                                                  self.dataset['discrete'],
                                                                  self.dataset['continuous'],
                                                                  self.size,
                                                                  False,
                                                                  False)

        dfZ, x = dataframe2explain(self.X2E, self.dataset, self.idx_record_to_explain, self.bb)

        # Generate Neighborhood
        df, Z = genetic_neighborhood(dfZ, x, self.bb, self.dataset, self.size)

        self._Xy = df
        self._X = df.drop(self.class_name, axis=1)
        self._y = self.bb.predict(Z)
        self._y_decoded = df[self.class_name]

    def get_y_decoded(self):
        return self._y_decoded
