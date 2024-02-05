# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pandas as pd
from scipy import stats

# Local application
from ._fuzzy_neighborhood import FuzzyNeighborhood
from teacher.neighbors import genetic_neighborhood, calculate_feature_values
from teacher.utils import dataframe2explain
from teacher.fuzzy import dataset_membership

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

    def _smooth_neighborhood(self, df, continuous, X2E, feat_idx, label_encoder, class_value):
        ndf = df.copy()
        for col in continuous:
            min_val = X2E[:, feat_idx[col]].min()
            max_val = X2E[:, feat_idx[col]].max()
            if col == 'age':
                print(min_val)
            ndf[col].loc[ndf[col] > max_val] = max_val
            ndf[col].loc[ndf[col] < min_val] = min_val

        old_length = len(ndf)
        ndf = ndf[(np.abs(stats.zscore(ndf[continuous])) < 3).all(axis=1)]
        new_length = len(ndf)
        while new_length - old_length > 0:
            ndf = ndf[(np.abs(stats.zscore(ndf[continuous])) < 3).all(axis=1)]

        y_bb = self.bb.predict(self.instance.reshape(1, -1))
        instance_df = pd.DataFrame(np.concatenate([y_bb, self.instance]).reshape(1, -1), columns=list(ndf))
        ndf = pd.concat([ndf, instance_df])
        edf = ndf.drop(self.class_name, axis=1).copy()
        for le in label_encoder:
            if le != self.class_name:
                edf[le] = label_encoder[le].transform(edf[le])

        return ndf, edf.to_numpy()

    def fit(self):
        decoded_instance = []
        features = [col for col in self.dataset['columns'] if col != self.class_name]
        for i, var in enumerate(features):
            try:
                decoded_instance.append(self.dataset['label_encoder'][var].inverse_transform([self.instance[i]])[0])
            except KeyError:
                decoded_instance += [self.instance[i]]

        self.decoded_instance = np.array([decoded_instance], dtype='object')
        y = self.bb.predict(self.instance.reshape(1, -1))
        self.decoded_target = self.dataset['label_encoder'][self.class_name].inverse_transform(y)

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

        feat_idx = {feat: idx for idx, feat in self.dataset['idx_features'].items()}
        df, Z = self._smooth_neighborhood(df,
                                          [col for col in self.dataset['continuous'] if col != self.class_name],
                                          self.X2E,
                                          feat_idx,
                                          self.dataset['label_encoder'],
                                          self.class_name)

        self._Xy = df
        self._X = df.drop(self.class_name, axis=1)
        self._y = self.bb.predict(Z)
        self._y_decoded = df[self.class_name]

    def fuzzify(self, get_division, **kwargs):
        super().fuzzify(get_division, **kwargs)
        self._instance_membership = dataset_membership(self.decoded_instance, self._fuzzy_variables)

    def get_y_decoded(self):
        return self._y_decoded
