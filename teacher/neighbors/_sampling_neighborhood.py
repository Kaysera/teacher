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

########################
# TODO: IMPORTANTE NO MERGEAR A LA RAMA MASTER HASTA NO LIMPIAR
########################

class SamplingNeighborhood(FuzzyNeighborhood):
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

    def _generate_prob_dist(self):
        cont_idx = [key for key, val in self.dataset['idx_features'].items() if val in self.dataset['continuous']]
        prob_dist = {}
        for i, col in enumerate(self.X2E.T):
            if i in cont_idx:
                vals = [x for x in np.unique(col) if x < self.instance[i] + col.std() and x > self.instance[i] - col.std()]
                dists = [np.count_nonzero(col == val) for val in vals]    
                dists = [d / sum(dists) for d in dists]
            else:
                vals = [x for x in np.unique(col)]
                dists = [np.count_nonzero(col == val) for val in vals]    
                dists = [d / sum(dists) for d in dists]
            prob_dist[i] = (vals, dists)
        
        return prob_dist
    
    def _generate_neighborhood(self):
        prob_dist = self._generate_prob_dist()
        class_values = [i for i in range(len(self.dataset['possible_outcomes']))]
        neighborhood = []
        for cv in class_values:
            neighs = 0
            while neighs < (self.size/len(class_values)):
                neigh = np.zeros(len(prob_dist))
                for i in prob_dist:
                    neigh[i] = np.random.choice(prob_dist[i][0], p=prob_dist[i][1])
                
                if self.bb.predict(np.array(neigh).reshape(1, -1)) == cv:
                    neighborhood.append(neigh)
                    neighs += 1
        
        features = [col for col in self.dataset['columns'] if col != self.class_name]
        return pd.DataFrame(np.array(neighborhood), columns=features)

    def fit(self):
        decoded_instance = []
        features = [col for col in self.dataset['columns'] if col != self.class_name]
        for i, var in enumerate(features):
            try:
                decoded_instance.append(self.dataset['label_encoder'][var].inverse_transform([self.instance[i]])[0])
            except:
                decoded_instance += [self.instance[i]]
        
        Z = self._generate_neighborhood()
        df = Z.copy()
        
        self.decoded_instance = np.array([decoded_instance], dtype='object')
        self.decoded_target = self.dataset['label_encoder'][self.class_name].inverse_transform(self.bb.predict(self.instance.reshape(1, -1)))

        for le in self.dataset['label_encoder']:
            if le != self.class_name:
                df[le] = self.dataset['label_encoder'][le].inverse_transform(df[le].astype(int))
        self._X = df
        self._y = self.bb.predict(Z)
        self._Xy = pd.concat([pd.DataFrame(self._y, columns=[self.class_name]), Z], axis=1)
        self._y_decoded = self.dataset['label_encoder'][self.class_name].inverse_transform(self._y)
    
    def fuzzify(self, get_division, **kwargs):
        super().fuzzify(get_division, **kwargs)
        self._instance_membership = dataset_membership(self.decoded_instance, self._fuzzy_variables)

    def get_y_decoded(self):
        return self._y_decoded
