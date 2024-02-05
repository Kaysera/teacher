# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pandas as pd
from scipy import stats

# Local application
from ._fuzzy_neighborhood import FuzzyNeighborhood
from teacher.fuzzy import dataset_membership
from teacher.metrics._counterfactual import _closest_instance
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

    def __init__(self, instance, size, class_name, bb, dataset, X2E, idx_record_to_explain, neighbor_generation='slow', neighbor_range='std'):
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
        self.neighbor_generation = neighbor_generation
        self.neighbor_range = neighbor_range
        super().__init__(instance, size, class_name, bb)

    def _generate_prob_dist(self, instance, cont_idx):
        prob_dist = {}
        for i, col in enumerate(self.X2E.T):
            if i in cont_idx:
                if self.neighbor_range == 'std':
                    col_range = col.std()
                # check if neighbor_range is between 0 and 1
                elif isinstance(self.neighbor_range, float) and self.neighbor_range > 0 and self.neighbor_range < 1:
                    col_range = (col.max() - col.min()) * self.neighbor_range
                else:
                    raise ValueError("Neighbor range must be between 0 and 1 or 'std'")
                vals = [x for x in np.unique(col) if x < instance[i] + col_range and x > instance[i] - col_range]
                dists = [np.count_nonzero(col == val) for val in vals]    
                dists = [d / sum(dists) for d in dists]
            else:
                vals = [x for x in np.unique(col)]
                dists = [np.count_nonzero(col == val) for val in vals]    
                dists = [d / sum(dists) for d in dists]
            prob_dist[i] = (vals, dists)
        
        return prob_dist

    def _get_instance_from_prob_dist(self, prob_dist):
        neigh = np.zeros(len(prob_dist))
        for i in prob_dist:
            neigh[i] = np.random.choice(prob_dist[i][0], p=prob_dist[i][1])
        return neigh

    def _generate_neighborhood_fast(self):
        cont_idx = [key for key, val in self.dataset['idx_features'].items() if val in self.dataset['continuous']]
        disc_idx = [key for key, val in self.dataset['idx_features'].items() if val in self.dataset['discrete']]
        prob_dist = self._generate_prob_dist(self.instance, cont_idx)
        target = self.bb.predict(self.instance.reshape(1, -1))
        y_train_pred = self.bb.predict(self.X2E)
        closest_instance = _closest_instance(self.instance, self.X2E[y_train_pred != target], cont_idx, disc_idx, None, distance='mixed')
        c_prob_dist = self._generate_prob_dist(closest_instance, cont_idx)
        class_values = {i: 0 for i in range(len(self.dataset['possible_outcomes']))}
        neighborhood = []
        tries = 0
        while len(neighborhood) < self.size:
            tries += 1
            i_neigh = self._get_instance_from_prob_dist(prob_dist)
            c_neigh = self._get_instance_from_prob_dist(c_prob_dist)

            neigh_pred = self.bb.predict(np.array(i_neigh).reshape(1, -1))[0]
            if class_values[neigh_pred] < (self.size/len(class_values)):
                class_values[neigh_pred] += 1
                neighborhood.append(i_neigh)
            
            neigh_pred = self.bb.predict(np.array(c_neigh).reshape(1, -1))[0]
            if class_values[neigh_pred] < (self.size/len(class_values)):
                class_values[neigh_pred] += 1
                neighborhood.append(c_neigh)
            if tries > self.size * 100:
                break
        neighborhood.append(self.instance)
        features = [col for col in self.dataset['columns'] if col != self.class_name]
        return pd.DataFrame(np.array(neighborhood), columns=features)
    
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
        NEIGH_GENERATION = {
            'slow': self._generate_neighborhood,
            'fast': self._generate_neighborhood_fast
        }
        decoded_instance = []
        features = [col for col in self.dataset['columns'] if col != self.class_name]
        for i, var in enumerate(features):
            try:
                decoded_instance.append(self.dataset['label_encoder'][var].inverse_transform(np.array([self.instance[i]], dtype=int))[0])
            except:
                decoded_instance += [self.instance[i]]

        Z = NEIGH_GENERATION[self.neighbor_generation]()
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
        # AS INSTANCE MEMBERSHIP IS COMPUTED HERE, PASS FLAG TO NOT COMPUTE IT BEFORE
        super().fuzzify(get_division, instance_membership=False, **kwargs)
        self._instance_membership = dataset_membership(self.decoded_instance, self._fuzzy_variables)

    def get_y_decoded(self):
        return self._y_decoded
