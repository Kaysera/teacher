# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np

# Local application
from ._factual_local_explainer import FactualLocalExplainer
from teacher.explanation import FID3_factual, FID3_counterfactual
from teacher.tree import ID3

# =============================================================================
# Classes
# =============================================================================


class FID3Explainer(FactualLocalExplainer):
    """This *Explainer* uses the :class:`.ID3` tree implemented in :mod:`teacher` as a white box model to
       explain a local instance of a scikit-learn compatible black box classifier. Only intended for use
       as a baseline. It uses its own set of factual and counterfactual explanation generation methods."""
    def __init__(self):
        self.local_explainer = None
        super().__init__()

    def fit(self, instance, target, neighborhood):
        """Build a FID3Explainer from the instance,
        the target and the neighborhood around the instance.

        Parameters
        ----------
        instance : array-like of shape (,n_features)
            The input instance
        target : array-like of shape (1,)
            The expected target
        neighborhood : LoreNeighborhood
            Neighborhood fitted around the instance to train
            the whitebox model
        """
        self.target = target
        X_membership = neighborhood.get_X_membership()
        X = neighborhood.get_X()
        instance_membership = neighborhood.get_instance_membership()
        y_decoded = neighborhood.get_y()

        X_membership = self._fuzzify_dataset(X, X_membership)
        self.local_explainer = ID3(X_membership.columns)

        self.local_explainer.fit(X_membership.values, y_decoded)
        rules = self.local_explainer.to_rule_based_system()
        self.exp_value = self.local_explainer.predict(instance.reshape(1, -1))[0]
        fact = FID3_factual(instance_membership, rules)
        cf, _ = FID3_counterfactual(fact, [rule for rule in rules if rule.consequent != self.exp_value])
        self.explanation = (fact, cf)

    def _get_categorical_fuzzy(self, var):
        x = [var[k] for k in var]
        label = {i: j for i, j in enumerate(var)}
        return np.array([label[elem] for elem in np.argmax(x, axis=0)])

    def _fuzzify_dataset(self, dataframe, fuzzy_set):
        ndf = dataframe.copy()
        for k in fuzzy_set:
            ndf[k] = self._get_categorical_fuzzy(fuzzy_set[k])
        return ndf
