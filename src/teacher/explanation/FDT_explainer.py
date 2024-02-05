"""
The *Explainer* are classes that follow the guidelines of scikit-learn modules
in that they can be fitted with data to generate an explanation.
"""

# =============================================================================
# Imports
# =============================================================================

# Third party
from sklearn.utils import check_array
from sklearn.metrics import f1_score
import numpy as np

# Local application
from ._factual_local_explainer import FactualLocalExplainer
from teacher.tree import FDT
from teacher.explanation import m_factual, mr_factual, c_factual, i_counterfactual, f_counterfactual, d_counterfactual


# =============================================================================
# Constants
# =============================================================================

FACTUAL_METHODS = {
    'm_factual': m_factual,
    'mr_factual': mr_factual,
    'c_factual': c_factual
}


COUNTERFACTUAL_METHODS = {
    'i_counterfactual': i_counterfactual,
    'f_counterfactual': f_counterfactual,
    'd_counterfactual': d_counterfactual
}

# =============================================================================
# Classes
# =============================================================================


class FDTExplainer(FactualLocalExplainer):
    """This *Explainer* uses the :class:`.FDT` implemented in :mod:`teacher` as a white box model to
       explain a local instance of a scikit-learn compatible black box classifier."""
    def __init__(self):
        self.local_explainer = None
        self.factual_method = None
        self.counterfactual_method = None
        super().__init__()

    def fit(self, instance, target, neighborhood, df_num_cols, factual, counterfactual, **kwargs):
        """
        .. _article: https://doi.org/10.1109/TFUZZ.2022.3179582

        Build a FDTExplainer from the instance, the target and the neighborhood around
        the instance

        Parameters
        ----------
        instance : array-like of shape (,n_features)
            The input instance
        target : array-like of shape (1,)
            The expected target
        neighborhood : class extending from BaseNeighborhood
            Neighborhood fitted around the instance to train
            the whitebox model
        df_num_cols : array-like of shape (n_numerical) where
            n_numerical are the number of numerical columns
        factual : {"m_factual", "mr_factual", "c_factual"}
            The function to compute the factual explanation. Supported
            methods are explained in this article_.
        counterfactual : {"f_counterfactual", "i_counterfactual"}
            The function to compute the factual explanation. Supported
            methods are explained in this article_.

        Raises
        ------
        ValueError
            Factual method invalid
        ValueError
            Counterfactual method invalid
        ValueError
            'c_factual' chosen but no 'lam' parameter given

        """
        instance = check_array(instance, dtype=['float64', 'object'])
        try:
            self.factual_method = FACTUAL_METHODS[factual]
            if factual == 'c_factual' and 'lam' not in kwargs:
                raise ValueError("Lambda parameter (lam) needed for factual {factual}")
        except KeyError:
            raise ValueError(f"Factual method '{factual}' invalid")

        try:
            self.counterfactual_method = COUNTERFACTUAL_METHODS[counterfactual]
        except KeyError:
            raise ValueError(f"Counterfactual method '{counterfactual}' invalid")

        self.target = target
        fuzzy_variables = neighborhood.get_fuzzy_variables()
        instance_membership = neighborhood.get_instance_membership()
        decoded_instance = neighborhood.decoded_instance[0]
        X = neighborhood.get_X()
        y = neighborhood.get_y()

        try:
            max_depth = kwargs['max_depth']
            del kwargs['max_depth']
        except KeyError:
            max_depth = 10

        try:
            min_num_examples = kwargs['min_num_examples']
            del kwargs['min_num_examples']
        except KeyError:
            min_num_examples = 1

        try:
            fuzzy_threshold = kwargs['fuzzy_threshold']
            del kwargs['fuzzy_threshold']
        except KeyError:
            fuzzy_threshold = 0.0001

        if counterfactual == 'd_counterfactual':
            try:
                cont_idx = kwargs['cont_idx']
                del kwargs['cont_idx']
            except KeyError:
                raise ValueError('Continuous index needed for d_counterfactual')

            try:
                disc_idx = kwargs['disc_idx']
                del kwargs['disc_idx']
            except KeyError:
                raise ValueError('Discrete index needed for d_counterfactual')

            try:
                mad = kwargs['mad']
                del kwargs['mad']
            except KeyError:
                raise ValueError('MAD needed for d_counterfactual')

            try:
                cf_dist = kwargs['cf_dist']
                del kwargs['cf_dist']
            except KeyError:
                cf_dist = 'moth'

        self.local_explainer = FDT(fuzzy_variables,
                                   max_depth=max_depth,
                                   min_num_examples=min_num_examples,
                                   fuzzy_threshold=fuzzy_threshold)
        self.local_explainer.fit(X, y)
        local_prediction = self.local_explainer.predict(X)[0]
        if len(np.unique(y)) > 2:
            self.fidelity = f1_score(y, local_prediction, average='weighted')
        else:
            self.fidelity = f1_score(y, local_prediction)

        rules = self.local_explainer.to_rule_based_system()
        self.exp_value = self.local_explainer.predict(instance.reshape(1, -1))
        fact = self.factual_method(instance_membership, rules, self.exp_value, **kwargs)
        if counterfactual == 'i_counterfactual':
            cf = self.counterfactual_method(instance_membership, rules, self.exp_value, df_num_cols)
        elif counterfactual == 'f_counterfactual':
            cf = self.counterfactual_method(fact, instance_membership, rules, self.exp_value, df_num_cols)
        elif counterfactual == 'd_counterfactual':
            cf = self.counterfactual_method(decoded_instance,
                                            instance_membership,
                                            rules,
                                            self.exp_value,
                                            cont_idx,
                                            disc_idx,
                                            mad,
                                            cf_dist)
        self.explanation = (fact, cf)
