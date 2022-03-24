from ._factual_local_explainer import FactualLocalExplainer
from sklearn.utils import check_array
from teacher.tree import FDT
from teacher.explanation import m_factual, mr_factual, c_factual, i_counterfactual, f_counterfactual

FACTUAL_METHODS = {
    'm_factual': m_factual,
    'mr_factual': mr_factual,
    'c_factual': c_factual
}


COUNTERFACTUAL_METHODS = {
    'i_counterfactual': i_counterfactual,
    'f_counterfactual': f_counterfactual
}


class FDTExplainer(FactualLocalExplainer):
    def __init__(self):
        self.local_explainer = None
        self.factual_method = None
        self.counterfactual_method = None
        super().__init__()

    def fit(self, instance, target, neighborhood, df_num_cols, factual, counterfactual, **kwargs):
        instance = check_array(instance)
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
        X = neighborhood.get_X()
        y = neighborhood.get_y()

        self.local_explainer = FDT(fuzzy_variables)
        self.local_explainer.fit(X, y)

        rules = self.local_explainer.to_rule_based_system()
        self.exp_value = self.local_explainer.predict(instance)
        fact = self.factual_method(instance_membership, rules, self.exp_value, **kwargs)
        if counterfactual == 'i_counterfactual':
            cf = self.counterfactual_method(instance_membership, rules, self.exp_value, df_num_cols)
        elif counterfactual == 'f_counterfactual':
            cf = self.counterfactual_method(fact, instance_membership, rules, self.exp_value, df_num_cols)
        self.explanation = (fact, cf)
