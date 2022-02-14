from ._factual_local_explainer import FactualLocalExplainer
from flore.tree import FDT_dev
from flore.explanation import m_factual, mr_factual, c_factual, i_counterfactual, f_counterfactual


class FDTExplainer(FactualLocalExplainer):
    def __init__(self):
        self.local_explainer = None
        self.factual_method = None
        self.cf_method = None
        super().__init__()

    def fit(self, instance, target, neighborhood, df_num_cols, factual, counterfactual, **kwargs):
        if factual == 'm_factual':
            self.factual_method = m_factual
        elif factual == 'mr_factual':
            self.factual_method = mr_factual
        elif factual == 'c_factual':
            self.factual_method = c_factual
            if 'lam' not in kwargs:
                raise ValueError("Lambda parameter (lam) needed")
        else:
            raise ValueError("Factual method invalid")

        if counterfactual == 'i_counterfactual':
            self.cf_method = i_counterfactual
        elif counterfactual == 'f_counterfactual':
            self.cf_method = f_counterfactual
        else:
            raise ValueError("Counterfactual method invalid")

        self.target = target
        fuzzy_X = neighborhood.get_fuzzy_X()
        fuzzy_instance = neighborhood.get_fuzzy_instance()
        y = neighborhood.get_y()

        self.local_explainer = FDT_dev(fuzzy_X.keys())
        self.local_explainer.fit(fuzzy_X, y)

        rules = self.local_explainer.to_rule_based_system()
        self.exp_value = self.local_explainer.predict(fuzzy_instance)[0]
        fact = self.factual_method(fuzzy_instance, rules, self.exp_value, **kwargs)
        if counterfactual == 'i_counterfactual':
            cf = self.cf_method(fuzzy_instance, rules, self.exp_value, df_num_cols)
        elif counterfactual == 'f_counterfactual':
            cf = self.cf_method(fact, fuzzy_instance, rules, self.exp_value, df_num_cols)
        self.explanation = (fact, cf)
