# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
from sklearn.utils import check_X_y

# Local application
from ..fuzzy import dataset_membership
from ..fuzzy._discretize import _fuzzy_entropy
from .base_decision_tree import BaseDecisionTree
from .rule import Rule
from . import _voting


# =============================================================================
# Constants
# =============================================================================


VOTING_METHODS = {
    "agg_vote": _voting._aggregated_vote,
    "max_match": _voting._maximum_matching
}


# =============================================================================
# Classes
# =============================================================================

class TreeFDT:
    def __init__(self, fuzzy_variable, t_norm=np.minimum, voting='agg_vote'):
        self.is_leaf = True
        self.childlist = []
        self.class_value = -1
        self.level = -1
        self.value = None
        self.fuzzy_variable = fuzzy_variable
        self.mu = []
        self.t_norm = t_norm
        try:
            self._voting_method = VOTING_METHODS[voting]
        except KeyError:
            raise ValueError(f'{voting} voting method not implemented')

    def __str__(self):  # pragma: no cover
        output = '\t' * self.level
        try:
            output += 'Feature ' + str(self.fuzzy_variable.name) + ' '
            output += str(self.fuzzy_variable.fuzzy_sets[self.value[1]].name) + '\n'
        except Exception:  # TODO CHANGE FOR PROPER EXCEPTION
            output += 'Root \n'
        if self.is_leaf:
            output += '\t' * self.level + 'Class value: ' + str(self.class_value)
        else:
            for child in self.childlist:
                # output += '\t' * self.level
                # output += 'Feature ' + str(child.fuzzy_variable.name) + ' ' + str(child.fuzzy_variable.fuzzy_sets)
                # output += 'Feature ' + str(child.fuzzy_variable.fuzzy_sets[child.value[1]])
                output += str(child)
            output += '\n' + '\t' * self.level
        return output + '\n'

    def __eq__(self, other):
        if not isinstance(other, TreeFDT):
            return False
        return (self.is_leaf == other.is_leaf and
                self.childlist == other.childlist and
                self.class_value == other.class_value and
                self.level == other.level and
                self.value == other.value and
                np.array_equal(self.mu, other.mu))

    def predict(self, X):
        leaf_values = self._partial_predict(X, np.ones(len(X)), self)
        agg_vote = self._voting_method(leaf_values)
        all_classes = [(key, agg_vote[key]) for key in agg_vote]
        # TODO: REHACER AGG_VOTE PARA QUE EN VEZ DE ('one', [1]) SEA ('one', 1)
        # INPUT: all_classes = [('one', [1,2,3,4]), ('two', [4,3,2,1]), ('three', [0,0,0,9])]
        # OUTPUT: ['two', 'two', 'one', 'three']
        # classes_list = max(n_all_classes, key=lambda a: a[1])[0]
        weight_array = np.array([ac[1] for ac in all_classes])
        best_class = np.argmax(weight_array, axis=0)
        classes_list = [all_classes[idx][0] for idx in best_class]
        return classes_list

    def _partial_predict(self, X, mu, tree):
        if tree.value is not None:
            att, value = tree.value
            try:
                fuzzy_set = tree.fuzzy_variable.fuzzy_sets[value]
                pert_degree = fuzzy_set.membership(X[:, att])
            except KeyError:
                pert_degree = 0
            new_mu = self.t_norm(mu, pert_degree)
        else:
            new_mu = mu
        if tree.is_leaf:
            return np.array([(tree.class_value, new_mu)], dtype=object)
        else:
            return np.concatenate([self._partial_predict(X, new_mu, child) for child in tree.childlist])

    def to_rule_based_system(self, th=0.0001, verbose=False):
        rules = self._get_rules(self, [], th, verbose)
        return [Rule(antecedent, consequent, weight) for (antecedent, consequent, weight) in rules]

    def _get_rules(self, tree, rule, th=0.0001, verbose=False):
        if tree.value is not None:
            att, value = tree.value
            fuzzy_var_name = tree.fuzzy_variable.name
            fuzzy_set_name = tree.fuzzy_variable.fuzzy_sets[value].name
            clause = (fuzzy_var_name, fuzzy_set_name)
            new_rule = rule + [clause]
        else:
            new_rule = rule

        if tree.is_leaf:
            if verbose:  # pragma: no cover
                for leaf_class, weight in tree.class_value.items():
                    if weight > th:
                        print(f'{new_rule} => Class value: {leaf_class} (Weight: {weight})')
            return [(new_rule, leaf_class, weight) for leaf_class, weight in tree.class_value.items() if weight > th]
        else:
            current_rules = []
            for child in tree.childlist:
                child_rules = self._get_rules(child, new_rule, th, verbose)
                current_rules += child_rules
            return current_rules


class FDT(BaseDecisionTree):
    def __init__(self, fuzzy_variables, fuzzy_threshold=0.0001,
                 th=0.0001, max_depth=10, min_num_examples=1, prunning=True, t_norm=np.minimum, voting='agg_vote'):
        """
        Parameters
        ----------
        fuzzy_variables : list[FuzzyVariable]
            List of the fuzzy variables used in the tree
        fuzzy_threshold : float, optional
            Minimum threshold for a pertenence degree to activate a fuzzy set, by default 0.0001
        th : float, optional
            Minimum gain threshold to keep branching the tree, by default 0.0001
        max_depth : int, optional
            Maximum depth of the tree, by default 2
        min_num_examples : int, optional
            Minimum number of examples per leaf, by default 1
        prunning : bool, optional
            Whether or not to prune the tree, by default True
        t_norm : function, optional
            function to be used as T-norm, by default np.minimum
        voting : str, optional
            {'agg_vote', 'max_match'}, method of voting for the inference, by default 'agg_vote'
        """

        features = [fuzzy_var.name for fuzzy_var in fuzzy_variables]
        self.features_dict = {feat: i for i, feat in enumerate(features)}
        self.fuzzy_variables = fuzzy_variables
        super().__init__(set(features), th, max_depth, min_num_examples, prunning)
        self.tree_ = TreeFDT(None, t_norm, voting)
        self.fuzzy_threshold = fuzzy_threshold

    def _get_max_f_gain(self, tree, features, X_membership, y, t_norm=np.minimum, verbose=False):
        best_att = ''
        best_f_gain = 0
        best_child_mu = {}
        for feature in features:
            f_ent = _fuzzy_entropy(tree.mu, y)
            if verbose:  # pragma: no cover
                print('------------------------------')
                print(f'Feature: {feature}')
                print(f'F_ent: {f_ent}')
            child_mu = {}
            wef = 0  # Weighted Fuzzy Entropy
            crisp_cardinality = tree.mu.sum()
            for value in X_membership[feature]:
                child_mu[value] = t_norm(tree.mu, X_membership[feature][value])
                fuzzy_cardinality = child_mu[value].sum()
                child_f_ent = _fuzzy_entropy(child_mu[value], y, verbose=False)
                if verbose:  # pragma: no cover
                    print('------------------------------')
                    print(f'\tvalue: {value}')
                    # print(f'\tchild_mu: {child_mu[value]}')
                    print(f'\ty: {y}')
                    print(f'\tchild_f_ent: {child_f_ent}')
                wef += fuzzy_cardinality * child_f_ent
            wef /= crisp_cardinality
            if verbose:  # pragma: no cover
                print(f'Weighted Fuzzy Entropy: {wef}')
                print(f'Crisp cardinality: {crisp_cardinality}')
            f_gain = f_ent - wef

            if f_gain > best_f_gain:
                best_f_gain = f_gain
                best_att = feature
                best_child_mu = child_mu

        return (best_att, best_f_gain, best_child_mu)

    def _stop_met(self, f_gain, y_masked, level):
        if len(np.unique(y_masked)) < 2:
            return True
        if len(y_masked) < self.min_num_examples:
            return True
        if level >= self.max_depth:
            return True
        if f_gain < self.fuzzy_threshold:
            return True
        return False

    def _get_class_value(self, mu, y):
        cv = {}
        for class_value in np.unique(y):
            mask = y == class_value
            cv[class_value] = (mu * mask).sum() / mu.sum()
        # EACH LEAF HAS A DICTIONARY WITH A WEIGHT PER CLASS VALUE
        return cv

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=['float64', 'object'])
        X_membership = dataset_membership(X, self.fuzzy_variables)
        self.tree_.mu = np.ones(len(y))
        self._partial_fit(X_membership, y, self.tree_, self.features, 0)

    def _partial_fit(self, X_membership, y, current_tree, current_features, current_depth):
        current_tree.level = current_depth
        att, f_gain, child_mu = self._get_max_f_gain(current_tree, current_features, X_membership, y, verbose=False)
        # apply mask to y
        mask = [(x > 0) for x in current_tree.mu]
        y_masked = y[mask]

        if self._stop_met(f_gain, y_masked, current_depth):
            current_tree.is_leaf = True
            current_tree.class_value = self._get_class_value(current_tree.mu, y)
            return

        current_tree.is_leaf = False
        for i, value in enumerate(X_membership[att]):
            new_features = current_features.copy()
            new_features.remove(att)
            child = TreeFDT(self.fuzzy_variables[self.features_dict[att]])
            child.value = (self.features_dict[att], i)
            child.mu = child_mu[value]
            current_tree.childlist.append(child)
            if child.mu.sum() > 0:
                self._partial_fit(X_membership, y, child, new_features, current_depth + 1)
            else:
                child.is_leaf = True
                child.class_value = self._get_class_value(current_tree.mu, y)
