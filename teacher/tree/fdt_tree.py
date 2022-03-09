import numpy as np
from teacher.fuzzy import fuzzy_entropy
from collections import defaultdict

from .base_decision_tree import BaseDecisionTree
from .rule import Rule


class TreeFDT:
    def __init__(self, features, t_norm=np.minimum, voting='agg_vote'):
        self.is_leaf = True
        self.childlist = []
        self.features = features
        self.class_value = -1
        self.level = -1
        self.value = (0, 0)
        self.mu = []
        self.t_norm = t_norm
        if voting == 'agg_vote':
            self._voting_method = self._aggregated_vote
        elif voting == 'max_match':
            self._voting_method = self._maximum_matching
        else:
            raise ValueError('Voting method not implemented')

    def __str__(self):  # pragma: no cover
        output = '\t' * self.level
        if(self.is_leaf):
            output += 'Class value: ' + str(self.class_value)
        else:
            output += 'Feature ' + str(self.value)
            for child in self.childlist:
                output += '\n'
                output += '\t' * self.level
                output += 'Feature ' + str(child.value)
                output += '\n' + str(child)
            output += '\n' + '\t' * self.level
        return output

    def __eq__(self, other):
        if not isinstance(other, TreeFDT):
            return False
        return (self.is_leaf == other.is_leaf and
                self.childlist == other.childlist and
                self.features == other.features and
                self.class_value == other.class_value and
                self.level == other.level and
                self.value == other.value and
                np.array_equal(self.mu, other.mu))

    def _aggregated_vote(self, all_classes):
        agg_vote = defaultdict(lambda: np.zeros(len(all_classes[0][1])))
        for leaf in all_classes:
            for key in leaf[0]:
                agg_vote[key] += leaf[0][key] * leaf[1]
        return agg_vote

    def _maximum_matching(self, all_classes):
        max_match = defaultdict(lambda: np.zeros(len(all_classes[0][1])))
        for leaf in all_classes:
            for key in leaf[0]:
                max_match[key] = np.maximum(max_match[key], (leaf[0][key] * leaf[1]))
        return max_match

    def predict(self, X_membership):
        # Get the length of the array to predict
        X_size = 1
        leaf_values = self._partial_predict(X_membership, np.ones(X_size), self)
        agg_vote = self._voting_method(leaf_values)
        # all_classes = [(key, agg_vote[key]) for key in agg_vote]
        n_all_classes = [(key, agg_vote[key][0]) for key in agg_vote]
        # TODO: REHACER AGG_VOTE PARA QUE EN VEZ DE ('one', [1]) SEA ('one', 1)
        # INPUT: all_classes = [('one', [1,2,3,4]), ('two', [4,3,2,1]), ('three', [0,0,0,9])]
        # OUTPUT: ['two', 'two', 'one', 'three']
        classes_list = max(n_all_classes, key=lambda a: a[1])[0]
        return classes_list

    def _partial_predict(self, X_membership, mu, tree):
        if tree.value != (0, 0):
            att, value = tree.value
            try:
                pert_degree = X_membership[att][value]
            except KeyError:
                pert_degree = 0
            new_mu = self.t_norm(mu, pert_degree)
        else:
            new_mu = mu
        if tree.is_leaf:
            return [(tree.class_value, new_mu)]
        else:
            return np.concatenate([self._partial_predict(X_membership, new_mu, child) for child in tree.childlist])

    def to_rule_based_system(self, th=0.0001, verbose=False):
        rules = self._get_rules(self, [], th, verbose)
        return [Rule(antecedent, consequent, weight) for (antecedent, consequent, weight) in rules]

    def _get_rules(self, tree, rule, th=0.0001, verbose=False):
        if tree.value != (0, 0):
            att, value = tree.value
            clause = (att, value)
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
    def __init__(self, features, fuzzy_threshold=0.0001,
                 th=0.0001, max_depth=10, min_num_examples=1, prunning=True, t_norm=np.minimum, voting='agg_vote'):
        super().__init__(features, th, max_depth, min_num_examples, prunning)
        self.tree_ = TreeFDT(set(features), t_norm, voting)
        self.fuzzy_threshold = fuzzy_threshold

    def _get_max_f_gain(self, tree, X_membership, y, t_norm=np.minimum, verbose=False):
        best_att = ''
        best_f_gain = 0
        best_child_mu = {}
        for feature in tree.features:
            f_ent = fuzzy_entropy(tree.mu, y)
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
                child_f_ent = fuzzy_entropy(child_mu[value], y, verbose=False)
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

    def fit(self, X_membership, y):
        self.tree_.mu = np.ones(len(y))
        self._partial_fit(X_membership, y, self.tree_, 0)

    def _partial_fit(self, X_membership, y, current_tree, current_depth):
        current_tree.level = current_depth
        att, f_gain, child_mu = self._get_max_f_gain(current_tree, X_membership, y, verbose=False)
        # apply mask to y
        mask = [(x > 0) for x in current_tree.mu]
        y_masked = y[mask]

        if self._stop_met(f_gain, y_masked, current_depth):
            current_tree.is_leaf = True
            current_tree.class_value = self._get_class_value(current_tree.mu, y)
            return

        current_tree.is_leaf = False
        for value in X_membership[att]:
            new_features = current_tree.features.copy()
            new_features.remove(att)
            child = TreeFDT(new_features)
            child.value = (att, value)
            child.mu = child_mu[value]
            if child.mu.sum() > 0:
                current_tree.childlist.append(child)
                self._partial_fit(X_membership, y, child, current_depth + 1)
