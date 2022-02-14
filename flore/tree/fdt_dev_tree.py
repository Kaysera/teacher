import numpy as np
from flore.fuzzy import fuzzy_entropy

from .fdt_tree import TreeFDT
from .base_decision_tree import BaseDecisionTree


class FDT_dev(BaseDecisionTree):
    def __init__(self, features, fuzzy_threshold=0.0001,
                 th=0.0001, max_depth=10, min_num_examples=1, prunning=True, t_norm=np.minimum, voting='agg_vote'):
        super().__init__(features, th, max_depth, min_num_examples, prunning)
        self.tree_ = TreeFDT(set(features), t_norm, voting)
        self.fuzzy_threshold = fuzzy_threshold

    def _get_max_f_gain(self, tree, fuzzy_X, y, t_norm=np.minimum, verbose=False):
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
            for value in fuzzy_X[feature]:
                child_mu[value] = t_norm(tree.mu, fuzzy_X[feature][value])
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

    def fit(self, fuzzy_X, y):
        self.tree_.mu = np.ones(len(y))
        self._partial_fit(fuzzy_X, y, self.tree_, 0)

    def _partial_fit(self, fuzzy_X, y, current_tree, current_depth):
        current_tree.level = current_depth
        att, f_gain, child_mu = self._get_max_f_gain(current_tree, fuzzy_X, y, verbose=False)
        # apply mask to y
        mask = [(x > 0) for x in current_tree.mu]
        y_masked = y[mask]

        if self._stop_met(f_gain, y_masked, current_depth):
            current_tree.is_leaf = True
            current_tree.class_value = self._get_class_value(current_tree.mu, y)
            return

        current_tree.is_leaf = False
        for value in fuzzy_X[att]:
            new_features = current_tree.features.copy()
            new_features.remove(att)
            child = TreeFDT(new_features)
            child.value = (att, value)
            child.mu = child_mu[value]
            if child.mu.sum() > 0:
                current_tree.childlist.append(child)
                self._partial_fit(fuzzy_X, y, child, current_depth + 1)
