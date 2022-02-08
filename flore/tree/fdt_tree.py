from functools import reduce
import numpy as np
from flore.fuzzy import fuzzy_entropy
from collections import defaultdict
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

    def __str__(self):
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
                self.mu == other.mu)

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

    def predict(self, fuzzy_X):
        # Get the length of the array to predict
        X_size = 1
        leaf_values = self._partial_predict(fuzzy_X, np.ones(X_size), self)
        agg_vote = self._voting_method(leaf_values)
        # all_classes = [(key, agg_vote[key]) for key in agg_vote]
        n_all_classes = [(key, agg_vote[key][0]) for key in agg_vote]
        # TODO: REHACER AGG_VOTE PARA QUE EN VEZ DE ('one', [1]) SEA ('one', 1)
        # INPUT: all_classes = [('one', [1,2,3,4]), ('two', [4,3,2,1]), ('three', [0,0,0,9])]
        # OUTPUT: ['two', 'two', 'one', 'three']
        classes_list = max(n_all_classes, key=lambda a: a[1])[0]
        return classes_list

    def _partial_predict(self, fuzzy_X, mu, tree):
        if tree.value != (0, 0):
            att, value = tree.value
            new_mu = self.t_norm(mu, fuzzy_X[att][value])
        else:
            new_mu = mu
        if tree.is_leaf:
            return [(tree.class_value, new_mu)]
        else:
            return np.concatenate([self._partial_predict(fuzzy_X, new_mu, child) for child in tree.childlist])

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
            if verbose:
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


class FDT:
    def __init__(self, features, fuzzy_set_df, fuzzy_threshold=0.0001,
                 th=0.0001, max_depth=10, min_num_examples=1, prunning=True, voting='agg_vote'):
        self.max_depth = max_depth
        self.tree = TreeFDT(set(features))
        self.min_num_examples = min_num_examples
        self.prunning = prunning
        self.th = th
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_set_df = fuzzy_set_df
        if voting == 'agg_vote':
            self.voting_method = self.aggregated_vote
        elif voting == 'max_match':
            self.voting_method = self.maximum_matching
        else:
            raise Exception

    def get_max_f_gain(self, tree, y, t_norm=np.minimum, verbose=False):
        best_att = ''
        best_f_gain = 0
        best_child_mu = {}
        for feature in tree.features:
            f_ent = fuzzy_entropy(tree.mu, y.to_numpy())
            if verbose:
                print('------------------------------')
                print(f'Feature: {feature}')
                print(f'F_ent: {f_ent}')
            child_mu = {}
            wef = 0  # Weighted Fuzzy Entropy
            crisp_cardinality = tree.mu.sum()
            for value in self.fuzzy_set_df[feature]:
                child_mu[value] = t_norm(tree.mu, self.fuzzy_set_df[feature][value])
                fuzzy_cardinality = child_mu[value].sum()
                child_f_ent = fuzzy_entropy(child_mu[value], y.to_numpy(), verbose=False)
                if verbose:
                    print('------------------------------')
                    print(f'\tvalue: {value}')
                    # print(f'\tchild_mu: {child_mu[value]}')
                    print(f'\ty: {y.to_numpy()}')
                    print(f'\tchild_f_ent: {child_f_ent}')
                wef += fuzzy_cardinality * child_f_ent
            wef /= crisp_cardinality
            if verbose:
                print(f'Weighted Fuzzy Entropy: {wef}')
                print(f'Crisp cardinality: {crisp_cardinality}')
            f_gain = f_ent - wef

            if f_gain > best_f_gain:
                best_f_gain = f_gain
                best_att = feature
                best_child_mu = child_mu

        return (best_att, best_f_gain, best_child_mu)

    def stop_met(self, f_gain, y_masked, level):
        if len(np.unique(y_masked)) < 2:
            return True
        if len(y_masked) < self.min_num_examples:
            return True
        if level >= self.max_depth:
            return True
        if f_gain < self.fuzzy_threshold:
            return True
        return False

    def get_class_value(self, mu, y):
        cv = {}
        for class_value in np.unique(y):
            mask = y == class_value
            cv[class_value] = (mu * mask).sum() / mu.sum()
        # EACH LEAF HAS A DICTIONARY WITH A WEIGHT PER CLASS VALUE
        return cv

    def fit(self, X, y):
        self.tree.mu = np.ones(len(y))
        self.partial_fit(X, y, self.tree, 0)

    def partial_fit(self, X, y, current_tree, current_depth):
        current_tree.level = current_depth
        att, f_gain, child_mu = self.get_max_f_gain(current_tree, y, verbose=False)
        # print(current_tree.value)
        # print(current_tree.mu)
        # apply mask to y
        mask = [(x > 0) for x in current_tree.mu]
        y_masked = y.to_numpy()[mask]

        if self.stop_met(f_gain, y_masked, current_depth):
            current_tree.is_leaf = True
            current_tree.class_value = self.get_class_value(current_tree.mu, y)
            return

        current_tree.is_leaf = False
        for value in self.fuzzy_set_df[att]:
            new_features = current_tree.features.copy()
            new_features.remove(att)
            child = TreeFDT(new_features)
            child.value = (att, value)
            child.mu = child_mu[value]
            if child.mu.sum() > 0:
                current_tree.childlist.append(child)
                self.partial_fit(X, y, child, current_depth + 1)

    def aggregated_vote(self, all_classes):
        agg_vote = defaultdict(lambda: np.zeros(len(all_classes[0][1])))
        for leaf in all_classes:
            for key in leaf[0]:
                agg_vote[key] += leaf[0][key] * leaf[1]
        return agg_vote

    def maximum_matching(self, all_classes):
        max_match = defaultdict(lambda: np.zeros(len(all_classes[0][1])))
        for leaf in all_classes:
            for key in leaf[0]:
                max_match[key] = np.maximum(max_match[key], (leaf[0][key] * leaf[1]))
        return max_match

    def predict(self, fuzzy_X, t_norm=np.minimum):
        # Get the length of the array to predict
        try:
            X_size = len(list(list(fuzzy_X.values())[0].values())[0])
        except TypeError:
            X_size = 1

        leaf_values = self.partial_predict(fuzzy_X, np.ones(X_size), self.tree, t_norm)
        agg_vote = self.voting_method(leaf_values)
        all_classes = [(key, agg_vote[key]) for key in agg_vote]
        # TODO: REHACER ESTE ONE-LINER MAGICO
        # INPUT: all_classes = [('one', [1,2,3,4]), ('two', [4,3,2,1]), ('three', [0,0,0,9])]
        # OUTPUT: ['two', 'two', 'one', 'three']
        classes_list = [i for (i, j) in
                        [max(x, key=lambda a: a[1]) for x in
                         list(zip(*[[(x[0], y) for y in x[1]] for x in all_classes]))]]

        return classes_list

    def partial_predict(self, fuzzy_X, mu, tree, t_norm=np.minimum):
        if tree.value != (0, 0):
            att, value = tree.value
            new_mu = t_norm(mu, fuzzy_X[att][value])
        else:
            new_mu = mu
        if tree.is_leaf:
            return [(tree.class_value, new_mu)]
        else:
            return np.concatenate([self.partial_predict(fuzzy_X, new_mu, child, t_norm) for child in tree.childlist])

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / y.shape[0]

    def explain(self, fuzzy_X, class_value, n_rules='all', t_norm=np.minimum):
        """Explains a single instance by returning a number of rules
        that correspond to the class value of that instance

        Parameters
        ----------
        fuzzy_X : [type]
            Instance to explain
        class_value : [type]
            Target value to explain
        n_rules : str of int, optional
            Number of rules to return, by default 'all'
        t_norm : function, optional
            Numpy function to use as a T-norm, by default np.minimum

        Returns
        -------
        list
            List with the rules that explain the instance and their
            degree of pertenence, in the format:
            [degree, [(attribute_1, value_1), (attribute_2, value_2)...]]
        """
        # ONLY VALID TO EXPLAIN A SINGLE INSTANCE

        rules_list = self.partial_explain(fuzzy_X, 1, self.tree, class_value, [], t_norm)
        rules_list = sorted(rules_list, key=lambda rule: rule[1], reverse=True)

        if n_rules == 'all':
            return rules_list
        else:
            return rules_list[:n_rules]

    def partial_explain(self, fuzzy_X, mu, tree, class_value, rule, t_norm=np.minimum, threshold=0.0001):
        if tree.value != (0, 0):
            att, value = tree.value
            new_mu = t_norm(mu, fuzzy_X[att][value])
            clause = (att, value)
            new_rule = rule + [clause]
        else:
            new_mu = mu
            new_rule = rule

        if tree.is_leaf:
            final_mu = new_mu * tree.class_value[class_value]
            if type(final_mu) is not np.float64:
                final_mu = final_mu[0]
            if final_mu > threshold:
                return [(tree.class_value, final_mu, new_rule)]
        else:
            current_rules = []
            for child in tree.childlist:
                child_rules = self.partial_explain(fuzzy_X, new_mu, child, class_value, new_rule, t_norm)
                if child_rules:
                    current_rules += child_rules
            return current_rules

    def get_all_rules(self, all_classes, t_norm=np.minimum):
        rules_list = []
        for class_val in all_classes:
            rules_list += self.get_cf_rules(class_val, t_norm)

        return rules_list

    def get_cf_rules(self, class_value, t_norm=np.minimum):
        # print(class_value)
        rules_list = self.partial_get_cf_rules(self.tree, class_value, [], t_norm)
        return rules_list

    def partial_get_cf_rules(self, tree, class_value, rule, t_norm=np.minimum, threshold=0.0001):
        if tree.value != (0, 0):
            att, value = tree.value
            clause = (att, value)
            new_rule = rule + [clause]
        else:
            new_rule = rule

        if tree.is_leaf:
            leaf_class = max(tree.class_value, key=lambda x: tree.class_value[x])
            # print(class_value)
            if leaf_class == class_value:
                return [new_rule]
        else:
            current_rules = []
            for child in tree.childlist:
                child_rules = self.partial_get_cf_rules(child, class_value, new_rule, t_norm)
                if child_rules:
                    current_rules += child_rules
            return current_rules

    def robust_threshold(self, element, other_classes):
        max_threshold = 0
        for class_val in other_classes:
            class_explanation = self.explain(element, class_val)
            if len(class_explanation) > 0:
                total_mu = reduce(lambda x, y: x + y[1], class_explanation, 0)
                if total_mu > max_threshold:
                    max_threshold = total_mu

        return max_threshold
