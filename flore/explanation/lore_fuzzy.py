from flore.neighbors import genetic_neighborhood, calculate_feature_values
from flore.tree import ID3
from flore.utils import dataframe2explain
from flore.fuzzy import get_fuzzy_points, get_fuzzy_set_dataframe, get_fuzzy_triangle

import skfuzzy as fuzz
import numpy as np
from collections import defaultdict


class FuzzyLORE:

    def __init__(self):
        self.neighborhood = None
        self.fuzzy_neighborhood = None
        self.tree = None
        self.blackbox = None
        self.X2E = None
        self.y2E = None
        self.target = None
        self.instance = None
        self.explanation = None
        self.fuzzy_set = None
        self.fuzzy_set_test = None
        self.discrete = None
        self.continuous = None
        self.fuzzy_instance = None
        self.fuzzy_y = None
        self.fuzzy_neighborhood_y = None
        self.class_name = None
        self.outcomes_dict = None
        self.score = None
        self.X = None
        self.Z = None
        self.fhit = None
        self.fuzzy_points = None

    def fit(self, idx_record2explain, X2E, dataset, blackbox, fuzzy_labels, get_division, op,
            ng_function=genetic_neighborhood,
            discrete_use_probabilities=False,
            continuous_function_estimation=False):

        class_name = dataset['class_name']
        self.class_name = class_name
        columns = dataset['columns']
        discrete = dataset['discrete']
        self.discrete = discrete
        continuous = dataset['continuous']
        self.continuous = continuous
        self.blackbox = blackbox
        self.outcomes_dict = {i: j for i, j in enumerate(dataset['possible_outcomes'])}

        # Dataset Preprocessing
        dataset['feature_values'] = calculate_feature_values(X2E, columns, class_name, discrete, continuous, 1000,
                                                             discrete_use_probabilities, continuous_function_estimation)

        dfZ, x = dataframe2explain(X2E, dataset, idx_record2explain, blackbox)
        instance = dfZ.loc[idx_record2explain]
        self.X2E = dfZ.drop(class_name, axis=1)
        self.y2E = dfZ[class_name]
        self.target = instance[class_name]
        self.instance = instance

        # Generate Neighborhood
        df, Z = ng_function(dfZ, x, blackbox, dataset)
        self.Z = Z
        # df = pd.read_csv('neighbours.csv', index_col=0)
        # df.to_csv('neighbours.csv')

        self.neighborhood = df

        # Build Decision Tree
        X = df.drop(class_name, axis=1)
        self.X = X
        y = df[class_name]

        fuzzy_points = get_fuzzy_points(X, get_division, continuous, len(fuzzy_labels))
        self.fuzzy_points = fuzzy_points
        fuzzy_set = get_fuzzy_set_dataframe(X, get_fuzzy_triangle, fuzzy_points, continuous, fuzzy_labels)
        self.fuzzy_set = fuzzy_set
        fuzzy_X = self.fuzzify_dataset(X, fuzzy_set, self.get_categorical_fuzzy)
        self.fuzzy_neighborhood = fuzzy_X
        # fuzzy_X.to_csv('fuzzy_neighbourhood.csv')
        # y.to_csv('fuzzy_y.csv')

        X_np = fuzzy_X.values
        y_np = y.values

        # print(fuzzy_X.columns)
        # np.savetxt('X_np.csv', X_np, delimiter=',', fmt='%s')
        # np.savetxt('y_np.csv', y_np, delimiter=',', fmt='%s')

        id3_class = ID3(fuzzy_X.columns, X_np, y_np, max_depth=5, min_num_examples=10, prunning=True, th=0.00000001)
        id3_class.fit(X_np, y_np)
        self.score = id3_class.score(X_np, y_np)

        self.tree = id3_class

        self.explanation = id3_class.explainInstance(instance, idx_record2explain, fuzzy_set, discrete, verbose=False)

        self.fuzzy_set_test = get_fuzzy_set_dataframe(dfZ, get_fuzzy_triangle, fuzzy_points,
                                                      continuous, fuzzy_labels, verbose=False)

        self.fuzzy_target = self.fuzzy_inference(instance, idx_record2explain, self.fuzzy_set_test, discrete, op)

        self.fhit = self.fuzzy_target == self.target

        fuzzy_y = []

        for i in range(0, len(dfZ)):
            finstance = dfZ.loc[i]
            try:
                fy = self.fuzzy_inference(finstance, i, self.fuzzy_set_test, discrete, op)
            except:
                print(i, finstance)
                fy = None
            fuzzy_y += [fy]

        self.fuzzy_y = np.array(fuzzy_y)

        fuzzy_neighborhood_y = []

        for i in range(0, len(df)):
            finstance = df.loc[i]
            try:
                fy = self.fuzzy_inference(finstance, i, fuzzy_set, discrete, op)
            except:
                print(i, finstance)
                fy = None
            fuzzy_neighborhood_y += [fy]

        self.fuzzy_neighborhood_y = np.array(fuzzy_neighborhood_y)

        fuzzy_set_instance = get_fuzzy_set_dataframe(dfZ.loc[[idx_record2explain]], get_fuzzy_triangle,
                                                     fuzzy_points, continuous, fuzzy_labels, verbose=False)
        fuzzy_instance = self.fuzzify_dataset(dfZ.loc[[idx_record2explain]], fuzzy_set_instance,
                                              self.get_categorical_fuzzy)
        fuzzy_instance.drop(class_name, axis=1, inplace=True)
        self.fuzzy_instance = fuzzy_instance

    def get_explanation(self, operator):
        best_rule, best_score = self.get_best_rule(self.explanation, operator)
        return best_rule

    def get_score(self):
        return self.score

    def map_explanation(self, explanation, global_fuzzy_points, global_fuzzy_labels):
        global_fuzzy_set = get_fuzzy_set_dataframe(self.X, get_fuzzy_triangle, global_fuzzy_points,
                                                   self.continuous, global_fuzzy_labels)
        rules = explanation[0]

        mapped_exp = []

        for rule in rules:
            var = rule[0]
            fset = rule[1]
            universe = self.X[var].to_numpy()

            match_label = ''
            match_score = 0

            for label in global_fuzzy_set[var]:
                intersect = fuzz.fuzzy_and(universe, global_fuzzy_set[var][label], universe, self.fuzzy_set[var][fset])
                score = np.max(intersect[1])
                if score > match_score:
                    match_label = label
                    match_score = score

            mapped_exp += [(var, match_label)]

        print(f'{" AND ".join([f"{var}: {label}" for var, label in mapped_exp])} => {explanation[1]}')

    def _filter_dataframe(self, df, explanation, fuzzy_set_test={}, threshold=0.01):
        ndf = df.copy()
        for cond in explanation[0]:
            col = cond[0]
            val = cond[1]
            if col in fuzzy_set_test:
                ndf = ndf.loc[fuzzy_set_test[col][val][ndf.index] > threshold]
            else:
                ndf = ndf.loc[ndf[col] == val]

        return ndf

    def precision(self, explanation, threshold=0.01):
        ndf = self._filter_dataframe(self.X2E, explanation, self.fuzzy_set_test, threshold)
        precision = np.count_nonzero(self.fuzzy_y[ndf.index] == explanation[1]) / np.size(self.fuzzy_y[ndf.index])
        return precision

    def coverage(self, explanation, threshold=0.01):
        ndf = self._filter_dataframe(self.X2E, explanation, self.fuzzy_set_test, threshold)

        coverage = ndf.shape[0] / self.X2E.shape[0]
        return coverage

    def hit(self, explanation):
        return explanation[1] == self.target

    def get_best_rule(self, rules, op, target=None):
        best_rule = []
        best_score = 0

        for rule in rules:
            rule_score = 1
            if target is None or target == rule[1]:
                for clause in rule[0]:
                    rule_score = op([rule_score, clause[2]])

                if rule_score > best_score:
                    best_score = rule_score
                    best_rule = rule

        return (best_rule, best_score)

    def get_consensus(self, rules, op):

        scores = defaultdict(lambda: 0)

        for rule in rules:
            rule_score = 1
            for clause in rule[0]:
                rule_score = op([rule_score, clause[2]])

            scores[rule[1]] += rule_score
        try:
            return max(scores, key=lambda a: scores[a])
        except:
            return None

    def fuzzy_inference(self, instance, idx_record2explain, fuzzy_set, discrete, op):
        rules = self.tree.explainInstance(instance, idx_record2explain, fuzzy_set,
                                          discrete, verbose=False, threshold=0.0001)
        consensus = self.get_consensus(rules, op)
        return consensus

    def get_categorical_fuzzy(self, var):
        x = [var[k] for k in var]
        label = {i: j for i, j in enumerate(var)}
        return np.array([label[elem] for elem in np.argmax(x, axis=0)])

    def fuzzify_dataset(self, dataframe, fuzzy_set, fuzzify_variable):
        ndf = dataframe.copy()
        for k in fuzzy_set:
            ndf[k] = fuzzify_variable(fuzzy_set[k])
        return ndf

    def _compare_trees(self, classic_set, fuzzy_set):

        classic_y = self.blackbox.predict(classic_set)
        classic_y = np.vectorize(self.outcomes_dict.get)(classic_y)

        fuzzy_y = self.fuzzy_neighborhood_y[fuzzy_set.index]

        return np.sum(classic_y == fuzzy_y) / len(classic_y)

    def fidelity(self):
        return self._compare_trees(self.Z, self.fuzzy_neighborhood)

    def l_fidelity(self, explanation, threshold=0.01):
        l_fuzzy_set = self._filter_dataframe(self.neighborhood, explanation,
                                             fuzzy_set_test=self.fuzzy_set, threshold=threshold)
        l_classic_set = self.Z[l_fuzzy_set.index]

        return self._compare_trees(l_classic_set, l_fuzzy_set)

    def cl_fidelity(self, counterfactual, threshold=0.01):
        cl_fuzzy_set = self._filter_dataframe(self.neighborhood, counterfactual,
                                              fuzzy_set_test=self.fuzzy_set, threshold=threshold)
        cl_classic_set = self.Z[cl_fuzzy_set.index]

        return self._compare_trees(cl_classic_set, cl_fuzzy_set)

    def counterfactual(self, explanation):
        target = explanation[1]
        all_rules = self.tree.exploreTreeFn(verbose=False)
        counter_rules = []

        for rule in all_rules:
            if rule[1] != target:
                counter_rules += [rule]

        min_rule_distance = np.inf
        best_cr = []

        for counter_rule in counter_rules:
            rule_distance = self._compare_rule(explanation[0], counter_rule[0])

            if rule_distance < min_rule_distance:
                min_rule_distance = rule_distance
                best_cr = [counter_rule]

            elif rule_distance == min_rule_distance:
                best_cr += [counter_rule]

        return best_cr, min_rule_distance

    def _compare_rule(self, explanation, counter_rule):
        # TODO REFACTOR SOME DAY
        similarities = 0

        ex = {}
        cr = {}

        for elem in explanation:
            ex[elem[0]] = elem[1]

        for elem in counter_rule:
            cr[elem[0]] = elem[1]

        diffs = set([])

        for elem in ex:
            if elem in cr and ex[elem] == cr[elem]:
                similarities += 1
            else:
                diffs.add(elem)

        for elem in cr:
            if elem in ex and ex[elem] == cr[elem]:
                similarities += 1
            else:
                diffs.add(elem)

        return len(diffs)
