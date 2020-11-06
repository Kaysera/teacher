import numpy as np
from flore.fuzzy import fuzzy_entropy


class TreeFDT:
    def __init__(self, features):
        self.is_leaf = True
        self.childlist = []
        self.features = features
        self.class_value = -1
        self.level = -1
        self.value = (0, 0)
        self.mu = []

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
            output += '\n'+'\t' * self.level
        return output


class FDT:
    def __init__(self, features, fuzzy_set_df, fuzzy_threshold=0.0001,
                 th=0.0001, max_depth=10, min_num_examples=1, prunning=True):
        self.max_depth = max_depth
        self.tree = TreeFDT(set(features))
        self.min_num_examples = min_num_examples
        self.prunning = prunning
        self.th = th
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_set_df = fuzzy_set_df

    def fit(self, X, y):
        self.tree.mu = np.ones(len(y))
        self.partial_fit(X, y, self.tree, 0)

    def get_max_f_gain(self, tree, y, t_norm=np.minimum):
        best_att = ''
        best_f_gain = 0
        best_child_mu = {}
        for feature in tree.features:
            # print(f'Feature: {feature}')
            f_ent = fuzzy_entropy(tree.mu, y.to_numpy())
            child_mu = {}
            wef = 0  # Weighted Fuzzy Entropy
            for value in self.fuzzy_set_df[feature]:
                # print('------------------------------')
                # print(f'value: {value}')
                child_mu[value] = t_norm(tree.mu, self.fuzzy_set_df[feature][value])
                fuzzy_cardinality = child_mu[value].sum()
                crisp_cardinality = (child_mu[value] > 0).sum()
                # print(f'child_mu: {child_mu[value]}')
                # print(f'y: {y.to_numpy()}')
                child_f_ent = fuzzy_entropy(child_mu[value], y.to_numpy(), verbose=False)
                # print(f'child_f_ent: {child_f_ent}')
                wef += fuzzy_cardinality / crisp_cardinality * child_f_ent

            # print(f_ent, wef)
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

    def get_class_value(self, mu, y):
        cv = {}
        for class_value in np.unique(y):
            mask = y == class_value
            cv[class_value] = (mu * mask).sum()
        # RETURN THE ELEMENT WITH THE MAXIMUM SUM OF PERTENENCES (AGGREGATED VOTE)
        return max(cv, key=lambda x: cv[x])

    def partial_fit(self, X, y, current_tree, current_depth):
        current_tree.level = current_depth
        att, f_gain, child_mu = self.get_max_f_gain(current_tree, y)
        # print(current_tree.value)
        # print(current_tree.mu)
        # apply mask to y
        mask = [(x > 0) for x in current_tree.mu]
        y_masked = y.to_numpy()[mask]

        if self.stop_met(f_gain, y_masked, current_depth):
            current_tree.is_leaf = True

            if current_tree.value == (0, 0):
                # CASO NODO RAIZ
                current_tree.class_value = 0
            else:
                current_tree.class_value = self.get_class_value(current_tree.mu, y)
                # print(current_tree.class_value)
            return

        current_tree.is_leaf = False
        for value in self.fuzzy_set_df[att]:
            new_features = current_tree.features.copy()
            new_features.remove(att)
            child = TreeFDT(new_features)
            child.value = (att, value)
            child.mu = child_mu[value]
            current_tree.childlist.append(child)
            self.partial_fit(X, y, child, current_depth+1)

    def predict(self, fuzzy_X):

        X_size = len(list(list(fuzzy_X.values())[0].values())[0])

        all_classes = self.partial_predict(fuzzy_X, np.ones(X_size), self.tree)
        # TODO: POR DIOS REHACER ESTE ONE-LINER MAGICO
        # INPUT: all_classes = [('one', [1,2,3,4]), ('two', [4,3,2,1]), ('three', [0,0,0,9])]
        # OUTPUT: ['two', 'two', 'one', 'three']
        # return all_classes
        classes_list = [i for (i, j) in [max(x, key=lambda a: a[1]) for x in list(zip(*[[(x[0], y) for y in x[1]] for x in all_classes]))]]

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
