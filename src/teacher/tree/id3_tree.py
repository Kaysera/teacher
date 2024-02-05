# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
from scipy.stats import entropy
from sklearn.utils import check_array

# Local application
from .base_decision_tree import BaseDecisionTree
from .rule import Rule

# =============================================================================
# Classes
# =============================================================================


class TreeID3:
    def __init__(self, features):
        self.var_index = -1
        self.is_leaf = True
        self.childlist = []
        self.splits = []
        self.class_value = -1
        self.class_count = 0
        self.level = -1
        self.error = 0
        self.num_leaf = 0
        self.features = features

    def __str__(self):  # pragma: no cover
        output = '\t' * self.level
        if self.is_leaf:
            output += 'Class value: ' + str(self.class_value) + '\tCounts: ' + str(self.class_count)
        else:
            output += 'Feature ' + str(self.var_index)
            # output += str(self.splits)
            i = 0
            for child in self.childlist:
                output += '\n'
                output += '\t'*self.level
                output += self.features[self.var_index] + ' = ' + str(self.splits[i])
                i = i + 1
                output += '\n'+str(child)
            output += '\n'+'\t'*self.level
        return output

    def __eq__(self, other):
        if not isinstance(other, TreeID3):
            return False
        return (self.var_index == other.var_index and
                self.is_leaf == other.is_leaf and
                self.childlist == other.childlist and
                self.splits == other.splits and
                self.class_value == other.class_value and
                self.class_count == other.class_count and
                self.level == other.level and
                self.error == other.error and
                self.num_leaf == other.num_leaf and
                self.features == other.features)

    def predict(self, x):
        if self.is_leaf:
            return self.class_value
        else:
            i = 0
            for spl in self.splits:
                if x[self.var_index] == spl:
                    return self.childlist[i].predict(x)
                else:
                    i = i+1

    def to_rule_based_system(self, verbose=False):
        rules = []
        self._get_rules(self, rules, [], verbose)
        return [Rule(antecedent, consequent, 1) for (antecedent, consequent) in rules]

    def _get_rules(self, tree, rules, conditions, verbose=False):
        if not tree.is_leaf:
            for i in range(len(tree.splits)):
                cond = (tree.features[tree.var_index], tree.splits[i])
                self._get_rules(tree.childlist[i], rules, conditions + [cond], verbose)

        else:
            rules += [(conditions, tree.class_value)]
            if verbose:  # pragma: no cover
                print(f'{conditions} => Class value: {tree.class_value}, Counts: {tree.class_count}')


class ID3(BaseDecisionTree):
    def __init__(self, features, th=0.0001, max_depth=2, min_num_examples=1, prunning=True):
        super().__init__(features, th, max_depth, min_num_examples, prunning)

        self.tree_ = TreeID3(features)
        self.features_dic = {feature: i for i, feature in enumerate(features)}
        self.features_splits = None
        self.y_classes = None

    def fit(self, X, y, debug=False):
        self.features_splits = [np.unique(X[:, i]) for i in range(len(self.features))]
        self.y_classes = np.unique(y)
        self._partial_fit(X, y, self.tree_, 0, list(), debug)

    def _partial_fit(self, X, y, current_tree, current_depth, erased, debug):
        current_tree.level = current_depth
        num_cases = X.shape[0]

        # class entropy
        class_values = set(y)
        frecuency_list = []
        class_counts = []
        for i in class_values:
            actual_class_c = np.sum(y == i)
            class_counts.append([i, int(actual_class_c)])
            frecuency_list.append(actual_class_c/num_cases)
        class_ent = entropy(frecuency_list, base=2)

        # Before continuing, we check the depth
        class_counts = np.array(class_counts)
        if len(class_counts) == 0:
            current_tree.is_leaf = True
            current_tree.class_value = self.y_classes[0]
            current_tree.class_count = [0, 0]
            current_tree.error = 0
            current_tree.num_leaf = 1
            # print("--------------------------")
            # print("Error del nodo: ",error_node)
            # print(class_counts)
            # print("--------------------------")
            return

        counts = class_counts[:, 1].astype('int')
        current_tree.class_value = class_counts[np.argmax(counts), 0]
        current_tree.class_count = [num_cases, class_counts[np.argmax(counts), 1]]
        error_node = 1-(int(current_tree.class_count[1]) / int(current_tree.class_count[0]))

        if current_depth >= self.max_depth:
            current_tree.is_leaf = True
            current_tree.error = 1-(int(current_tree.class_count[1]) / int(current_tree.class_count[0]))
            current_tree.num_leaf = 1
            # print("--------------------------")
            # print("Error del nodo: ",error_node)
            # print(class_counts)
            # print("--------------------------")
            return

        if X.shape[0] < self.min_num_examples:
            current_tree.is_leaf = True
            current_tree.error = 1-(int(current_tree.class_count[1]) / int(current_tree.class_count[0]))
            current_tree.num_leaf = 1
            return

        # feature gains
        best_feature = -1
        best_splits = []
        best_gain = -1
        features_to_test = set(self.features)
        features_to_test.difference_update(set(erased))
        features_to_test = list(features_to_test)
        # print(features_to_test)
        # print(self.features_dic)
        for ft in features_to_test:
            # print(ft)
            # creamos una tabla auxiliar con la variable objetivo y la clase
            feature_index = self.features_dic[ft]
            # print(feature_index)
            feature_data = X[:, feature_index]
            # print(feature_data)
            splits = self.features_splits[feature_index]  # np.unique(feature_data)
            new_table = np.vstack((feature_data, y)).T
            actual_entropy = 0.0
            for spl in list(splits):
                split_classes = new_table[new_table[:, 0] == spl, 1]
                if split_classes.shape[0] == 0:
                    continue
                cumulative = []
                for j in class_values:
                    cumulative.append(np.sum(split_classes == j) / split_classes.shape[0])

                entropy_split = entropy(cumulative, base=2)

                actual_entropy += len(split_classes) * entropy_split / num_cases

            gain = class_ent - actual_entropy
            if gain > best_gain:
                best_feature = feature_index
                best_splits = list(splits)
                best_gain = gain

        # update the tree
        # print("*************************************************")
        # print(features[best_feature])
        # print(best_splits)
        # print(best_gain)
        # print("*************************************************")
        erased.append(self.features[best_feature])
        current_tree.is_leaf = False
        current_tree.error = 0
        current_tree.var_index = best_feature
        current_tree.splits = best_splits
        current_tree.childlist = []
        # print(current_tree)
        for sp in best_splits:
            current_tree.childlist.append(TreeID3(self.features))
            X_indexes = X[:, current_tree.var_index] == sp
            # print("*************************************************")
            # print(sp)
            # print(X_indexes)
            # print(X[X_indexes], y[X_indexes])
            # print("*************************************************")
            if len(X[X_indexes]) > 0:
                self._partial_fit(X[X_indexes], y[X_indexes], current_tree.childlist[-1],
                                  current_depth+1, erased.copy(), debug)
                current_tree.error += current_tree.childlist[-1].error*(len(y[X_indexes])/len(y))
            else:
                current_tree.childlist[-1].is_leaf = True
                current_tree.childlist[-1].class_value = current_tree.class_value
                current_tree.childlist[-1].class_count = [0, 0]
                current_tree.childlist[-1].error = 0
                current_tree.childlist[-1].num_leaf = 1
                current_tree.childlist[-1].level = current_depth+1
            # print("*************************************************")
            # print(current_tree.childlist[-1].error)
            # print((len(y[X_indexes])/len(y)))
            # print(current_tree.error) #R(current_tree)
            # print("*************************************************")
            current_tree.num_leaf += 1
        # print("*************************************************")
        # print("Subarbol: ",current_tree)
        # print("Error del sub_arbol",current_tree.error)
        # print("Numero de hojas ",current_tree.num_leaf)
        if self.prunning:
            th = ((error_node-current_tree.error)/(current_tree.num_leaf-1)) < self.th
            if th:
                current_tree.is_leaf = True
                current_tree.error = 1-(int(current_tree.class_count[1]) / int(current_tree.class_count[0]))
                current_tree.num_leaf = 1
                return
        # print("Corte ",(error_node-current_tree.error)/(current_tree.num_leaf-1))
        # print("*************************************************")
        return

    def predict(self, X):
        X = check_array(X, dtype=['float64', 'object'])
        return np.array([self.tree_.predict(x) for x in X])
