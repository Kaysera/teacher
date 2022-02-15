import numpy as np
from scipy.stats import entropy
from ..id3_tree import TreeID3


class ID3_Legacy:
    def __init__(self, features, X, y, max_depth=2, min_num_examples=1, prunning=True, th=0.0001):
        self.max_depth = max_depth
        self.tree = TreeID3(features)
        # lista de caracteristicas. tiene que ser de las mismas que X y
        # ademas ordenadas por el indice ambas. No puede estar la clase.
        self.features = features
        self.features_dic = {feature: i for i, feature in enumerate(features)}
        self.features_splits = [np.unique(X[:, i]) for i in range(len(self.features))]
        self.min_num_examples = min_num_examples
        self.prunning = prunning
        self.th = th
        self.y_classes = np.unique(y)

    def fit(self, X, y):
        self.partial_fit(X, y, self.tree, 0, list())

    def predict(self, X):
        return np.array([self.tree.predict(x) for x in X])

    # tambien se da una funcion score que calcula el accuracy
    def score(self, X, y):
        return np.sum(self.predict(X) == y)/y.shape[0]

    def partial_fit(self, X, y, actual_tree, actual_depth, borradas):
        actual_tree.level = actual_depth
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

        # antes de continuar, comprobamos la profundidad
        class_counts = np.array(class_counts)
        if len(class_counts) == 0:
            actual_tree.is_leaf = True
            actual_tree.class_value = self.y_classes[0]
            actual_tree.class_count = [0, 0]
            actual_tree.error = 0
            actual_tree.num_leaf = 1
            # print("--------------------------")
            # print("Error del nodo: ",error_node)
            # print(class_counts)
            # print("--------------------------")
            return

        counts = class_counts[:, 1].astype('int')
        actual_tree.class_value = class_counts[np.argmax(counts), 0]
        actual_tree.class_count = [num_cases, class_counts[np.argmax(counts), 1]]
        error_node = 1-(int(actual_tree.class_count[1]) / int(actual_tree.class_count[0]))

        if actual_depth >= self.max_depth:
            actual_tree.is_leaf = True
            actual_tree.error = 1-(int(actual_tree.class_count[1]) / int(actual_tree.class_count[0]))
            actual_tree.num_leaf = 1
            # print("--------------------------")
            # print("Error del nodo: ",error_node)
            # print(class_counts)
            # print("--------------------------")
            return

        if X.shape[0] < self.min_num_examples:
            actual_tree.is_leaf = True
            actual_tree.error = 1-(int(actual_tree.class_count[1]) / int(actual_tree.class_count[0]))
            actual_tree.num_leaf = 1
            return

        # feature gains
        best_feature = -1
        best_splits = []
        best_gain = -1
        features_to_test = set(self.features)
        features_to_test.difference_update(set(borradas))
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
        borradas.append(self.features[best_feature])
        actual_tree.is_leaf = False
        actual_tree.error = 0
        actual_tree.var_index = best_feature
        actual_tree.splits = best_splits
        actual_tree.childlist = []
        # print(actual_tree)
        for sp in best_splits:
            actual_tree.childlist.append(TreeID3(self.features))
            X_indexes = X[:, actual_tree.var_index] == sp
            # print("*************************************************")
            # print(sp)
            # print(X_indexes)
            # print(X[X_indexes], y[X_indexes])
            # print("*************************************************")
            if len(X[X_indexes]) > 0:
                self.partial_fit(X[X_indexes], y[X_indexes], actual_tree.childlist[-1], actual_depth+1, borradas.copy())
                actual_tree.error += actual_tree.childlist[-1].error*(len(y[X_indexes])/len(y))
            else:
                actual_tree.childlist[-1].is_leaf = True
                actual_tree.childlist[-1].class_value = actual_tree.class_value
                actual_tree.childlist[-1].class_count = [0, 0]
                actual_tree.childlist[-1].error = 0
                actual_tree.childlist[-1].num_leaf = 1
                actual_tree.childlist[-1].level = actual_depth+1
            # print("*************************************************")
            # print(actual_tree.childlist[-1].error)
            # print((len(y[X_indexes])/len(y)))
            # print(actual_tree.error) #R(actual_tree)
            # print("*************************************************")
            actual_tree.num_leaf += 1
        # print("*************************************************")
        # print("Subarbol: ",actual_tree)
        # print("Error del sub_arbol",actual_tree.error)
        # print("Numero de hojas ",actual_tree.num_leaf)
        if self.prunning:
            th = ((error_node-actual_tree.error)/(actual_tree.num_leaf-1)) < self.th
            if th:
                actual_tree.is_leaf = True
                actual_tree.error = 1-(int(actual_tree.class_count[1]) / int(actual_tree.class_count[0]))
                actual_tree.num_leaf = 1
                return
        # print("Corte ",(error_node-actual_tree.error)/(actual_tree.num_leaf-1))
        # print("*************************************************")
        return

    def activateRules(self, tree, rules, conditions, instance, instance_idx,
                      fuzzy_set, df_categorical_columns, threshold, verbose):
        if not tree.is_leaf:
            for i in range(len(tree.splits)):
                feat = tree.features[tree.var_index]
                branch = tree.splits[i]
                if feat in df_categorical_columns and instance[feat] == branch:
                    self.activateRules(tree.childlist[i], rules, conditions + [(feat, branch, 1)], instance,
                                       instance_idx, fuzzy_set, df_categorical_columns, threshold, verbose)
                elif feat not in df_categorical_columns:
                    pert = fuzzy_set[feat][branch][instance_idx]
                    if pert > threshold:
                        self.activateRules(tree.childlist[i], rules, conditions + [(feat, branch, pert)], instance,
                                           instance_idx, fuzzy_set, df_categorical_columns, threshold, verbose)
        else:
            if verbose:
                print(f'{conditions} => Class value: {tree.class_value}, Counts: {tree.class_count}')
            rules += [(conditions, tree.class_value)]

    def explainInstance(self, instance, instance_idx, fuzzy_set, df_categorical_columns, threshold=0.01, verbose=False):
        rules = []
        self.activateRules(self.tree, rules, [], instance, instance_idx, fuzzy_set,
                           df_categorical_columns, threshold, verbose)
        return rules

    def exploreTree(self, tree, rules, conditions, verbose=True):
        if not tree.is_leaf:
            for i in range(len(tree.splits)):
                cond = (tree.features[tree.var_index], tree.splits[i])
                self.exploreTree(tree.childlist[i], rules, conditions + [cond], verbose)

        else:
            rules += [(conditions, tree.class_value)]
            if verbose:
                print(f'{conditions} => Class value: {tree.class_value}, Counts: {tree.class_count}')

    def exploreTreeFn(self, verbose=True):
        rules = []
        self.exploreTree(self.tree, rules, [], verbose)
        return rules
