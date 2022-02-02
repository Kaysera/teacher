from flore.neighbors import get_feature_values, genetic_neighborhood, calculate_feature_values
from flore.tree import FDT
from flore.utils import dataframe2explain
from flore.fuzzy import get_fuzzy_points_entropy, get_fuzzy_triangle


class FuzzyLORE_new:
    def __init__(self):
        self.class_name = None
        self.discrete = None
        self.Z = None
        self.neighborhood = None
        self.X = None
        self.continuous = None
        self.score = None
        self.bb_outcome = None
        self.prediction = None
        self.fuzzy_instance = None
        self.tree = None
        self.class_le = None
        self.bb_outcome_decoded = None

    def fit(self, idx_record2explain, X2E, dataset, blackbox, ng_params=[]):

        class_name = dataset['class_name']
        self.class_name = class_name
        discrete = dataset['discrete']
        self.discrete = discrete
        continuous = dataset['continuous']
        self.continuous = continuous
        self.class_le = dataset['label_encoder'][class_name]

        dfZ, x = dataframe2explain(X2E, dataset, idx_record2explain, blackbox)
        instance = dfZ.loc[idx_record2explain]
        self.X2E = dfZ.drop(class_name, axis=1)
        self.y2E = dfZ[class_name]
        self.target = instance[class_name]
        self.instance = instance
        n_instance = instance.to_frame().transpose()
        dataset['feat_values'] = get_feature_values(dfZ.drop(class_name, axis=1), discrete)

        self.bb_outcome = blackbox.predict(x.reshape(1, -1))[0]
        self.bb_outcome_decoded = self.class_le.inverse_transform([self.bb_outcome])

        # TODO: Make statistics optional

        # df, Z, mean_gd, std_gd, unique_gd = genetic_neighborhood_flore(dfZ, x, blackbox, dataset, ng_params)
        # self.Z = Z
        # Dataset Preprocessing
        # TODO: REPLACE LORE GENETIC FOR OWN GENETIC
        columns = dataset['columns']
        dataset['feature_values'] = calculate_feature_values(X2E, columns, class_name, discrete, continuous, 1000,
                                                             False, False)

        df, Z = genetic_neighborhood(dfZ, x, blackbox, dataset)
        self.Z = Z

        self.neighborhood = df

        # Build Decision Tree
        X = df.drop(class_name, axis=1)
        self.X = X
        y = df[class_name]

        fuzzy_points = get_fuzzy_points_entropy(df, continuous, class_name)

        # TODO: THIS IS GET_FUZZY_SET_DATAFRAME
        fuzzy_set = {}
        for column in continuous:
            labels = [f'{label}' for label in fuzzy_points[column]]
            fuzzy_set[column] = get_fuzzy_triangle(X[column].to_numpy(),
                                                   list(zip(labels, fuzzy_points[column])),
                                                   False)

        for column in discrete:
            if column is not class_name:
                element = {}
                for value in X[column].unique():
                    element[value] = (X[column] == value).to_numpy().astype(int)
                fuzzy_set[column] = element

        # TODO: PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
        fdt = FDT(fuzzy_set.keys(), fuzzy_set)
        fdt.fit(X, y)
        # print(fdt.tree)
        self.tree = fdt
        self.score = fdt.score(fuzzy_set, y)

        fuzzy_instance = {}
        for column in continuous:
            labels = [f'{label}' for label in fuzzy_points[column]]
            fuzzy_instance[column] = get_fuzzy_triangle(n_instance[column].to_numpy(),
                                                        list(zip(labels, fuzzy_points[column])),
                                                        False)

        for column in discrete:
            if column is not class_name:
                element = {}
                for value in X[column].unique():
                    element[value] = (n_instance[column] == value).to_numpy().astype(int)
                fuzzy_instance[column] = element
        # print(n_instance)
        self.fuzzy_instance = fuzzy_instance
        # print(fuzzy_instance)
        self.prediction = fdt.predict(fuzzy_instance)

    def get_score(self):
        return self.score

    def get_prediction(self):
        return self.prediction

    def hit(self):
        return self.prediction == self.bb_outcome_decoded

    def fhit(self, explanation):
        # TODO: EXPAND WITH MULTIPLE RULES
        leaf = explanation[0]
        return min(leaf[self.bb_outcome_decoded[0]], explanation[1])

    def explain(self, n_rules='all'):
        return self.tree.explain(self.fuzzy_instance, self.target, n_rules)
