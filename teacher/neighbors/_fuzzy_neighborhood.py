from abc import ABC
from teacher.fuzzy import get_fuzzy_variables, get_fuzzy_points, dataset_membership
from teacher.neighbors import BaseNeighborhood
from ._base_neighborhood import NotFittedError
import pandas as pd


class NotFuzzifiedError(Exception):
    pass


class FuzzyNeighborhood(BaseNeighborhood, ABC):

    def __init__(self, instance, size, class_name, bb):
        self._X_membership = None
        self._instance_membership = None
        self._fuzzy_variables = None
        super().__init__(instance, size, class_name, bb)

    def fuzzify(self, get_division, **kwargs):
        # EXPECTED PARAMS IN KWARGS: df_numerical_columns, df_categorical_columns, sets,
        #                            fuzzy_labels, class_name, verbose

        if self._X is None or self._y is None or self._Xy is None:
            raise NotFittedError

        fuzzy_points_params = ['sets', 'verbose']
        fuzzy_points_args = {param: kwargs[param] for param in fuzzy_points_params if param in kwargs.keys()}

        if (get_division == 'equal_width' or get_division == 'equal_freq') and 'sets' not in kwargs.keys():
            raise ValueError('Number of sets needed for division method specified')

        if 'df_numerical_columns' not in kwargs.keys() or 'df_categorical_columns' not in kwargs.keys():
            raise ValueError('Numerical and categorical columns needed to get fuzzy X')

        th = 0.00001  # THRESHOLD TO AVOID PARTITIONS WITH ONLY ONE VALUE

        # GET POINT VARIABLES
        point_vars = set([])
        for num in kwargs['df_numerical_columns']:
            if self._X[num].std() < th:
                point_vars.add(num)
        fuzzy_points_args['point_variables'] = point_vars

        X_num = self._X[kwargs['df_numerical_columns']]
        num_cols = X_num.columns
        fuzzy_points = get_fuzzy_points(get_division, num_cols, X_num, self._y, **fuzzy_points_args)

        discrete_fuzzy_values = {col: self._X[col].unique() for col in kwargs['df_categorical_columns']}
        fuzzy_variables_order = {col: i for i, col in enumerate(self._X.columns)}
        self._fuzzy_variables = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, fuzzy_variables_order)
        self._X_membership = dataset_membership(self._X, self._fuzzy_variables)

        instance_dict = {self._X.columns[i]: [self.instance[i]] for i in range(len(self.instance))}
        self._instance_membership = dataset_membership(pd.DataFrame(instance_dict), self._fuzzy_variables)

    def get_X_membership(self):
        if self._X_membership is None:
            raise NotFuzzifiedError
        else:
            return self._X_membership

    def get_instance_membership(self):
        if self._instance_membership is None:
            raise NotFuzzifiedError
        else:
            return self._instance_membership

    def get_fuzzy_variables(self):
        if self._fuzzy_variables is None:
            raise NotFuzzifiedError
        else:
            return self._fuzzy_variables
