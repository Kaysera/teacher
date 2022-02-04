from abc import ABC
from flore.fuzzy import get_fuzzy_points, get_fuzzy_set_dataframe, get_fuzzy_triangle
from flore.neighbors import BaseNeighborhood
from ._base_neighborhood import NotFittedException


class NotFuzzifiedException(Exception):
    pass


class FuzzyNeighborhood(BaseNeighborhood, ABC):

    def __init__(self, instance, size, class_name, bb):
        self._fuzzy_X = None
        super().__init__(instance, size, class_name, bb)

    def fuzzify(self, get_division, **kwargs):
        # EXPECTED PARAMS IN KWARGS: df_numerical_columns, df_categorical_columns, sets, fuzzy_labels, class_name,
        # verbose
        if self._X is None or self._y is None:
            raise NotFittedException

        fuzzy_points_params = ['df_numerical_columns', 'sets', 'class_name', 'verbose']
        fuzzy_points_args = {param: kwargs[param] for param in fuzzy_points_params if param in kwargs.keys()}

        fuzzy_set_params = ['df_numerical_columns', 'df_categorical_columns', 'labels', 'verbose']
        fuzzy_set_args = {param: kwargs[param] for param in fuzzy_set_params if param in kwargs.keys()}

        if (get_division == 'equal_width' or get_division == 'equal_freq') and 'sets' not in kwargs.keys():
            raise ValueError('Number of sets needed for division method specified')

        if get_division == 'entropy' and 'class_name' not in kwargs.keys():
            raise ValueError('Class Name needed for division method specified')

        if 'df_numerical_columns' not in kwargs.keys() or 'df_categorical_columns' not in kwargs.keys():
            raise ValueError('Numerical and categorical columns needed to get fuzzy X')

        fuzzy_points = get_fuzzy_points(self._X, get_division, **fuzzy_points_args)
        fuzzy_set = get_fuzzy_set_dataframe(self._X, get_fuzzy_triangle, fuzzy_points, **fuzzy_set_args)

        self._fuzzy_X = fuzzy_set

    def get_fuzzy_X(self):
        if self._fuzzy_X is None:
            raise NotFuzzifiedException
        else:
            return self._fuzzy_X
