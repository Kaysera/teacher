from abc import ABC, abstractmethod
from flore.fuzzy import get_fuzzy_points, get_fuzzy_set_dataframe, get_fuzzy_triangle


class NotFittedException(Exception):
    pass


class NotFuzzifiedException(Exception):
    pass


class BaseNeighborhood(ABC):

    def __init__(self, instance, size, class_name):
        self._data = None
        self._fuzzy_data = None
        self.instance = instance
        self.size = size
        self.class_name = class_name
        pass

    @abstractmethod
    def fit(self):
        pass

    def fuzzify(self, get_division, **kwargs):
        # EXPECTED PARAMS IN KWARGS: df_numerical_columns, df_categorical_columns, sets, fuzzy_labels, class_name,
        # verbose
        # sets must be equal to len(fuzzy_labels)
        if self._data is None:
            raise NotFittedException
        X = self._data.drop(self.class_name, axis=1)

        fuzzy_points_params = ['df_numerical_columns', 'sets', 'class_name', 'verbose']
        fuzzy_points_args = {param: kwargs[param] for param in fuzzy_points_params if param in kwargs.keys()}

        fuzzy_set_params = ['df_numerical_columns', 'df_categorical_columns', 'labels', 'verbose']
        fuzzy_set_args = {param: kwargs[param] for param in fuzzy_set_params if param in kwargs.keys()}
        fuzzy_points = get_fuzzy_points(X, get_division, **fuzzy_points_args)
        fuzzy_set = get_fuzzy_set_dataframe(X, get_fuzzy_triangle, fuzzy_points, **fuzzy_set_args)

        self._fuzzy_data = fuzzy_set

    def get_data(self):
        if self._data is None:
            raise NotFittedException
        else:
            return self._data

    def get_fuzzy_data(self):
        if self._fuzzy_data is None:
            raise NotFuzzifiedException
        else:
            return self._fuzzy_data
