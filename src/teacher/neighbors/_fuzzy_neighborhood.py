# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC

# Third party
import pandas as pd

# Local application
from teacher.fuzzy import get_fuzzy_variables, get_fuzzy_points, dataset_membership
from teacher.neighbors import BaseNeighborhood
from ._exceptions import NotFittedError, NotFuzzifiedError

# =============================================================================
# Classes
# =============================================================================


class FuzzyNeighborhood(BaseNeighborhood, ABC):
    """
    Base Fuzzy abstract neighborhood that provides the
    starting point for all fuzzy neighborhoods
    """

    def __init__(self, instance, size, class_name, bb):
        self._X_membership = None
        self._instance_membership = None
        self._fuzzy_variables = None
        super().__init__(instance, size, class_name, bb)

    def fuzzify(self, get_division, **kwargs):
        """
        Method to fuzzify a fitted neighborhood, obtaining the fuzzy partitions and
        obtaining the membership degrees of the different fuzzy sets for all the
        instances of the neighborhood.

        Parameters
        ----------
        get_division : str, {"equal_width", "equal_freq", "entropy"}
            Type of fuzzy discretization of the neighborhood

        Keyword Arguments
        -----------------
        df_numerical_columns : list
            List with all the numerical columns
        df_categorical_columns : list
            List with all the categorical columns
        sets : int
            Number of sets, needed for *get_division="equal_width"*
            or *get_division="equal_freq"*
        fuzzy_labels : list, of shape (n_sets)
            Optional list of names for the sets generated with the
            methods *get_division="equal_width"* or *get_division="equal_freq"*
        class_name : str
            Name of the class variable, needed for *get_division="entropy"*
        verbose : bool
            Debug flag

        Raises
        ------
        NotFittedError
            Neighborhood must be fitted before fuzzifying
        ValueError
            Argument mismatch
        """

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
        if 'th' in kwargs.keys():
            fuzzy_points_args['th'] = kwargs['th']
        X_num = self._X[kwargs['df_numerical_columns']]
        num_cols = X_num.columns
        fuzzy_points = get_fuzzy_points(get_division, num_cols, X_num, self._y, **fuzzy_points_args)

        discrete_fuzzy_values = {col: self._X[col].unique() for col in kwargs['df_categorical_columns']}
        fuzzy_variables_order = {col: i for i, col in enumerate(self._X.columns)}
        self._fuzzy_variables = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, fuzzy_variables_order)
        self._X_membership = dataset_membership(self._X, self._fuzzy_variables)

        if 'instance_membership' not in kwargs.keys() or kwargs['instance_membership']:
            instance_dict = {self._X.columns[i]: [self.instance[i]] for i in range(len(self.instance))}
            self._instance_membership = dataset_membership(pd.DataFrame(instance_dict), self._fuzzy_variables)

    def get_X_membership(self):
        """
        Return the membership degrees of the neighborhood

        Returns
        -------
        dict
            Dictionary of shape {feature: {set_1: [pert_1, pert_2, ...]}} with the
            pertenence degrees of the neighborhood extracted by :meth:`teacher.fuzzy.dataset_membership`

        Raises
        ------
        NotFuzzifiedError
            If the neighborhood is not fuzzified yet
        """
        if self._X_membership is None:
            raise NotFuzzifiedError
        else:
            return self._X_membership

    def get_instance_membership(self):
        """
        Return the membership degrees of the instance

        Returns
        -------
        dict
            Dictionary of shape {feature: {set_1: [pert_1, pert_2, ...]}} with the
            pertenence degrees of the instance extracted by :meth:`teacher.fuzzy.dataset_membership`

        Raises
        ------
        NotFuzzifiedError
            If the neighborhood is not fuzzified yet
        """
        if self._instance_membership is None:
            raise NotFuzzifiedError
        else:
            return self._instance_membership

    def get_fuzzy_variables(self):
        """
        Return the fuzzy variables used in the neighborhood

        Returns
        -------
        list[FuzzyVariable]
            List with all the fuzzy variables already generated

        Raises
        ------
        NotFuzzifiedError
            If the neighborhood is not fuzzified yet
        """
        if self._fuzzy_variables is None:
            raise NotFuzzifiedError
        else:
            return self._fuzzy_variables
