# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
from sklearn.utils import check_array, check_X_y

# Local application
from .fuzzy_variable import FuzzyVariable
from .fuzzy_set import FuzzyContinuousSet, FuzzyDiscreteSet
from . import _discretize

# =============================================================================
# Constants
# =============================================================================

DISCRETIZE_METHODS = {
    'equal_freq': _discretize._equal_freq,
    'equal_width': _discretize._equal_width,
    'entropy': _discretize._fuzzy_discretization
}

# =============================================================================
# Functions
# =============================================================================


def get_fuzzy_points(discretize_method, df_numerical_columns, X, y=None, sets=0,
                     point_variables=None, max_depth=0, th=None, debug=False):
    """
    Obtain the peak of the fuzzy triangles of
    the continuous variables of a dataset.

    Parameters
    ----------
    discretize_method : str, {'equal_freq', 'equal_width', 'entropy'}
        Function used to get the divisions.
    df_numerical_columns : list
        Ordered numerical columns of the input samples.
    X : array-like, of shape (n_samples, n_features)
        Training input samples. Must only have numerical columns.
    y : array-like of shape (n_samples,)
        Target values (class labels) as integers or strings.
    sets : int
        Number of fuzzy sets that the variable will
        be divided into.
    point_variables : set, None by default
        Set of the variables to be considered point variables
        to return a list with the point value.
    debug : boolean, False by default
        Debugging flag
    Returns
    -------
    dict
        Dictionary with the format {feature_name : [start, peak, end]}
        for each feature in *df_numerical_columns*

    Raises
    -------
    ValueError
        Discretize method not supported
    """

    if y is not None:
        X, y = check_X_y(X, y)
    else:
        X = check_array(X)

    fuzzy_points = {}

    try:
        discretize = DISCRETIZE_METHODS[discretize_method]
    except KeyError:
        raise ValueError(f"Discretize method '{discretize_method}' not supported")

    for i, column in enumerate(df_numerical_columns):
        if point_variables and column in point_variables:
            fuzzy_points[column] = np.unique(X[:, i])
        elif discretize_method == 'entropy':
            fuzzy_points[column] = discretize(X[:, i],
                                              y,
                                              np.min(X[:, i]),
                                              depth=0,
                                              max_depth=max_depth,
                                              th=th,
                                              verbose=debug)
        else:
            fuzzy_points[column] = discretize(X[:, i], sets)

    return fuzzy_points


def dataset_membership(X, fuzzy_variables):
    """
    Compute the membership of the values of all the instances
    of a dataset to each Fuzzy Set of the different Fuzzy Variables.

    Parameters
    ----------
    X : array-like, of shape (n_samples, n_features)
        The input samples of which to obtain the membership.
    fuzzy_variables : list[FuzzyVariable]
        List of the fuzzy variables to compute the membership. Must be
        ordered according to the features of X.

    Returns
    -------
    dict
        Dictionary with format {variable: {set_1: pert_1, ...}, ...} with all
        the variables in *fuzzy_variables* and the pertenence degree to
        all the corresponding sets.
    """

    X = check_array(X, dtype=['float64', 'object'])

    dataset_membership = {}
    for i, fuzzy_var in enumerate(fuzzy_variables):
        dataset_membership[fuzzy_var.name] = fuzzy_var.membership(X[:, i])
    return dataset_membership


def get_fuzzy_variables(continuous_fuzzy_points, discrete_fuzzy_values, order,
                        continuous_labels=None, discrete_labels=None, point_set_method='point_set'):
    """Build the fuzzy variables given the points of the triangles that
    define them, as well as the values of the discrete variables.

    Parameters
    ----------
    continuous_fuzzy_points : dict
        Dictionary with format {feature: [peak_1, peak_2, ...]} with the name
        of the features and the peaks of the triangles of each fuzzy set.
    discrete_fuzzy_values : dict
        Dictionary with format {feature : [value_1, value_2, ...]} with the name
        of the features and the unique values thatthe discrete variable can take
    order : dict
        Dictionary with format {name : position} where each name is the label
        of the fuzzy variable and the position relative to an input dataset
    continuous_labels : dict, optional
        Dictionary with format {feature : [label_1, label_2, ...]} with the name
        the continuous variable and the labels of the fuzzy
        sets associated to the peaks peak_1, peak_2, ...
    discrete_labels : dict, optional
        Dictionary with format {feature : [label_1, label_2, ...]} with the name
        the discrete variable and the labels of the fuzzy
        sets associated to the values value_1, value_2, ...
    point_set : str, 'point_set' by default
        Method to generate the point sets. Defaults to `point_set`

    Returns
    -------
    list[FuzzyVariable]
        Ordered list of all the fuzzy variables
    """
    fuzzy_variables = [None] * len(order)
    for name, points in continuous_fuzzy_points.items():
        if continuous_labels is None or name not in continuous_labels:
            col_labels = [f'{label}' for label in continuous_fuzzy_points[name]]
        else:
            col_labels = continuous_labels[name]
        fuzzy_variables[order[name]] = FuzzyVariable(name, get_fuzzy_continuous_sets(list(zip(col_labels, points)),
                                                                                     point_set_method=point_set_method))

    for name, values in discrete_fuzzy_values.items():
        if discrete_labels is None or name not in discrete_labels:
            col_labels = [f'{label}' for label in discrete_fuzzy_values[name]]
        else:
            col_labels = discrete_labels[name]
        fuzzy_variables[order[name]] = FuzzyVariable(name, get_fuzzy_discrete_sets(list(zip(col_labels, values))))

    return fuzzy_variables


def _point_set(divisions):
    """Generate a FuzzyContinuousSet of a single point
    with all three values the same

    Parameters
    ----------
    divisions : tuple
        Tuple with the name of the set and the peak of the triangle
        like ('low', 0)
    """
    return [FuzzyContinuousSet(divisions[0], [divisions[1], divisions[1], divisions[1]], point_set=True)]


def get_fuzzy_continuous_sets(divisions, point_set_method='point_set'):
    """Generate a list with the triangular fuzzy sets of
    a variable of a DataFrame given the peaks of
    the triangles

    Parameters
    ----------
    divisions : list
        List of tuples with the names of the sets and the peak of the triangle
        like [('low', 0), ('mid', 2), ('high', 5)]
    point_set_method : str, 'point_set' by default
        Name of the method to generate the point sets.
        Defaults to `point_set`

    Returns
    -------
    list
        List with all the Fuzzy Continuous Sets that form a Fuzzy Variable
    """
    # WE FIRST CHECK IF IT IS A POINT VALUE
    if len(divisions) == 1:
        if point_set_method == 'point_set':
            return _point_set(divisions[0])
        else:
            raise ValueError(f'Point set method {point_set_method} is not valid')

    fuzzy_sets = []
    fuzzy_sets.append(FuzzyContinuousSet(divisions[0][0], [divisions[0][1], divisions[0][1], divisions[1][1]]))
    # First triangle is only half triangle

    for i in range(len(divisions) - 2):
        fuzzy_sets.append(FuzzyContinuousSet(divisions[i + 1][0], [divisions[i][1],
                                                                   divisions[i + 1][1], divisions[i + 2][1]]))

    # Last triangle is only half triangle
    fuzzy_sets.append(FuzzyContinuousSet(divisions[-1][0], [divisions[-2][1], divisions[-1][1], divisions[-1][1]]))

    return fuzzy_sets


def get_fuzzy_discrete_sets(divisions):
    """Generate a list with the discrete fuzzy sets of
    a variable of a DataFrame given the unique values
    it can take

    Parameters
    ----------
    divisions : list
        List of tuples with the names of the sets and the peak of the triangle
        like [('low', 0), ('mid', 2), ('high', 5)]

    Returns
    -------
    list
        List with all the Fuzzy Discrete Sets that form a Fuzzy Variable
    """

    return [FuzzyDiscreteSet(name, value) for name, value in divisions]
