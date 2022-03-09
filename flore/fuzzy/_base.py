import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from math import log2, inf, pow
from .fuzzy_variable import FuzzyVariable
from .fuzzy_set import FuzzyContinuousSet, FuzzyDiscreteSet


def get_equal_width_division(variable, sets):
    """Generate partitions of equal width from a variable

    Parameters
    ----------
    variable : numpy.ndarray
        Variable from the dataset to partition
    sets : int
        Number of partitions that will be generated

    Returns
    -------
    array
        Points of division between the partitions
    """
    th = 0.00001  # THRESHOLD TO AVOID PARTITIONS WITH ONLY ONE VALUE
    if np.std(variable) < th:
        raise ValueError(f"There is only one unique value in {variable} and cannot be partitioned")
    try:
        cut = pd.cut(variable, sets - 1)
        sol = [cat.left for cat in cut.categories] + [cut.categories[-1].right]
    except ValueError:  # pragma: no cover
        cut = pd.cut(variable, sets - 1, duplicates='drop')
        sol = [variable.min()] + [cat.left for cat in cut.categories] + [cut.categories[-1].right]
    sol[0] = variable.min()
    return sol


def get_equal_freq_division(variable, sets):
    """Generate partitions of equal frequency from a variable

    Parameters
    ----------
    variable : numpy.ndarray
        Variable from the dataset to partition
    sets : int
        Number of partitions that will be generated

    Returns
    -------
    array
        Points of division between the partitions
    """
    th = 0.00001  # THRESHOLD TO AVOID PARTITIONS WITH ONLY ONE VALUE
    if np.std(variable) < th:
        raise ValueError(f"There is only one unique value in {variable} and cannot be partitioned")
    try:
        qcut = pd.qcut(variable, sets - 1)
        sol = [cat.left for cat in qcut.categories] + [qcut.categories[-1].right]
        sol[0] = variable.min()
    except ValueError:
        qcut = pd.qcut(variable, sets - 1, duplicates='drop')
        sol = [variable.min()] + [cat.left for cat in qcut.categories] + [qcut.categories[-1].right]
        sol[1] = sol[0]
    return sol


def get_fuzzy_points(df, get_divisions, df_numerical_columns, sets=0,
                     class_name=None, point_variables=None, verbose=False):
    """Obtain the peak of the fuzzy triangles of
    the continuous variables of a DataFrame

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame from which to obtain the fuzzy points
    get_divisions : string
        Function used to get the divisions. Currently
        supported: 'equal_freq', 'equal_width', 'entropy'
    df_numerical_columns : list
        List with the columns to get the fuzzy points
    sets : int
        Number of fuzzy sets that the variable will
        be divided into
    class_name : str, None by default
        Name of the class variable necessary for 'entropy'
        division
    point_variables : set, None by default
        Set of the variables to be considered point variables
        to return a list with the point value

    Returns
    -------
    dict
        Dictionary with the format {key : [points]}
    """
    fuzzy_points = {}
    for column in df_numerical_columns:
        if point_variables and column in point_variables:
            fuzzy_points[column] = df[column].unique()
        elif get_divisions == 'equal_freq':
            fuzzy_points[column] = get_equal_freq_division(df[column].to_numpy(), sets)
        elif get_divisions == 'equal_width':
            fuzzy_points[column] = get_equal_width_division(df[column].to_numpy(), sets)
        elif get_divisions == 'entropy':
            fuzzy_points[column] = _fuzzy_partitioning(df[column].to_numpy(), df[class_name].to_numpy(),
                                                       df[column].min(), verbose)
        else:
            raise ValueError('Division method not supported')
    return fuzzy_points


def _fuzzy_partitioning(variable, class_variable, min_point, verbose=False):
    max_point = variable.max()
    best_point = 0
    best_wef = inf
    best_fuzzy_triangle = []
    for point in np.unique(variable):
        if point != min_point and point != max_point:
            divisions = [('low', min_point), ('mid', point), ('high', max_point)]
            fuzzy_triangle = get_fuzzy_triangle(variable, divisions)
            wef = weighted_fuzzy_entropy(fuzzy_triangle, class_variable)

            if verbose:  # pragma: no cover
                print('\t----------------')
                print(f'\t{divisions}')
                print('\t----------------')
                print(f'\tPoint: {point}')
                print(f'\tWEF: {wef}')
                print('\t-----------------')

            if wef < best_wef:
                best_wef = wef
                best_point = point
                best_fuzzy_triangle = fuzzy_triangle

    divisions = [('low', min_point), ('high', max_point)]
    global_fuzzy_triangles = get_fuzzy_triangle(variable, divisions)
    global_wef = weighted_fuzzy_entropy(global_fuzzy_triangles, class_variable)
    if verbose:  # pragma: no cover
        print(f'Best point: {best_point}')
        print(f'Best wef: {best_wef}')
        print(f'Global Weighted Fuzzy Entropy: {global_wef}')

    f_gain = global_wef - best_wef

    cardinality = len(variable)
    delta = _get_delta_point(global_fuzzy_triangles, best_fuzzy_triangle, class_variable)
    threshold = (log2(cardinality - 1) + delta) / cardinality

    if verbose:   # pragma: no cover
        print('-----------------')
        print(f'Pass Threshold: {f_gain >= threshold}')
        print('-----------------')

    if not f_gain < threshold:
        left = ([(p, c) for p, c in zip(variable, class_variable) if p <= best_point])
        right = ([(p, c) for p, c in zip(variable, class_variable) if p > best_point])

        left_variable, left_class = zip(*left)
        right_variable, right_class = zip(*right)

        left_points = []
        right_points = []

        if len(left_variable) > 1:
            left_points = _fuzzy_partitioning(np.array(left_variable), left_class, min_point, verbose)
        if len(right_variable) > 1:
            right_points = _fuzzy_partitioning(np.array(right_variable), right_class, best_point, verbose)
        points = left_points + right_points
        return np.unique(points).tolist()
    else:
        return [min_point, max_point]


def _get_delta_point(global_fuzzy_triangles, best_fuzzy_triangle, class_variable, verbose=False):
    """Given a set of instances partitioned in two fuzzy sets (global_fuzzy_triangles)
    and a partitioning of three fuzzy sets of the same instances (best_fuzzy_triangle),
    identifies if it is necessary to do that partitioning

    Parameters
    ----------
    global_fuzzy_triangles : dict
        Dictionary with the format {key : value} obtained from get_fuzzy_triangle
        with two fuzzy partitions
    best_fuzzy_triangle : dict
        Dictionary with the format {key : value} obtained from get_fuzzy_triangle
        with three fuzzy partitions
    class_variable : np.array
        Numpy array with the class values for the instances of the fuzzy partition
    verbose : bool, optional
        Verbose flag, by default False

    Returns
    -------
    Float
        Delta point
    """
    n_classes = len(np.unique(class_variable))

    old_f_entropy = 0
    new_f_entropy = 0
    for triangle in global_fuzzy_triangles:
        old_f_entropy += n_classes * fuzzy_entropy(global_fuzzy_triangles[triangle], class_variable)

    for triangle in best_fuzzy_triangle:
        bft = np.array(best_fuzzy_triangle[triangle])
        cv = np.array(class_variable)
        classes = cv[bft > 0]
        new_n_classes = len(np.unique(classes))
        new_f_entropy += new_n_classes * fuzzy_entropy(best_fuzzy_triangle[triangle], class_variable)
    if verbose:   # pragma: no cover
        print(f'Old Entropy: {old_f_entropy:.3f}')
        print(f'New Entropy: {new_f_entropy:.3f}')

    delta = log2(pow(3, n_classes) - 2) - (old_f_entropy - new_f_entropy)

    return delta


def weighted_fuzzy_entropy(fuzzy_triangle, class_variable):
    """Function to compute the weighted fuzzy entropy of
    a given fuzzy partition

    Parameters
    ----------
    fuzzy_triangle : dict
        Dictionary with the fuzzy sets and an array with the pertenence degree
        of each instance of the partition to that fuzzy set
    class_variable : np.array
        Numpy array with the class value of each instance of the partition

    Returns
    -------
    float
        Weighted fuzzy entropy
    """
    wef = 0
    crisp_cardinality = len(class_variable)  # Number of elements in the partition
    for triangle in fuzzy_triangle:
        fuzzy_cardinality = fuzzy_triangle[triangle].sum()
        fent = fuzzy_entropy(fuzzy_triangle[triangle], class_variable)
        wef += fuzzy_cardinality * fent
    return wef / crisp_cardinality


def fuzzy_entropy(triangle, class_variable, verbose=False):
    """Compute the fuzzy entropy of a given fuzzy subset

    Parameters
    ----------
    triangle : np.array
        Numpy array with the degree of pertenence for a particular
        fuzzy variable for each instance
    class_variable : np.array
        Numpy array with the values of the class for each instance
    verbose : bool, optional
        Verbose flag, by default False

    Returns
    -------
    float
        Fuzzy entropy of the subset
    """
    fe = 0
    for value in np.unique(class_variable):
        class_fuzzy_cardinality = 0
        for i in range(len(triangle)):
            if class_variable[i] == value:
                class_fuzzy_cardinality += triangle[i]

        if class_fuzzy_cardinality > 0:  # i.e. There are elements belonging to this class value
            fuzzy_cardinality = triangle.sum()
            if verbose:   # pragma: no cover
                print(f'class_fuzzy_cardinality: {class_fuzzy_cardinality}')
                print(f'fuzzy_cardinality: {fuzzy_cardinality}')
            ratio = class_fuzzy_cardinality / fuzzy_cardinality
            fe += -ratio * log2(ratio)

    return fe


def get_fuzzy_triangle(variable, divisions, verbose=False):
    """Function that generates a dictionary with the pertenence to each
    triangular fuzzy set of a variable of a DataFrame given the peaks of
    the triangles

    Parameters
    ----------
    variable : numpy.ndarray
        Variable from the dataset to create the fuzzy set
    divisions : list
        List of tuples with the names of the sets and the peak of the triangle
        like [('low', 0), ('mid', 2), ('high', 5)]
    verbose : bool, optional
        Enables verbosity by displaying the fuzzy sets, by default False

    Returns
    -------
    dict
        Dictionary with the format {key : value} where the key is the name of the set
        and the value is an array of the pertenence to the set of each value of the
        variable
    """
    fuzz_dict = {}
    # First triangle is only half triangle
    fuzz_dict[divisions[0][0]] = fuzz.trimf(variable, [divisions[0][1], divisions[0][1], divisions[1][1]])

    for i in range(len(divisions) - 2):
        fuzz_dict[divisions[i + 1][0]] = fuzz.trimf(variable,
                                                    [divisions[i][1], divisions[i + 1][1], divisions[i + 2][1]])

    # Last triangle is only half triangle
    fuzz_dict[divisions[-1][0]] = fuzz.trimf(variable, [divisions[-2][1], divisions[-1][1], divisions[-1][1]])

    if verbose:  # pragma: no cover
        fig, ax0 = plt.subplots(nrows=1)
        for div in divisions:
            ax0.plot(variable, fuzz_dict[div[0]], linewidth=1.5, label=div[0])
        ax0.set_title('Fuzzy sets')
        ax0.legend()

        plt.tight_layout()

    return fuzz_dict


def get_dataset_membership(df, fuzzy_variables, verbose=False):
    """Obtain the membership of the values of all the columns of a DataFrame to each
    Fuzzy Set of the different Fuzzy Variables

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame to process
    fuzzy_variables : list[FuzzySet]
        List of the fuzzy variables to compute the membership
    verbose : bool, optional
        Enables verbosity and passes it down, by default False

    Returns
    -------
    dict
        Dictionary with format {key : value} where the key is the name of the column of the DataFrame
        and the value is a dictionary with the membership to each fuzzy set of the variable
    """
    dataset_membership = {}
    for fuzzy_var in fuzzy_variables:
        dataset_membership[fuzzy_var.name] = fuzzy_var.membership(df[fuzzy_var.name].to_numpy())
    return dataset_membership


def get_fuzzy_variables(continuous_fuzzy_points, discrete_fuzzy_values, continuous_labels=None, discrete_labels=None):
    """Build the fuzzy variables given the points of the triangles that
    define them, as well as the values of the discrete variables

    Parameters
    ----------
    continuous_fuzzy_points : dict
        Dictionary with format {key : [v1, v2, ...]} where key is the
        name of the continuous variable and v1, v2, ... are the peaks of the triangles
        of each fuzzy set
    discrete_fuzzy_values : dict
        Dictionary with format {key : [v1, v2, ...]} where key is the
        name of the discrete variable and v1, v2, ... are the unique values that
        the discrete variable can take
    continuous_labels : dict, optional
        Dictionary with format {key : [l1, l2, ...]} where key is the
        name of the continuous variable and l1, l2, ... are the labels of the fuzzy
        sets associated to the peaks v1, v2, ...
    discrete_labels : dict, optional
        Dictionary with format {key : [l1, l2, ...]} where key is the
        name of the discrete variable and l1, l2, ... are the labels of the fuzzy
        sets associated to the values v1, v2, ...

    Returns
    -------
    list[FuzzyVariable]
        List of all the fuzzy variables
    """
    fuzzy_variables = []
    for name, points in continuous_fuzzy_points.items():
        if continuous_labels is None or name not in continuous_labels:
            col_labels = [f'{label}' for label in continuous_fuzzy_points[name]]
        else:
            col_labels = continuous_labels[name]
        fuzzy_variables.append(FuzzyVariable(name, get_fuzzy_continuous_sets(list(zip(col_labels, points)))))

    for name, values in discrete_fuzzy_values.items():
        if discrete_labels is None or name not in discrete_labels:
            col_labels = [f'{label}' for label in discrete_fuzzy_values[name]]
        else:
            col_labels = discrete_labels[name]
        fuzzy_variables.append(FuzzyVariable(name, get_fuzzy_discrete_sets(list(zip(col_labels, values)))))

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
