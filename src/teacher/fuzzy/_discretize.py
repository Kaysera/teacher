# =============================================================================
# Imports
# =============================================================================

# Standard
from math import log2, inf, pow

# Third party
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# =============================================================================
# Functions
# =============================================================================


def _equal_width(variable, sets):
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
    try:
        cut = pd.cut(variable, sets - 1)
        sol = [cat.left for cat in cut.categories] + [cut.categories[-1].right]
    except ValueError:  # pragma: no cover
        cut = pd.cut(variable, sets - 1, duplicates='drop')
        sol = [variable.min()] + [cat.left for cat in cut.categories] + [cut.categories[-1].right]
    sol[0] = variable.min()
    return sol


def _equal_freq(variable, sets):
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
    try:
        qcut = pd.qcut(variable, sets - 1)
        sol = [cat.left for cat in qcut.categories] + [qcut.categories[-1].right]
        sol[0] = variable.min()
    except ValueError:
        qcut = pd.qcut(variable, sets - 1, duplicates='drop')
        sol = [variable.min()] + [cat.left for cat in qcut.categories] + [qcut.categories[-1].right]
        sol[1] = sol[0]
    return sol


def _fuzzy_discretization(variable, class_variable, min_point, depth=0, max_depth=0, th=None, verbose=False):
    max_point = variable.max()
    best_point = 0
    best_wfe = inf
    best_fuzzy_triangle = []
    for point in np.unique(variable):
        if point != min_point and point != max_point:
            divisions = [('low', min_point), ('mid', point), ('high', max_point)]
            fuzzy_triangle = _fuzzy_triangle(variable, divisions)
            wfe = _weighted_fuzzy_entropy(fuzzy_triangle, class_variable)

            if verbose:  # pragma: no cover
                print('\t----------------')
                print(f'\t{divisions}')
                print('\t----------------')
                print(f'\tPoint: {point}')
                print(f'\twfe: {wfe}')
                print('\t-----------------')

            if wfe < best_wfe:
                best_wfe = wfe
                best_point = point
                best_fuzzy_triangle = fuzzy_triangle

    divisions = [('low', min_point), ('high', max_point)]
    global_fuzzy_triangles = _fuzzy_triangle(variable, divisions)
    global_wfe = _weighted_fuzzy_entropy(global_fuzzy_triangles, class_variable)
    if verbose:  # pragma: no cover
        print(f'Best point: {best_point}')
        print(f'Best Weighted Fuzzy Entropy: {best_wfe}')
        print(f'Global Weighted Fuzzy Entropy: {global_wfe}')

    f_gain = global_wfe - best_wfe

    cardinality = len(variable)
    delta = _get_delta_point(global_fuzzy_triangles, best_fuzzy_triangle, class_variable)
    if th is None:
        threshold = (log2(cardinality - 1) + delta) / cardinality
    else:
        threshold = th

    if verbose:   # pragma: no cover
        print('-----------------')
        print(f'Pass Threshold: {f_gain >= threshold}')
        print('-----------------')

    if (max_depth > 0 and max_depth > depth) or max_depth == 0:
        if not f_gain < threshold:
            left = ([(p, c) for p, c in zip(variable, class_variable) if p <= best_point])
            right = ([(p, c) for p, c in zip(variable, class_variable) if p > best_point])

            left_variable, left_class = zip(*left)
            right_variable, right_class = zip(*right)

            left_points = []
            right_points = []

            if len(left_variable) > 1:
                left_points = _fuzzy_discretization(np.array(left_variable), left_class, min_point, depth+1, max_depth, th, verbose)
            if len(right_variable) > 1:
                right_points = _fuzzy_discretization(np.array(right_variable), right_class, best_point, depth+1, max_depth, th, verbose)
            points = left_points + right_points
            return np.unique(points).tolist()
        else:
            return [min_point, max_point]
    else:
        return [min_point, max_point]


def _fuzzy_entropy(triangle, class_variable, verbose=False):
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
        class_fuzzy_cardinality = np.sum(triangle[class_variable == value])
        if class_fuzzy_cardinality > 0:  # i.e. There are elements belonging to this class value
            fuzzy_cardinality = triangle.sum()
            if verbose:   # pragma: no cover
                print(f'class_fuzzy_cardinality: {class_fuzzy_cardinality}')
                print(f'fuzzy_cardinality: {fuzzy_cardinality}')
            ratio = class_fuzzy_cardinality / fuzzy_cardinality
            fe += -ratio * log2(ratio)

    return fe


def _fuzzy_triangle(variable, divisions, verbose=False):
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


def _weighted_fuzzy_entropy(fuzzy_triangle, class_variable):
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
    wfe = 0
    crisp_cardinality = len(class_variable)  # Number of elements in the partition
    for triangle in fuzzy_triangle:
        fuzzy_cardinality = fuzzy_triangle[triangle].sum()
        fent = _fuzzy_entropy(fuzzy_triangle[triangle], class_variable)
        wfe += fuzzy_cardinality * fent
    return wfe / crisp_cardinality


def _get_delta_point(global_fuzzy_triangles, best_fuzzy_triangle, class_variable, verbose=False):
    """Given a set of instances partitioned in two fuzzy sets (global_fuzzy_triangles)
    and a partitioning of three fuzzy sets of the same instances (best_fuzzy_triangle),
    identifies if it is necessary to do that partitioning

    Parameters
    ----------
    global_fuzzy_triangles : dict
        Dictionary with the format {key : value} obtained from _fuzzy_triangle
        with two fuzzy partitions
    best_fuzzy_triangle : dict
        Dictionary with the format {key : value} obtained from _fuzzy_triangle
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

    old_f_entropy = n_classes * _weighted_fuzzy_entropy(global_fuzzy_triangles, class_variable)
    new_f_entropy = 0

    for triangle in best_fuzzy_triangle:
        bft = np.array(best_fuzzy_triangle[triangle])
        cv = np.array(class_variable)
        classes = cv[bft > 0]
        new_n_classes = len(np.unique(classes))
        new_f_entropy += new_n_classes * _fuzzy_entropy(best_fuzzy_triangle[triangle], class_variable)
    if verbose:   # pragma: no cover
        print(f'Old Entropy: {old_f_entropy:.3f}')
        print(f'New Entropy: {new_f_entropy:.3f}')

    delta = log2(pow(3, n_classes) - 2) - (old_f_entropy - new_f_entropy)

    return delta
