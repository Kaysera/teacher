import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from math import log2, inf, pow


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
    try:
        cut = pd.cut(variable, sets-1)
        sol = [cat.left for cat in cut.categories] + [cut.categories[-1].right]
    except ValueError:
        cut = pd.cut(variable, sets-1, duplicates='drop')
        sol = [variable.min()] + [cat.left for cat in cut.categories] + [cut.categories[-1].right]
    sol[0] = variable.min()
    return sol

def get_equal_freq_division(variable, sets):
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
        qcut = pd.qcut(variable, sets-1)
        sol = [cat.left for cat in qcut.categories] + [qcut.categories[-1].right]
        sol[0] = variable.min()
    except ValueError:
        qcut = pd.qcut(variable, sets-1, duplicates='drop')
        sol = [variable.min()] + [cat.left for cat in qcut.categories] + [qcut.categories[-1].right]
        sol[1] = sol[0]
    return sol

def get_fuzzy_points(df, get_divisions, df_numerical_columns, sets):
    """Function that obtains the peak of the fuzzy triangles of
    the continuous variables of a DataFrame

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame from which to obtain the fuzzy points
    get_divisions : function
        Function used to get the divisions. Currently 
        supported: equal freq and equal width
    df_numerical_columns : list
        List with the columns to get the fuzzy points
    sets : int
        Number of fuzzy sets that the variable will
        be divided into

    Returns
    -------
    dict
        Dictionary with the format {key : [points]}
    """
    fuzzy_points = {}
    for column in df_numerical_columns:
        fuzzy_points[column] = get_divisions(df[column].to_numpy(), sets)
    return fuzzy_points

def get_fuzzy_points_entropy(df, df_numerical_columns, class_name):
    """Function that obtains the peak of the fuzzy triangles of
    the continuous variables of a DataFrame according to the 
    Fuzzy Partitioning

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame from which to obtain the fuzzy points
    df_numerical_columns : list
        List with the columns to get the fuzzy points
    class_name : str
        Name of the class variable

    Returns
    -------
    dict
        Dictionary with the format {key : [points]}
    """
    fuzzy_points = {}
    for column in df_numerical_columns:
        fuzzy_points[column] = fuzzy_partitioning(df[column].to_numpy(), df[class_name].to_numpy())
    return fuzzy_points

def fuzzy_partitioning(variable, class_variable):
    min_point = variable.min()
    max_point = variable.max()
    best_point = 0
    best_wef = inf
    best_fuzzy_triangle = []
    for point in variable:
        if point != min_point and point != max_point:
            divisions = [('low', min_point), ('mid', point), ('high', max_point)]
            fuzzy_triangle = get_fuzzy_triangle(variable, divisions)
            # print(divisions)
            # print(fuzzy_triangle)
            wef = weighted_fuzzy_entropy(fuzzy_triangle, class_variable)
            # print(point, wef)
            if wef < best_wef:
                best_wef = wef
                best_point = point
                best_fuzzy_triangle = fuzzy_triangle
    
    left = ([(p,c) for p,c in zip(variable, class_variable) if p <= best_point])
    right = ([(p,c) for p,c in zip(variable, class_variable) if p > best_point])
    divisions = [('low', min_point), ('high', max_point)]
    global_fuzzy_triangles =  get_fuzzy_triangle(variable, divisions)

    global_wef = weighted_fuzzy_entropy(global_fuzzy_triangles, class_variable)

    f_gain = global_wef - best_wef

    cardinality = len(variable)

    delta = get_delta_point(global_fuzzy_triangles, best_fuzzy_triangle, class_variable)
    
    threshold = (log2(cardinality - 1) + delta) / cardinality

    if not f_gain < threshold:
        left_variable, left_class = zip(*left)
        right_variable, right_class = zip(*right)
        left_points = fuzzy_partitioning(np.array(left_variable), left_class)
        right_points = fuzzy_partitioning(np.array(right_variable), right_class)
        points = left_points + right_points
        return np.unique(points).tolist()
    
    else:
        return [min_point, max_point]




def get_delta_point(global_fuzzy_triangles, best_fuzzy_triangle, class_variable):
    n_classes = len(np.unique(class_variable))

    old_f_entropy = 0
    new_f_entropy = 0
    for triangle in global_fuzzy_triangles:
        old_f_entropy += fuzzy_entropy(global_fuzzy_triangles[triangle], class_variable)
    
    for triangle in best_fuzzy_triangle:
        new_f_entropy += fuzzy_entropy(best_fuzzy_triangle[triangle], class_variable)
    
    old_f_entropy *= n_classes

    new_f_entropy *= n_classes # TODO: REVISE THIS, MAYBE NOT ALL N_CLASSES

    delta = log2(pow(3, n_classes) - 2) - old_f_entropy - new_f_entropy

    return delta




def weighted_fuzzy_entropy(fuzzy_triangle, class_variable):
    wef = 0
    for triangle in fuzzy_triangle:
        fuzzy_cardinality = fuzzy_triangle[triangle].sum()
        crisp_cardinality = (fuzzy_triangle[triangle] > 0).sum()
        if triangle == 'low' or triangle == 'high':
            # Only has one extreme, i.e. [0,2.5]
            crisp_cardinality += 1
        else: 
            # Has two extremes, i.e. [0,2.5,5]
            crisp_cardinality += 2
        # print(f'Set: {triangle}')
        # print(f'fuzzy_cardinality: {fuzzy_cardinality}')
        # print(f'crisp_cardinality: {crisp_cardinality}')

        wef += fuzzy_cardinality / crisp_cardinality * fuzzy_entropy(fuzzy_triangle[triangle], class_variable)
    return wef

def fuzzy_entropy(triangle, class_variable, verbose=False):
    fe = 0
    for value in np.unique(class_variable):
        class_fuzzy_cardinality = 0
        for i in range(len(triangle)):
            if class_variable[i] == value:
                class_fuzzy_cardinality += triangle[i]
        
        if class_fuzzy_cardinality > 0: # i.e. There are elements belonging to this class value
            if verbose:
                print(triangle)
                print(class_variable)
            fuzzy_cardinality = triangle.sum()
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
        fuzz_dict[divisions[i+1][0]] = fuzz.trimf(variable, [divisions[i][1], divisions[i+1][1], divisions[i+2][1]])

    # Last triangle is only half triangle 
    fuzz_dict[divisions[-1][0]] = fuzz.trimf(variable, [divisions[-2][1], divisions[-1][1], divisions[-1][1]])

    if verbose:
        fig, ax0 = plt.subplots(nrows=1)
        for div in divisions:
            ax0.plot(variable, fuzz_dict[div[0]], linewidth=1.5, label=div[0])
        ax0.set_title('Fuzzy sets')
        ax0.legend()

        plt.tight_layout()

    return fuzz_dict 


def get_fuzzy_set_dataframe(df, gen_fuzzy_set, fuzzy_points, df_numerical_columns, labels, verbose=False):
    """Get all the fuzzy sets from the columns of a DataFrame, and the pertenence value of
    each register to each fuzzy set

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame to process
    gen_fuzzy_set : function
        Function used to get the fuzzy sets and their degree of pertenence. Currently supported
        get_fuzzy_triangle
    fuzzy_points : list
        List with the peaks of the triangles (Trapezium not supported)
    df_numerical_columns : list
        List with the columns of the DataFrame to fuzzify
    labels : list
        List with the names of the fuzzy sets
    verbose : bool, optional
        Enables verbosity and passes it down, by default False

    Returns
    -------
    dict
        Dictionary with format {key : value} where the key is the name of the column of the DataFrame
        and the value is the output of the gen_fuzzy_set function for that column
    """
    fuzzy_set = {}
    for column in df_numerical_columns:
        fuzzy_set[column] = gen_fuzzy_set(df[column].to_numpy(), list(zip(labels,fuzzy_points[column])), verbose)
    return fuzzy_set

def get_fuzzy_set_instance(df, gen_fuzzy_set, fuzzy_points, df_numerical_columns, labels, verbose=False):
    """Get all the fuzzy sets from the columns of a DataFrame, and the pertenence value of
    each register to each fuzzy set

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame to process
    gen_fuzzy_set : function
        Function used to get the fuzzy sets and their degree of pertenence. Currently supported
        get_fuzzy_triangle
    fuzzy_points : list
        List with the peaks of the triangles (Trapezium not supported)
    df_numerical_columns : list
        List with the columns of the DataFrame to fuzzify
    labels : list
        List with the names of the fuzzy sets
    verbose : bool, optional
        Enables verbosity and passes it down, by default False

    Returns
    -------
    dict
        Dictionary with format {key : value} where the key is the name of the column of the DataFrame
        and the value is the output of the gen_fuzzy_set function for that column
    """
    fuzzy_set = {}
    for column in df_numerical_columns:
        fuzzy_set[column] = gen_fuzzy_set(df[column], list(zip(labels,fuzzy_points[column])), verbose)
    return fuzzy_set