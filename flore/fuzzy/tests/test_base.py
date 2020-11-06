import numpy as np
from numpy.testing import assert_equal
import pandas as pd

from flore.fuzzy import (get_equal_width_division, get_equal_freq_division, get_fuzzy_points,
                         get_fuzzy_triangle, get_fuzzy_set_dataframe, get_fuzzy_points_entropy)


def test_get_equal_width_division():
    array = np.array([0, 1, 1, 2, 2, 2, 3, 3, 6, 7, 9, 9, 10])
    parts_three = get_equal_width_division(array, 3)
    parts_five = get_equal_width_division(array, 5)
    assert parts_three == [0, 5, 10]
    assert parts_five == [0, 2.5, 5, 7.5, 10]


def test_get_equal_freq_division():
    array = np.array([0, 0, 0, 0, 0, 1, 1, 2, 5, 6, 8, 9, 10, 10, 10])
    parts_three = get_equal_freq_division(array, 3)
    parts_five = get_equal_freq_division(array, 5)
    assert parts_three == [0, 2, 10]
    assert parts_five == [0, 0, 2, 8.5, 10]


def test_get_fuzzy_points():
    df = pd.DataFrame(
        [
            [0, 7, 'six'],
            [2, 9, 'nine'],
            [10, 10, 'ninety']
        ],
        columns=['one', 'two', 'three']
    )

    df_numerical_columns = ['one', 'two']
    sets = 3

    width_points = get_fuzzy_points(df, get_equal_width_division, df_numerical_columns, sets)
    freq_points = get_fuzzy_points(df, get_equal_freq_division, df_numerical_columns, sets)

    assert width_points == {'one': [0, 5, 10], 'two': [7, 8.5, 10]}
    assert freq_points == {'one': [0, 2, 10], 'two': [7, 9, 10]}


def test_get_fuzzy_triangle():
    variable_three = np.array([1.25, 5, 8.75])
    variable_five = np.array([1.25, 4.375, 8.125])

    three_divisions = [('low', 0), ('mid', 5), ('high', 10)]
    five_divisions = [('very low', 0), ('low', 2.5), ('mid', 5), ('high', 7.5), ('very high', 10)]

    three_triangles = get_fuzzy_triangle(variable_three, three_divisions)
    five_triangles = get_fuzzy_triangle(variable_five, five_divisions)

    assert_equal(three_triangles, {'low': np.array([0.75, 0., 0.]), 'mid': np.array([0.25, 1., 0.25]),
                                   'high': np.array([0., 0., 0.75])})
    assert_equal(five_triangles, {'very low': np.array([0.5, 0., 0.]), 'low': np.array([0.5, 0.25, 0.]),
                                  'mid': np.array([0., 0.75, 0.]), 'high': np.array([0., 0., 0.75]),
                                  'very high': np.array([0., 0., 0.25])})


def test_get_fuzzy_set_dataframe():
    df = pd.DataFrame(
        [
            [0, 1.25, 'six'],
            [2, 5, 'nine'],
            [10, 8.75, 'ninety']
        ],
        columns=['one', 'two', 'three']
    )
    fuzzy_points = {
        'one': [0, 5, 10],
        'two': [0, 5, 10]
    }
    fuzzy_labels = ['low', 'mid', 'high']
    df_numerical_columns = ['one', 'two']

    fuzzy_set_dataframe = get_fuzzy_set_dataframe(df, get_fuzzy_triangle, fuzzy_points,
                                                  df_numerical_columns, fuzzy_labels)
    assert_equal(fuzzy_set_dataframe, {'one': {'low': np.array([1., 0.6, 0.]),
                                               'mid': np.array([0., 0.4, 0.]),
                                               'high': np.array([0., 0., 1.])},
                                       'two': {'low': np.array([0.75, 0., 0.]),
                                               'mid': np.array([0.25, 1., 0.25]),
                                               'high': np.array([0., 0., 0.75])}})


def test_get_fuzzy_points_entropy():
    df = pd.DataFrame(
        [
            [1.85, 9.7, True],
            [5.74, 3.01, True],
            [3.13, 0.67, False],
            [1.91, 5.32, True],
            [7.7, 9.63, True],
            [7.24, 6.84, True],
            [3.08, 3.58, True],
            [7.75, 0.43, True],
            [0.9, 2.73, False],
            [8.4, 5.37, True],
            [0.81, 9.7, True],
            [9.85, 8.39, True],
            [9.64, 2.14, True],
            [5.84, 1.67, True],
            [3.94, 1.25, True],
            [9.87, 0.76, True],
            [1.26, 9.78, True],
            [9.34, 5.9, True],
            [1.53, 0.78, False],
            [0.94, 0.33, False],
            [5.09, 6.06, True],
            [4.91, 9.56, True],
            [9.09, 2.7, True],
            [1.88, 0.32, False],
            [4.98, 4.65, True],
            [2.15, 6.77, True],
            [4.58, 6.87, True],
            [6.52, 3.2, True],
            [7.33, 7.84, True],
            [5.94, 4.14, True],
            [1.55, 7.01, True],
            [8.69, 2.61, True],
            [1.25, 2.77, False],
            [9.85, 1.15, True],
            [9.96, 6.73, True],
            [1.91, 1.58, False],
            [9.15, 6.34, True],
            [2.64, 9.22, True],
            [6.46, 3.54, True],
            [1.77, 0.38, False]
        ],

        # [
        #     [0,3,True],
        #     [2.5,3,True],
        #     [5,3,False],
        #     [7.5,3,True],
        #     [10,3,False],
        # ],
        columns=['theory', 'practice', 'class']
    )

    df_numerical_columns = ['theory', 'practice']
    class_name = 'class'

    print(get_fuzzy_points_entropy(df, df_numerical_columns, class_name))


def test_get_fuzzy_points_entropy_two():
    theory = np.array([0, 0, 3, 3, 7, 7, 9])
    practice = np.array([0, 3, 3, 9, 1, 4, 9])
    df = pd.DataFrame(([i, j, i + j >= 10] for i, j in zip(theory, practice)), columns=['theory', 'practice', 'class'])
    df_numerical_columns = ['theory', 'practice']
    class_name = 'class'

    fuzzy_points = get_fuzzy_points_entropy(df, df_numerical_columns, class_name)
    print(fuzzy_points)


def test_fail():
    assert 1 == 2
