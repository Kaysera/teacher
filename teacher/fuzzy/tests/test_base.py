import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from pytest import raises, fixture
import pandas as pd

from teacher.fuzzy import (get_equal_width_division, get_equal_freq_division, get_fuzzy_points,
                           get_fuzzy_triangle,
                           fuzzy_entropy, weighted_fuzzy_entropy, dataset_membership, get_fuzzy_variables,
                           FuzzyContinuousSet, FuzzyDiscreteSet, FuzzyVariable)

from .._base import _get_delta_point, _fuzzy_partitioning

from sklearn import datasets
from sklearn.model_selection import train_test_split


@fixture
def toy_dataset():
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
    return df, df_numerical_columns, sets


@fixture
def toy_fuzzy_variables(toy_dataset):
    df, df_numerical_columns, sets = toy_dataset
    fuzzy_points = {
        'one': [0, 5, 10],
        'two': [0, 5, 10]
    }
    continuous_labels = {
        'one': ['low', 'mid', 'high']
    }

    discrete_labels = {
        'three': ['six', 'nine', 'ninety']
    }

    df_categorical_columns = ['three']
    discrete_fuzzy_values = {col: df[col].unique() for col in df_categorical_columns}
    return df, fuzzy_points, continuous_labels, discrete_labels, discrete_fuzzy_values


def test_get_equal_width_division():
    array = np.array([0, 1, 1, 2, 2, 2, 3, 3, 6, 7, 9, 9, 10])
    parts_three = get_equal_width_division(array, 3)
    parts_five = get_equal_width_division(array, 5)
    assert parts_three == [0, 5, 10]
    assert parts_five == [0, 2.5, 5, 7.5, 10]


def test_valueerror_get_equal_width_division():
    with raises(ValueError):
        array = np.array([2, 2])
        get_equal_width_division(array, 3)


def test_get_equal_freq_division():
    array = np.array([0, 0, 0, 0, 0, 1, 1, 2, 5, 6, 8, 9, 10, 10, 10])
    parts_three = get_equal_freq_division(array, 3)
    parts_five = get_equal_freq_division(array, 5)
    assert parts_three == [0, 2, 10]
    assert parts_five == [0, 0, 2, 8.5, 10]


def test_valueerror_get_equal_freq_division():
    with raises(ValueError):
        array = np.array([2, 2])
        get_equal_freq_division(array, 3)


def test_get_fuzzy_points(toy_dataset):
    df, df_numerical_columns, sets = toy_dataset

    width_points = get_fuzzy_points('equal_width', df_numerical_columns, df[df_numerical_columns], sets=sets)
    freq_points = get_fuzzy_points('equal_freq', df_numerical_columns, df[df_numerical_columns], sets=sets)

    assert width_points == {'one': [0, 5, 10], 'two': [7, 8.5, 10]}
    assert freq_points == {'one': [0, 2, 10], 'two': [7, 9, 10]}


def test_get_fuzzy_points_unsupported_division(toy_dataset):
    df, df_numerical_columns, sets = toy_dataset
    with raises(ValueError):
        get_fuzzy_points(df, 'None', df_numerical_columns, sets=sets)


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


def test_fuzzy_entropy():
    variable = np.array([1.25, 3.75, 5])
    three_divisions = [('low', 0), ('mid', 5), ('high', 10)]
    three_triangles = get_fuzzy_triangle(variable, three_divisions)

    class_var = np.array([1, 0, 1])
    assert_almost_equal(fuzzy_entropy(three_triangles['low'], class_var, verbose=True), 0.8112781)


def test_weighted_fuzzy_entropy():
    variable = np.array([1.25, 3.75, 5])
    three_divisions = [('low', 0), ('mid', 5), ('high', 10)]
    three_triangles = get_fuzzy_triangle(variable, three_divisions)

    class_var = np.array([1, 0, 1])
    assert_almost_equal(weighted_fuzzy_entropy(three_triangles, class_var), 0.9067153767)


def test_get_delta_point():
    variable = np.array([1.25, 3.75, 5])
    three_divisions = [('low', 0), ('mid', 5), ('high', 10)]
    two_divisions = [('low', 0), ('high', 10)]
    three_triangles = get_fuzzy_triangle(variable, three_divisions)
    two_triangles = get_fuzzy_triangle(variable, two_divisions)

    class_var = np.array([1, 0, 1])

    assert_almost_equal(_get_delta_point(two_triangles, three_triangles, class_var, verbose=True), 2.6378347)


def test_fuzzy_partitioning():
    iris = datasets.load_iris(as_frame=True)
    class_name = 'target'

    X_train, _, _, _ = train_test_split(iris.data,
                                        iris.target,
                                        test_size=0.33,
                                        random_state=42)

    df_train = iris.frame.loc[X_train.index]
    column = 'petal length (cm)'
    fuzzy_points = _fuzzy_partitioning(df_train[column].to_numpy(), df_train[class_name].to_numpy(),
                                       df_train[column].min(), verbose=False)
    expected_fuzzy_points = [1.1, 1.9, 4.0, 5.0, 6.7]

    assert fuzzy_points == expected_fuzzy_points


def test_get_fuzzy_points_entropy():
    iris = datasets.load_iris(as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.33,
                                                        random_state=42)

    df_numerical_columns = iris.feature_names

    fuzzy_points_generic = get_fuzzy_points('entropy', df_numerical_columns, X_train, y_train)

    expected_fuzzy_points = {'sepal length (cm)': [4.3, 5.7, 7.7],
                             'sepal width (cm)': [2.0, 4.2],
                             'petal length (cm)': [1.1, 1.9, 4.0, 5.0, 6.7],
                             'petal width (cm)': [0.1, 0.6, 1.0, 1.7, 2.5]}

    assert fuzzy_points_generic == expected_fuzzy_points


def test_get_fuzzy_variables(toy_fuzzy_variables):
    _, fuzzy_points, continuous_labels, discrete_labels, discrete_fuzzy_values = toy_fuzzy_variables

    ordered_dict = {'one': 0, 'two': 1, 'three': 2}
    fuzzy_vars_labels = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, ordered_dict,
                                            continuous_labels, discrete_labels)
    fuzzy_vars_no_labels = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, ordered_dict)

    expected_fuzzy_vars_labels = [
        FuzzyVariable(name='one', fuzzy_sets=[FuzzyContinuousSet(name='low', fuzzy_points=[0, 0, 5]),
                                              FuzzyContinuousSet(name='mid', fuzzy_points=[0, 5, 10]),
                                              FuzzyContinuousSet(name='high', fuzzy_points=[5, 10, 10])]),
        FuzzyVariable(name='two', fuzzy_sets=[FuzzyContinuousSet(name='0', fuzzy_points=[0, 0, 5]),
                                              FuzzyContinuousSet(name='5', fuzzy_points=[0, 5, 10]),
                                              FuzzyContinuousSet(name='10', fuzzy_points=[5, 10, 10])]),
        FuzzyVariable(name='three', fuzzy_sets=[FuzzyDiscreteSet(name='six', value='six'),
                                                FuzzyDiscreteSet(name='nine', value='nine'),
                                                FuzzyDiscreteSet(name='ninety', value='ninety')])
    ]

    expected_fuzzy_vars_no_labels = [
        FuzzyVariable(name='one', fuzzy_sets=[FuzzyContinuousSet(name='0', fuzzy_points=[0, 0, 5]),
                                              FuzzyContinuousSet(name='5', fuzzy_points=[0, 5, 10]),
                                              FuzzyContinuousSet(name='10', fuzzy_points=[5, 10, 10])]),
        FuzzyVariable(name='two', fuzzy_sets=[FuzzyContinuousSet(name='0', fuzzy_points=[0, 0, 5]),
                                              FuzzyContinuousSet(name='5', fuzzy_points=[0, 5, 10]),
                                              FuzzyContinuousSet(name='10', fuzzy_points=[5, 10, 10])]),
        FuzzyVariable(name='three', fuzzy_sets=[FuzzyDiscreteSet(name='six', value='six'),
                                                FuzzyDiscreteSet(name='nine', value='nine'),
                                                FuzzyDiscreteSet(name='ninety', value='ninety')])
    ]

    assert fuzzy_vars_labels == expected_fuzzy_vars_labels
    assert fuzzy_vars_no_labels == expected_fuzzy_vars_no_labels


def test_get_dataset_membership(toy_fuzzy_variables):
    df, fuzzy_points, continuous_labels, _, discrete_fuzzy_values = toy_fuzzy_variables
    ordered_dict = {'one': 0, 'two': 1, 'three': 2}
    fuzzy_variables = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, ordered_dict, continuous_labels)
    df_membership = dataset_membership(df, fuzzy_variables)
    expected_dataset_membership = {'one': {'low': np.array([1, 0.6, 0]),
                                           'mid': np.array([0, 0.4, 0]),
                                           'high': np.array([0, 0., 1.])},
                                   'two': {'0': np.array([0, 0, 0]),
                                           '5': np.array([0.6, 0.2, 0]),
                                           '10': np.array([0.4, 0.8, 1])},
                                   'three': {'six': np.array([1, 0, 0]),
                                             'nine': np.array([0, 1, 0]),
                                             'ninety': np.array([0, 0, 1])}}

    assert_equal(df_membership, expected_dataset_membership)
