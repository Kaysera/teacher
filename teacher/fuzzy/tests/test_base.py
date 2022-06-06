import numpy as np
from numpy.testing import assert_equal
from pytest import raises, fixture
import pandas as pd

from teacher.fuzzy import (get_fuzzy_points, dataset_membership, get_fuzzy_variables,
                           FuzzyContinuousSet, FuzzyDiscreteSet, FuzzyVariable)

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
        'two': [7]
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


def test_get_fuzzy_points(toy_dataset):
    df, df_numerical_columns, sets = toy_dataset

    width_points = get_fuzzy_points('equal_width', df_numerical_columns, df[df_numerical_columns], sets=sets)
    freq_points = get_fuzzy_points('equal_freq', df_numerical_columns, df[df_numerical_columns], sets=sets)

    assert width_points == {'one': [0, 5, 10], 'two': [7, 8.5, 10]}
    assert freq_points == {'one': [0, 2, 10], 'two': [7, 9, 10]}


def test_get_fuzzy_points_point_set(toy_dataset):
    df = pd.DataFrame(
        [
            [0, 7, 'six'],
            [2, 7, 'nine'],
            [10, 7, 'ninety']
        ],
        columns=['one', 'two', 'three']
    )

    df_numerical_columns = ['one', 'two']
    sets = 3
    width_points = get_fuzzy_points('equal_width', df_numerical_columns, df[df_numerical_columns],
                                    sets=sets, point_variables=['two'])
    assert width_points == {'one': [0, 5.0, 10.0], 'two': np.array([7])}


def test_get_fuzzy_points_bad_method(toy_dataset):
    with raises(ValueError):
        df, df_numerical_columns, sets = toy_dataset
        get_fuzzy_points('None', df_numerical_columns, df[df_numerical_columns], sets=sets)


def test_get_fuzzy_points_unsupported_division(toy_dataset):
    df, df_numerical_columns, sets = toy_dataset
    with raises(ValueError):
        get_fuzzy_points(df, 'None', df_numerical_columns, sets=sets)


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


def test_get_fuzzy_variables_wrong_point_set_method(toy_fuzzy_variables):
    with raises(ValueError):
        _, _, continuous_labels, _, _ = toy_fuzzy_variables

        ordered_dict = {'one': 0}
        fuzzy_points = {'one': [1]}
        get_fuzzy_variables(fuzzy_points, {}, ordered_dict,
                            continuous_labels, {}, point_set_method='None')


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
        FuzzyVariable(name='two', fuzzy_sets=[FuzzyContinuousSet(name='7', fuzzy_points=[7, 7, 7], point_set=True)]),
        FuzzyVariable(name='three', fuzzy_sets=[FuzzyDiscreteSet(name='six', value='six'),
                                                FuzzyDiscreteSet(name='nine', value='nine'),
                                                FuzzyDiscreteSet(name='ninety', value='ninety')])
    ]

    expected_fuzzy_vars_no_labels = [
        FuzzyVariable(name='one', fuzzy_sets=[FuzzyContinuousSet(name='0', fuzzy_points=[0, 0, 5]),
                                              FuzzyContinuousSet(name='5', fuzzy_points=[0, 5, 10]),
                                              FuzzyContinuousSet(name='10', fuzzy_points=[5, 10, 10])]),
        FuzzyVariable(name='two', fuzzy_sets=[FuzzyContinuousSet(name='7', fuzzy_points=[7, 7, 7], point_set=True)]),
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
                                   'two': {'7': np.array([1., 0., 0.])},
                                   'three': {'six': np.array([1, 0, 0]),
                                             'nine': np.array([0, 1, 0]),
                                             'ninety': np.array([0, 0, 1])}}

    assert_equal(df_membership, expected_dataset_membership)
