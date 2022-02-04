import numpy as np
import pandas as pd

from flore.neighbors import (SimpleNeighborhood, BaseNeighborhood, FuzzyNeighborhood,
                             LoreNeighborhood, NotFittedException, NotFuzzifiedException)
from flore.datasets import load_beer

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import random
from pytest import raises, fixture


@fixture
def set_random():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    return seed


@fixture
def prepare_iris(set_random):
    iris = datasets.load_iris(as_frame=True)

    class_name = 'target'

    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.33,
                                                        random_state=set_random)

    blackbox = RandomForestClassifier(n_estimators=20, random_state=set_random)
    blackbox.fit(X_train, y_train)

    instance = X_train.loc[1]
    size = 3

    return [instance, size, class_name, blackbox]


@fixture
def prepare_beer(set_random):
    dataset = load_beer()

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=set_random)

    blackbox = RandomForestClassifier(n_estimators=20, random_state=set_random)
    blackbox.fit(X_train, y_train)

    idx_record2explain = 3
    instance = X_train[idx_record2explain]
    size = 20
    class_name = dataset['class_name']

    df_numerical_columns = [col for col in dataset['continuous'] if col != class_name]
    df_categorical_columns = [col for col in dataset['discrete'] if col != class_name]

    return [instance, size, class_name, blackbox, dataset, X_test,
            idx_record2explain, df_numerical_columns, df_categorical_columns]


def test_not_fitted_exception_x(prepare_iris):
    with raises(NotFittedException):
        neighborhood = SimpleNeighborhood(*prepare_iris)
        neighborhood.get_X()


def test_not_fitted_exception_y(prepare_iris):
    with raises(NotFittedException):
        neighborhood = SimpleNeighborhood(*prepare_iris)
        neighborhood.get_y()


def test_not_fitted_exception_fuzzify(prepare_beer):
    with raises(NotFittedException):
        [instance, size, class_name, blackbox, dataset, X_test,
            idx_record2explain, df_numerical_columns, df_categorical_columns] = prepare_beer

        neighborhood = LoreNeighborhood(instance, size, class_name, blackbox, dataset, X_test, idx_record2explain)
        neighborhood.fuzzify('equal_width',
                             sets=3,
                             class_name=class_name,
                             df_numerical_columns=df_numerical_columns,
                             df_categorical_columns=df_categorical_columns)


def test_equal_width_no_sets(prepare_beer):
    with raises(ValueError):
        (instance, size, class_name, blackbox, dataset, X_test,
         idx_record2explain, df_numerical_columns, df_categorical_columns) = prepare_beer

        neighborhood = LoreNeighborhood(instance, size, class_name, blackbox, dataset, X_test, idx_record2explain)
        neighborhood.fit()
        neighborhood.fuzzify('equal_width',
                             class_name=class_name,
                             df_numerical_columns=df_numerical_columns,
                             df_categorical_columns=df_categorical_columns)


def test_equal_freq_no_sets(prepare_beer):
    with raises(ValueError):
        (instance, size, class_name, blackbox, dataset, X_test,
         idx_record2explain, df_numerical_columns, df_categorical_columns) = prepare_beer

        neighborhood = LoreNeighborhood(instance, size, class_name, blackbox, dataset, X_test, idx_record2explain)
        neighborhood.fit()
        neighborhood.fuzzify('equal_freq',
                             class_name=class_name,
                             df_numerical_columns=df_numerical_columns,
                             df_categorical_columns=df_categorical_columns)


def test_entropy_no_class_name(prepare_beer):
    with raises(ValueError):
        (instance, size, class_name, blackbox, dataset, X_test,
         idx_record2explain, df_numerical_columns, df_categorical_columns) = prepare_beer

        neighborhood = LoreNeighborhood(instance, size, class_name, blackbox, dataset, X_test, idx_record2explain)
        neighborhood.fit()
        neighborhood.fuzzify('entropy',
                             sets=3,
                             df_numerical_columns=df_numerical_columns,
                             df_categorical_columns=df_categorical_columns)


def test_entropy_not_fuzzified(prepare_beer):
    with raises(NotFuzzifiedException):
        (instance, size, class_name, blackbox, dataset, X_test,
         idx_record2explain, df_numerical_columns, df_categorical_columns) = prepare_beer

        neighborhood = LoreNeighborhood(instance, size, class_name, blackbox, dataset, X_test, idx_record2explain)
        neighborhood.get_fuzzy_X()


def test_no_numerical_columns(prepare_beer):
    with raises(ValueError):
        (instance, size, class_name, blackbox, dataset, X_test,
         idx_record2explain, df_numerical_columns, df_categorical_columns) = prepare_beer

        neighborhood = LoreNeighborhood(instance, size, class_name, blackbox, dataset, X_test, idx_record2explain)
        neighborhood.fit()
        neighborhood.fuzzify('entropy',
                             class_name=class_name,
                             df_categorical_columns=df_categorical_columns)


def test_no_categorical_columns(prepare_beer):
    with raises(ValueError):
        (instance, size, class_name, blackbox, dataset, X_test,
         idx_record2explain, df_numerical_columns, df_categorical_columns) = prepare_beer

        neighborhood = LoreNeighborhood(instance, size, class_name, blackbox, dataset, X_test, idx_record2explain)
        neighborhood.fit()
        neighborhood.fuzzify('entropy',
                             class_name=class_name,
                             df_numerical_columns=df_numerical_columns)


def test_simple_neighborhood(prepare_iris):
    instance, size, class_name, blackbox = prepare_iris
    neighborhood = SimpleNeighborhood(instance, size, class_name, blackbox)

    assert issubclass(neighborhood.__class__, BaseNeighborhood)
    assert not issubclass(neighborhood.__class__, FuzzyNeighborhood)

    neighborhood.fit()
    neighborhood_X = neighborhood.get_X()
    neighborhood_y = neighborhood.get_y()

    expected_X = pd.DataFrame([instance] * size)
    expected_y = blackbox.predict(expected_X)

    pd.testing.assert_frame_equal(expected_X, neighborhood_X)
    np.testing.assert_equal(expected_y, neighborhood_y)


def test_lore_neighborhood(prepare_beer):
    (instance, size, class_name, blackbox, dataset, X_test,
     idx_record2explain, df_numerical_columns, df_categorical_columns) = prepare_beer

    neighborhood = LoreNeighborhood(instance, size, class_name, blackbox, dataset, X_test, idx_record2explain)

    assert issubclass(neighborhood.__class__, BaseNeighborhood)
    assert issubclass(neighborhood.__class__, FuzzyNeighborhood)

    neighborhood.fit()
    neighborhood_X = neighborhood.get_X()
    neighborhood_y = neighborhood.get_y()
    neighborhood_y_decoded = neighborhood.get_y_decoded()

    data = [
        [9.000000, 25.0, 0.081],
        [9.000000, 25.0, 0.081],
        [9.000000, 25.0, 0.053],
        [9.000000, 25.0, 0.107],
        [9.000000, 25.0, 0.081],
        [9.000000, 25.0, 0.053],
        [9.000000, 25.0, 0.053],
        [15.472054, 25.0, 0.053],
        [9.000000, 25.0, 0.081]
    ]

    cols = ['color', 'bitterness', 'strength']

    expected_X = pd.DataFrame(data=data, columns=cols)
    pd.testing.assert_frame_equal(expected_X, neighborhood_X)

    expected_y = [1, 1, 4, 1, 1, 4, 4, 4, 1]
    np.testing.assert_equal(expected_y, neighborhood_y)

    le = dataset['label_encoder'][class_name]
    expected_y_decoded = pd.Series(le.inverse_transform(neighborhood_y), name=class_name)
    pd.testing.assert_series_equal(expected_y_decoded, neighborhood_y_decoded)

    df_numerical_columns = [col for col in dataset['continuous'] if col != class_name]
    df_categorical_columns = [col for col in dataset['discrete'] if col != class_name]

    neighborhood.fuzzify('equal_width',
                         sets=3,
                         class_name=class_name,
                         df_numerical_columns=df_numerical_columns,
                         df_categorical_columns=df_categorical_columns)

    neighborhood_fuzzy_X = neighborhood.get_fuzzy_X()
    expected_fuzzy_X = {
        'color': {'9.0': np.array([1., 1., 1., 1., 1., 1., 1., 0., 1.]),
                  '12.236': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                  '15.472': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])},
        'bitterness': {'25.0': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]),
                       '25.025': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])},
        'strength': {'0.053': np.array([0., 0., 1., 0., 0., 1., 1., 1., 0.]),
                     '0.08': np.array([0.96296296, 0.96296296, 0., 0., 0.96296296, 0., 0., 0., 0.96296296]),
                     '0.107': np.array([0.03703704, 0.03703704, 0., 1., 0.03703704, 0., 0., 0., 0.03703704])}}

    for key in neighborhood_fuzzy_X.keys():
        var = neighborhood_fuzzy_X[key]
        for fuzzy_set in var.keys():
            np.testing.assert_almost_equal(neighborhood_fuzzy_X[key][fuzzy_set], expected_fuzzy_X[key][fuzzy_set])
