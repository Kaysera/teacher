import numpy as np
import pandas as pd

from flore.neighbors import SimpleNeighborhood, BaseNeighborhood, FuzzyNeighborhood, LoreNeighborhood
from flore.datasets import load_beer

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import random


def test_simple_neighborhood():
    iris = datasets.load_iris(as_frame=True)
    seed = 42

    random.seed(seed)
    np.random.seed(seed)

    class_name = 'target'

    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.33,
                                                        random_state=seed)

    blackbox = RandomForestClassifier(n_estimators=20, random_state=seed)
    blackbox.fit(X_train, y_train)

    instance = X_train.loc[1]
    size = 3

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


def test_lore_neighborhood():
    seed = 0

    random.seed(seed)
    np.random.seed(seed)

    dataset = load_beer()
    print('\n')

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    blackbox = RandomForestClassifier(n_estimators=20, random_state=seed)
    blackbox.fit(X_train, y_train)

    idx_record2explain = 3
    instance = X_train[idx_record2explain]
    size = 20
    class_name = dataset['class_name']

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
