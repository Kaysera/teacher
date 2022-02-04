import numpy as np
import pandas as pd

from flore.neighbors import SimpleNeighborhood, BaseNeighborhood, FuzzyNeighborhood

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

    simple_neighborhood = SimpleNeighborhood(instance, size, class_name, blackbox)

    assert issubclass(SimpleNeighborhood, BaseNeighborhood)
    assert not issubclass(SimpleNeighborhood, FuzzyNeighborhood)

    simple_neighborhood.fit()
    neighborhood_X = simple_neighborhood.get_X()
    neighborhood_y = simple_neighborhood.get_y()

    expected_X = pd.DataFrame([instance] * size)
    expected_y = blackbox.predict(expected_X)

    pd.testing.assert_frame_equal(expected_X, neighborhood_X)
    np.testing.assert_equal(expected_y, neighborhood_y)
