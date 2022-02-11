import random
from pytest import raises
import numpy as np
from pytest import fixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flore.explanation import FID3Explainer
from flore.datasets import load_compas
from flore.tree import Rule
from flore.neighbors import LoreNeighborhood, NotFittedError
from .._base_explainer import BaseExplainer
from .._factual_local_explainer import FactualLocalExplainer


@fixture
def set_random():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    return seed


class MockBaseExplainer(BaseExplainer):
    """Mock Base Explainer, not intended for use"""
    def fit(self):
        """Mock fit method that does nothing"""
        return True


class MockFactualLocalExplainer(FactualLocalExplainer):
    """Mock Base Explainer, not intended for use"""
    def fit(self):
        """Mock fit method that does nothing"""
        self.exp_value = 1
        self.target = 1


def test_explainer_not_fitted():
    with raises(NotFittedError):
        mbe = MockBaseExplainer()
        mbe.explain()


def test_explainer_hit_not_fitted():
    with raises(NotFittedError):
        mbe = MockFactualLocalExplainer()
        mbe.hit()


def test_explainer_hit():
    mbe = MockFactualLocalExplainer()
    mbe.fit()
    assert mbe.hit() is True


def test_FID3Explainer(set_random):
    dataset = load_compas()

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=set_random)

    idx_record2explain = 3
    instance = X_test[idx_record2explain]
    size = 300
    class_name = dataset['class_name']
    get_division = 'equal_freq'

    df_numerical_columns = [col for col in dataset['continuous'] if col != class_name]
    df_categorical_columns = [col for col in dataset['discrete'] if col != class_name]

    blackbox = RandomForestClassifier(n_estimators=20, random_state=set_random)
    blackbox.fit(X_train, y_train)
    target = blackbox.predict(instance.reshape(1, -1))

    neighborhood = LoreNeighborhood(instance, size, class_name, blackbox, dataset, X_test, idx_record2explain)
    neighborhood.fit()
    neighborhood.fuzzify(get_division,
                         sets=3,
                         class_name=class_name,
                         df_numerical_columns=df_numerical_columns,
                         df_categorical_columns=df_categorical_columns)

    explainer = FID3Explainer()
    explainer.fit(instance, target, neighborhood)

    factual, counterfactual = explainer.explain()

    expected_fact = Rule((('days_b_screening_arrest', '1.0'), ('priors_count', '1.0')), 1, 1)
    expected_cf = [Rule((('days_b_screening_arrest', '1.0'), ('priors_count', '1.0')), 1, 1)]

    assert factual == expected_fact
    assert counterfactual == expected_cf
