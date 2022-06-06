import random
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from teacher.datasets import load_compas
from teacher.tree import Rule
from teacher.neighbors import LoreNeighborhood


@pytest.fixture
def mock_fired_rules():

    instance = {
        'color': {
            'low': 0.5,
            'high': 0.5
        },
        'bitterness': {
            'very high': 0.8,
            'extremely high': 0.2
        },
        'strength': {
            'high': 0.47,
            'very high': 0.53
        }
    }

    class_val = 'IPA'

    rule_one = Rule([('color', 'low'), ('strength', 'high')], 'IPA', 0.93)
    rule_two = Rule([('color', 'high'), ('strength', 'high')], 'IPA', 0.65)
    rule_three = Rule([('color', 'low'), ('strength', 'very high')], 'IPA', 0.44)
    rule_four = Rule([('color', 'high'), ('strength', 'very high')], 'IPA', 0.12)
    rule_five = Rule([('color', 'high'), ('strength', 'very high')], 'Barleywine', 0.88)
    rule_six = Rule([('color', 'low'), ('strength', 'very high')], 'Barleywine', 0.28)
    rule_seven = Rule([('color', 'high'), ('strength', 'high')], 'Barleywine', 0.35)
    rule_eight = Rule([('color', 'low'), ('strength', 'high')], 'Barleywine', 0.07)

    rule_list = [rule_one, rule_two, rule_three, rule_four, rule_five, rule_six, rule_seven, rule_eight]

    return instance, rule_list, class_val


@pytest.fixture
def prepare_dummy_no_cf():
    instance = {
        'feat1': {
            'low': 0.7,
            'mid': 0.3,
            'high': 0
        },
        'feat2': {
            'low': 0,
            'mid': 0.2,
            'high': 0.8
        },
        'feat3': {
            'r': 0,
            'g': 1,
            'b': 0
        }
    }

    rule_list = [
        Rule([('feat1', 'low'), ('feat2', 'high'), ('feat3', 'g')], 1, 1),
        Rule([('feat1', 'mid'), ('feat2', 'mid'), ('feat3', 'g')], 1, 0.7),
        Rule([('feat1', 'mid'), ('feat2', 'mid'), ('feat3', 'g')], 0, 0.3),
    ]

    class_val = 1

    df_numerical_columns = ['feat1', 'feat2']

    return [instance, rule_list, class_val, df_numerical_columns]


@pytest.fixture
def prepare_dummy():
    instance = {
        'feat1': {
            'low': 0.7,
            'mid': 0.3,
            'high': 0
        },
        'feat2': {
            'low': 0,
            'mid': 0.2,
            'high': 0.8
        },
        'feat3': {
            'r': 0,
            'g': 1,
            'b': 0
        }
    }

    rule_list = [
        Rule([('feat1', 'low'), ('feat2', 'high'), ('feat3', 'g')], 1, 1),
        Rule([('feat1', 'low'), ('feat2', 'mid'), ('feat3', 'g')], 0, 1),
        Rule([('feat1', 'high'), ('feat2', 'high'), ('feat3', 'r')], 0, 1),
        Rule([('feat1', 'mid'), ('feat2', 'mid'), ('feat3', 'b')], 0, 1),
    ]

    class_val = 1

    df_numerical_columns = ['feat1', 'feat2']

    return [instance, rule_list, class_val, df_numerical_columns]


@pytest.fixture
def set_random():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def prepare_compas(set_random):
    dataset = load_compas()

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=set_random)

    idx_record2explain = 3
    instance = X_test[idx_record2explain]
    size = 300
    class_name = dataset['class_name']
    get_division = 'entropy'

    df_numerical_columns = [col for col in dataset['continuous'] if col != class_name]
    df_categorical_columns = [col for col in dataset['discrete'] if col != class_name]

    blackbox = RandomForestClassifier(n_estimators=20, random_state=set_random)
    blackbox.fit(X_train, y_train)
    target = blackbox.predict(instance.reshape(1, -1))

    neighborhood = LoreNeighborhood(instance, size, class_name, blackbox, dataset, X_test, idx_record2explain)
    neighborhood.fit()
    neighborhood.fuzzify(get_division,
                         class_name=class_name,
                         df_numerical_columns=df_numerical_columns,
                         df_categorical_columns=df_categorical_columns)
    instance = instance.reshape(1, -1)
    return [instance, target, neighborhood, df_numerical_columns]
