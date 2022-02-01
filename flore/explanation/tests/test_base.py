from flore.explanation import FuzzyLORE
from flore.datasets import load_compas, load_german
from flore.neighbors import genetic_neighborhood
from flore.fuzzy import get_equal_freq_division
from math import prod

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import random
import numpy as np


def _sumthin():
    seed = 0

    random.seed(seed)
    np.random.seed(seed)

    dataset = load_german()

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    blackbox = RandomForestClassifier(n_estimators=20, random_state=seed)
    blackbox.fit(X_train, y_train)

    # X2E = X_test
    # y2E = blackbox.predict(X2E)
    # idx_record2explain = 1

    # explanation, lore_info = fit(idx_record2explain, X_test, dataset, blackbox,
    #                                 ng_function=genetic_neighborhood,
    #                                 discrete_use_probabilities=True,
    #                                 continuous_function_estimation=True)

    # X_testdf = lore_info['X_testdf']
    # fuzzy_set_test = lore_info['fuzzy_set_test']

    # assert explanation == ([('duration_in_month', 'low', 0.9999583350693723)], 0)


def _sumthinother():
    seed = 0

    random.seed(seed)
    np.random.seed(seed)

    dataset = load_compas()
    print('\n')

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    blackbox = RandomForestClassifier(n_estimators=20, random_state=seed)
    blackbox.fit(X_train, y_train)

    idx_record2explain = 3

    flore = FuzzyLORE()

    fuzzy_labels = ['low', 'mid']
    get_division = get_equal_freq_division
    operator = prod

    flore.fit(idx_record2explain, X_test, dataset, blackbox, fuzzy_labels, get_division, operator,
              ng_function=genetic_neighborhood,
              discrete_use_probabilities=True)

    explanation = flore.get_explanation(operator)

    fidelity = flore.fidelity()
    l_fidelity = flore.l_fidelity(explanation, 0.01)

    hit = flore.hit(explanation)

    precision = flore.precision(explanation)

    print(fidelity, l_fidelity, hit, flore.fhit, precision)
