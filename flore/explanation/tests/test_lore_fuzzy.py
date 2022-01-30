from flore.explanation import FuzzyLORE_new
from flore.datasets import load_german

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import random
import numpy as np


def test_new_flore():
    seed = 0

    random.seed(seed)
    np.random.seed(seed)

    dataset = load_german()
    print('\n')

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    blackbox = RandomForestClassifier(n_estimators=20, random_state=seed)
    blackbox.fit(X_train, y_train)

    idx_record2explain = 2

    flore = FuzzyLORE_new()

    params = {}

    params['perturbation_prob'] = 0.4
    params['cxprob'] = 0.3
    params['crossprob'] = 0.3
    params['mutprob'] = 0.4
    params['mut_ext_prob'] = 0.5
    params['tournament_k'] = 3
    params['rounds'] = 10
    params['random_size'] = 250
    params['informed_size'] = 250
    params['outfile'] = f"{params['informed_size']}_{params['random_size']}_{params['rounds']}_{params['tournament_k']}"

    flore.fit(idx_record2explain, X_test, dataset, blackbox, params)
    print(flore.get_score())
    print(flore.hit())
    explanation = flore.explain(n_rules=1)
    print(explanation)
    print(flore.fhit(explanation[0]))
