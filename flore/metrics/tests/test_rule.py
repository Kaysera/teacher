import numpy as np
from flore.metrics import coverage, precision, fidelity, rule_fidelity
from flore.tree import Rule


def test_coverage():
    fuzzy_set_df = {
        'feat1': {
            'val1': np.array([0.7, 1, 0.4]),
            'val2': np.array([0.3, 0, 0.6])
        },
        'feat2': {
            'val1': np.array([0.3, 0, 0.7]),
            'val2': np.array([0.7, 1, 0.3])
        },
        'feat3': {
            'val1': np.array([0.5, 0.9, 0.3]),
            'val2': np.array([0.5, 0.1, 0.7])
        }
    }

    rule = Rule((('feat1', 'val1'), ('feat2', 'val1'), ('feat3', 'val1')), 'conse', 0.5)
    np.testing.assert_almost_equal(coverage(rule, fuzzy_set_df), 0.6666666666)


def test_precision():
    fuzzy_set_df = {
        'feat1': {
            'val1': np.array([0.7, 1, 0.4]),
            'val2': np.array([0.3, 0, 0.6])
        },
        'feat2': {
            'val1': np.array([0.3, 0, 0.7]),
            'val2': np.array([0.7, 1, 0.3])
        },
        'feat3': {
            'val1': np.array([0.5, 0.9, 0.3]),
            'val2': np.array([0.5, 0.1, 0.7])
        }
    }

    y = np.array(['conse', 'conse', 'noconse'])

    rule = Rule((('feat1', 'val1'), ('feat2', 'val1'), ('feat3', 'val1')), 'conse', 0.5)
    np.testing.assert_almost_equal(precision(rule, fuzzy_set_df, y), 0.3333333333333)


def test_fidelity():

    y = np.array(['conse', 'conse', 'noconse'])
    y_local = np.array(['conse', 'noconse', 'noconse'])

    np.testing.assert_almost_equal(fidelity(y, y_local), 0.6666666666666666)


def test_rule_fidelity():
    fuzzy_set_df = {
        'feat1': {
            'val1': np.array([0.7, 1, 0.4]),
            'val2': np.array([0.3, 0, 0.6])
        },
        'feat2': {
            'val1': np.array([0.3, 0, 0.7]),
            'val2': np.array([0.7, 1, 0.3])
        },
        'feat3': {
            'val1': np.array([0.5, 0.9, 0.3]),
            'val2': np.array([0.5, 0.1, 0.7])
        }
    }

    y = np.array(['conse', 'conse', 'noconse'])
    y_local = np.array(['conse', 'noconse', 'conse'])

    rule = Rule((('feat1', 'val1'), ('feat2', 'val1'), ('feat3', 'val1')), 'conse', 0.5)
    np.testing.assert_almost_equal(rule_fidelity(y, y_local, fuzzy_set_df, rule), 0.5)
