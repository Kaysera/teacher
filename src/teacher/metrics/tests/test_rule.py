import numpy as np
from teacher.metrics import coverage, precision, fidelity, rule_fidelity
from teacher.tree import Rule


def test_coverage():
    dataset_membership = {
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
    np.testing.assert_almost_equal(coverage([rule], dataset_membership), 0.6666666666)


def test_coverage_multiple_rules():
    dataset_membership = {
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

    r1 = Rule((('feat1', 'val1'), ('feat2', 'val1'), ('feat3', 'val1')), 'conse', 0.5)
    r2 = Rule((('feat1', 'val1'), ('feat2', 'val2'), ('feat3', 'val1')), 'conse', 0.5)
    np.testing.assert_almost_equal(coverage([r1, r2], dataset_membership), 0.6666666666)


def test_precision():
    dataset_membership = {
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
    np.testing.assert_almost_equal(precision([rule], dataset_membership, y), 0.5)


def test_fidelity():

    y = np.array(['conse', 'conse', 'noconse'])
    y_local = np.array(['conse', 'noconse', 'noconse'])

    np.testing.assert_almost_equal(fidelity(y, y_local), 0.6666666666666666)


def test_rule_fidelity():
    dataset_membership = {
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
    np.testing.assert_almost_equal(rule_fidelity(y, y_local, dataset_membership, [rule]), 0.5)
