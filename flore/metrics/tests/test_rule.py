import numpy as np
from flore.metrics import coverage
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
    assert coverage(rule, fuzzy_set_df) == 2
