import numpy as np


def coverage(rule, fuzzy_dataset, threshold=0.001):
    first_feat = list(fuzzy_dataset.keys())[0]
    first_val = list(fuzzy_dataset[first_feat].keys())[0]
    ds_len = len(fuzzy_dataset[first_feat][first_val])
    mu = np.ones(ds_len)

    for feat, val in rule.antecedent:
        mu = np.minimum(mu, fuzzy_dataset[feat][val])
    return np.sum(mu > threshold)
