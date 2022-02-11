import numpy as np


def coverage(rule, fuzzy_dataset, threshold=0.001):
    first_feat = list(fuzzy_dataset.keys())[0]
    first_val = list(fuzzy_dataset[first_feat].keys())[0]
    ds_len = len(fuzzy_dataset[first_feat][first_val])
    mu = np.ones(ds_len)

    for feat, val in rule.antecedent:
        mu = np.minimum(mu, fuzzy_dataset[feat][val])
    return np.sum(mu > threshold) / ds_len


def precision(rule, fuzzy_dataset, y, threshold=0.001):
    ds_len = len(y)
    mu = np.ones(ds_len)

    for feat, val in rule.antecedent:
        mu = np.minimum(mu, fuzzy_dataset[feat][val])

    return np.sum((mu > threshold) & (y == rule.consequent)) / ds_len


def fidelity(y, y_local):
    return np.sum(y == y_local) / len(y)


def rule_fidelity(y, y_local, fuzzy_dataset, rule, threshold=0.001):
    ds_len = len(y)
    mu = np.ones(ds_len)

    for feat, val in rule.antecedent:
        mu = np.minimum(mu, fuzzy_dataset[feat][val])

    return np.sum(y[mu > threshold] == y_local[mu > threshold]) / np.sum(mu > threshold)
