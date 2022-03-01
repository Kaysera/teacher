import numpy as np


def _get_covered_instances(rule, fuzzy_dataset, threshold=0.001):
    first_feat = list(fuzzy_dataset.keys())[0]
    first_val = list(fuzzy_dataset[first_feat].keys())[0]
    ds_len = len(fuzzy_dataset[first_feat][first_val])
    mu = np.ones(ds_len)

    for feat, val in rule.antecedent:
        mu = np.minimum(mu, fuzzy_dataset[feat][val])
    return (mu > threshold, ds_len)


def _get_fuzzy_coverage(rules, fuzzy_dataset, threshold=0.001):
    covered_instances, ds_len = _get_covered_instances(rules[0], fuzzy_dataset, threshold)
    for rule in rules:
        n_covered_instance, _ = _get_covered_instances(rule, fuzzy_dataset, threshold)
        covered_instances = covered_instances | n_covered_instance
    return covered_instances, ds_len


def coverage(rules, fuzzy_dataset, threshold=0.001):
    covered_instances, ds_len = _get_fuzzy_coverage(rules, fuzzy_dataset, threshold)
    return np.sum(covered_instances) / ds_len


def precision(rules, fuzzy_dataset, y, threshold=0.001):
    covered_instances, ds_len = _get_fuzzy_coverage(rules, fuzzy_dataset, threshold)
    # All rules have the same consequent
    return np.sum((covered_instances) & (y == rules[0].consequent)) / np.sum(covered_instances)


def fidelity(y, y_local):
    return np.sum(y == y_local) / len(y)


def rule_fidelity(y, y_local, fuzzy_dataset, rules, threshold=0.001):
    covered_instances, ds_len = _get_fuzzy_coverage(rules, fuzzy_dataset, threshold)
    return np.sum(y[covered_instances] == y_local[covered_instances]) / np.sum(covered_instances)
