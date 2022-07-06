# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np

# =============================================================================
# Functions
# =============================================================================


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
        covered_instances = covered_instances & n_covered_instance
    return covered_instances, ds_len


def coverage(rules, fuzzy_dataset, threshold=0.001):
    """
    Compute the fuzzy coverage of a dataset given a set of rules
    so that if a rule matches an instance above a threshold it
    is considered covered.

    Parameters
    ----------
    rules : list[Rule]
        List of rules to cover the dataset
    fuzzy_dataset : dict
        Dataset membership with the format {feature_1: {set_1: [memb_1, memb_2, ...]}}
        with each feature and set and an array of shape (n_instances) with t
        he membership degree of all the instances of the dataset
    threshold : float, optional
        Activation threshold which sets when an instance is considered covered, by default 0.001

    Returns
    -------
    float
        Ratio of number of covered instances divided by dataset length
    """
    covered_instances, ds_len = _get_fuzzy_coverage(rules, fuzzy_dataset, threshold)
    return np.sum(covered_instances) / ds_len


def precision(rules, fuzzy_dataset, y, threshold=0.001):
    """
    Compute the precision of a dataset covered by set of rules
    to check if the class value matches the rule consequent.

    Parameters
    ----------
    rules : list[Rule]
        List of rules to cover the dataset
    fuzzy_dataset : dict
        Dataset membership with the format {feature_1: {set_1: [memb_1, memb_2, ...]}}
        with each feature and set and an array of shape (n_instances) with t
        he membership degree of all the instances of the dataset
    y : array-like, of shape (n_instances)
        Class values corresponding to each instance of the dataset
    threshold : float, optional
        Activation threshold which sets when an instance is considered covered, by default 0.001

    Returns
    -------
    float
        Ratio of number of instances with same class value over the covered instances
    """
    covered_instances, ds_len = _get_fuzzy_coverage(rules, fuzzy_dataset, threshold)
    # All rules have the same consequent because it is a factual explanation
    return np.sum((covered_instances) & (y == rules[0].consequent)) / np.sum(covered_instances)


def fidelity(y, y_local):
    """
    Compute the fidelity of an agnostic white-box classifier

    Parameters
    ----------
    y : array-like, of shape (n_neighbor_instances)
        Black-box predictions for all the instances of the neighborhood
    y_local : array-like, of shape (n_neighbor_instances)
        White-box predictions for all the instances of the neighborhood

    Returns
    -------
    float
        Ratio of number of matching predictions
    """
    return np.sum(y == y_local) / len(y)


def rule_fidelity(y, y_local, fuzzy_dataset, rules, threshold=0.001):
    """
    Compute the fidelity of an agnostic white-box classifier given
    a set of rules

    Parameters
    ----------
    y : array-like, of shape (n_neighbor_instances)
        Black-box predictions for all the instances of the neighborhood
    y_local : array-like, of shape (n_neighbor_instances)
        White-box predictions for all the instances of the neighborhood
    rules : list[Rule]
        List of rules to cover the dataset
    fuzzy_dataset : dict
        Dataset membership with the format {feature_1: {set_1: [memb_1, memb_2, ...]}}
        with each feature and set and an array of shape (n_neighbor_instances) with t
        he membership degree of all the instances of the dataset
    threshold : float, optional
        Activation threshold which sets when an instance is considered covered, by default 0.001

    Returns
    -------
    float
        Ratio of number of matching predictions from the covered instances divided by the
        number of covered instances
    """
    covered_instances, ds_len = _get_fuzzy_coverage(rules, fuzzy_dataset, threshold)
    return np.sum(y[covered_instances] == y_local[covered_instances]) / np.sum(covered_instances)
