from functools import reduce
import numpy as np


def _get_maximum_weight_rules(rule_list):
    """Obtain the maximum weight rules from a given rule list"""
    max_weight = {}
    for rule in rule_list:
        if rule.antecedent not in max_weight:
            max_weight[rule.antecedent] = rule
        else:
            if max_weight[rule.antecedent].weight < rule.weight:
                max_weight[rule.antecedent] = rule
    return max_weight.values()


def _prepare_factual(rule_list, class_val):
    max_weight = _get_maximum_weight_rules(rule_list)
    return [rule for rule in max_weight if rule.consequent == class_val]


def _robust_threshold(instance, rule_list, class_val):
    """Obtain the robust threshold as explained in [ref]"""
    other_classes = np.unique([rule.consequent for rule in rule_list if rule.consequent != class_val])
    all_th = []
    for cv in other_classes:
        th = 0
        for rule in rule_list:
            if rule.consequent == cv:
                th += rule.matching(instance) * rule.weight

        all_th.append(th)

    return max(all_th)


def FID3_factual(instance, rule_list, threshold=0.01):
    """Returns the factual extracted for the Fuzzy ID3
    tree in this package

    Parameters
    ----------
    instance : dict, {feature: {set_1: pert_1, set_2: pert_2, ...}, ...}
        Fuzzy representation of the instance with all the features and pertenence
        degrees to each fuzzy set
    rule_list : list(Rule)
        List of candidate rules to form part of the factual
    threshold : float, optional
        Activation threshold with which a rule is
        considered to be fired by the instance, by default 0.01

    Returns
    -------
    list(Rule)
        List of factual rules
    """
    return [rule for rule in rule_list if rule.matching(instance) > threshold]


def m_factual(instance, rule_list, class_val):
    """Returns the factual associated to the mean generated
    as explained in [ref]

    Parameters
    ----------
    instance : dict, {feature: {set_1: pert_1, set_2: pert_2, ...}, ...}
        Fuzzy representation of the instance with all the features and pertenence
        degrees to each fuzzy set
    rule_list : list(Rule)
        List of candidate rules to form part of the factual
    class_val : str
        Predicted value that the factual will explain

    Returns
    -------
    list(Rule)
        List of factual rules
    """
    fired_rules = FID3_factual(instance, rule_list)
    max_weight_class = _prepare_factual(fired_rules, class_val)
    max_weight_class.sort(key=lambda rule: rule.matching(instance) * rule.weight, reverse=True)
    avg = reduce(lambda x, y: x + y.matching(instance), fired_rules, 0) / len(fired_rules)
    return [rule for rule in max_weight_class if rule.matching(instance) > avg]


def mr_factual(instance, rule_list, class_val):
    """Returns the minimum robust factual generated
    as explained in [ref]

    Parameters
    ----------
    instance : dict, {feature: {set_1: pert_1, set_2: pert_2, ...}, ...}
        Fuzzy representation of the instance with all the features and pertenence
        degrees to each fuzzy set
    rule_list : list(Rule)
        List of candidate rules to form part of the factual
    class_val : str
        Predicted value that the factual will explain

    Returns
    -------
    list(Rule)
        List of factual rules
    """
    fired_rules = FID3_factual(instance, rule_list)
    max_weight_class = _prepare_factual(fired_rules, class_val)
    max_weight_class.sort(key=lambda rule: rule.matching(instance) * rule.weight, reverse=True)
    robust_threshold = _robust_threshold(instance, rule_list, class_val)

    factual = []
    AD_sum = 0
    for rule in max_weight_class:
        if robust_threshold < AD_sum:
            break
        factual.append(rule)
        AD_sum += rule.matching(instance) * rule.weight
    return factual


def c_factual(instance, rule_list, class_val, lam, beta=None):
    """Returns the factual associated to the lambda quotient generated
    as explained in [ref]. If beta is passed, it returns the minimum
    mass factual

    Parameters
    ----------
    instance : dict, {feature: {set_1: pert_1, set_2: pert_2, ...}, ...}
        Fuzzy representation of the instance with all the features and pertenence
        degrees to each fuzzy set
    rule_list : list(Rule)
        List of candidate rules to form part of the factual
    class_val : str
        Predicted value that the factual will explain
    lam : float, between (0,1)
        Lambda quotient to determine which rules form part of the factual
    beta : float, between (0,1), optional
        If passed, determines the minimum mass to obtain the rules which
        form part of the factual, by default None

    Returns
    -------
    list(Rule)
        List of factual rules
    """
    fired_rules = FID3_factual(instance, rule_list)
    max_weight_class = _prepare_factual(fired_rules, class_val)
    max_weight_class.sort(key=lambda rule: rule.matching(instance) * rule.weight, reverse=True)
    factual = [max_weight_class[0]]
    prev_matching = factual[0].matching(instance) * factual[0].weight
    AD_sum = prev_matching
    for rule in max_weight_class[1:]:
        matching = rule.matching(instance) * rule.weight
        factor = prev_matching / matching
        if factor > 1 + lam:
            if beta is None or beta >= AD_sum:
                break

        prev_matching = matching
        AD_sum += matching
        factual.append(rule)

    return factual
