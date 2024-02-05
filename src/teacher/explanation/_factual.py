"""Methods to generate a factual explanation"""

# =============================================================================
# Imports
# =============================================================================

# Standard library
from functools import reduce

# Third party
import numpy as np

# =============================================================================
# Functions
# =============================================================================


def _fired_rules(instance, rule_list, threshold=0.001):
    """
    Return the rules fired by the instance given a threshold.

    Parameters
    ----------
    instance : dict, of format {set_1: pert_1, set_2: pert_2, ...}, ...}
        Fuzzy representation of the instance with all the features and pertenence
        degrees to each fuzzy set
    rule_list : list[Rule]
        List of candidate rules to form part of the factual
    threshold : float, optional
        Activation threshold with which a rule is
        considered to be fired by the instance, by default 0.01

    Returns
    -------
    list[Rule]
        List of fired rules
    """
    return [rule for rule in rule_list if rule.matching(instance) > threshold]


def _get_class_value_rules(rule_list, class_val):
    """Obtain the rules with the consequent equals to a certain value"""
    return [rule for rule in rule_list if rule.consequent == class_val]


def _robust_threshold(instance, rule_list, class_val):
    """Obtain the robust threshold"""
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
    """
    Generate the factual extracted for the :class:`.ID3`,
    this is, the rule with the maximum matching with the instance.

    Parameters
    ----------
    instance : dict, of format {feature: {set_1: pert_1, set_2: pert_2, ...}, ...}
        Fuzzy representation of the instance with all the features and pertenence
        degrees to each fuzzy set
    rule_list : list[Rule]
        List of candidate rules to form part of the factual
    threshold : float, optional
        Activation threshold with which a rule is
        considered to be fired by the instance, by default 0.01

    Returns
    -------
    list[Rule]
        List of factual rules
    """
    fired_rules = _fired_rules(instance, rule_list, threshold)
    return max(fired_rules, key=lambda rule: rule.matching(instance))


def m_factual(instance, rule_list, class_val):
    """
    Generate the factual associated to the mean.

    Parameters
    ----------
    instance : dict, of format {feature: {set_1: pert_1, set_2: pert_2, ...}, ...}
        Fuzzy representation of the instance with all the features and pertenence
        degrees to each fuzzy set
    rule_list : list[Rule]
        List of candidate rules to form part of the factual
    class_val : str
        Predicted value that the factual will explain

    Returns
    -------
    list[Rule]
        List of factual rules
    """
    fired_rules = _fired_rules(instance, rule_list)
    class_fired_rules = _get_class_value_rules(fired_rules, class_val)
    class_fired_rules.sort(key=lambda rule: rule.matching(instance) * rule.weight, reverse=True)
    avg = reduce(lambda x, y: x + (y.matching(instance) * y.weight), class_fired_rules, 0) / len(class_fired_rules)
    return [rule for rule in class_fired_rules if rule.matching(instance) * rule.weight >= avg]


def mr_factual(instance, rule_list, class_val):
    """
    Generate the minimum robust factual.

    Parameters
    ----------
    instance : dict, of format {set_1: pert_1, set_2: pert_2, ...}, ...}
        Fuzzy representation of the instance with all the features and pertenence
        degrees to each fuzzy set
    rule_list : list[Rule]
        List of candidate rules to form part of the factual
    class_val : str
        Predicted value that the factual will explain

    Returns
    -------
    list[Rule]
        List of factual rules
    """
    fired_rules = _fired_rules(instance, rule_list)
    class_fired_rules = _get_class_value_rules(fired_rules, class_val)
    class_fired_rules.sort(key=lambda rule: rule.matching(instance) * rule.weight, reverse=True)
    robust_threshold = _robust_threshold(instance, rule_list, class_val)

    factual = []
    AD_sum = 0
    for rule in class_fired_rules:
        if robust_threshold < AD_sum:
            break
        factual.append(rule)
        AD_sum += rule.matching(instance) * rule.weight
    return factual


def c_factual(instance, rule_list, class_val, lam, beta=None):
    """
    Generate the factual associated to the lambda quotient.
    If beta is passed, it generate the minimum mass factual.

    Parameters
    ----------
    instance : dict, of format {set_1: pert_1, set_2: pert_2, ...}, ...}
        Fuzzy representation of the instance with all the features and pertenence
        degrees to each fuzzy set
    rule_list : list[Rule]
        List of candidate rules to form part of the factual
    class_val : str
        Predicted value that the factual will explain
    lam : float, greater or equals to 0
        Lambda quotient to determine which rules form part of the factual
    beta : float, greater than 0, optional
        If passed, minimum mass to obtain the rules which
        form part of the factual, by default None

    Returns
    -------
    list[Rule]
        List of factual rules
    """
    fired_rules = _fired_rules(instance, rule_list)
    class_fired_rules = _get_class_value_rules(fired_rules, class_val)
    class_fired_rules.sort(key=lambda rule: rule.matching(instance) * rule.weight, reverse=True)
    factual = [class_fired_rules[0]]
    prev_matching = factual[0].matching(instance) * factual[0].weight
    AD_sum = prev_matching
    for rule in class_fired_rules[1:]:
        matching = rule.matching(instance) * rule.weight
        factor = prev_matching / matching
        if factor > 1 + lam:
            if beta is None or beta <= AD_sum:
                break

        prev_matching = matching
        AD_sum += matching
        factual.append(rule)

    return factual
