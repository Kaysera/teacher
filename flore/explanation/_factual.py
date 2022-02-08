from functools import reduce
import numpy as np


def get_factual_FID3(instance, rule_list, threshold=0.01):
    return [rule for rule in rule_list if rule.matching(instance) > threshold]


def _get_maximum_weight_rules(rule_list):
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
    other_classes = np.unique([rule.consequent for rule in rule_list if rule.consequent != class_val])
    all_th = []
    for cv in other_classes:
        th = 0
        for rule in rule_list:
            if rule.consequent == cv:
                th += rule.matching(instance) * rule.weight

        all_th.append(th)

    return max(all_th)


def get_factual_threshold(instance, rule_list, class_val, threshold, debug=False):
    fired_rules = get_factual_FID3(instance, rule_list)
    if threshold == 'mean':
        avg = reduce(lambda x, y: x + y.matching(instance), fired_rules, 0) / len(fired_rules)
        max_weight_class = _prepare_factual(fired_rules, class_val)
        return [rule for rule in max_weight_class if rule.matching(instance) > avg]
    elif threshold == 'robust':
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
    else:
        raise ValueError('Threshold method not supported')
