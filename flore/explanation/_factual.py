from functools import reduce


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


def get_factual_threshold(instance, rule_list, class_val, threshold, robust_threshold=None, debug=False):
    if threshold == 'mean':
        fired_rules = get_factual_FID3(instance, rule_list)
        avg = reduce(lambda x, y: x + y.matching(instance), fired_rules, 0) / len(fired_rules)
        max_weight_class = _prepare_factual(fired_rules, class_val)
        return [rule for rule in max_weight_class if rule.matching(instance) > avg]
    elif threshold == 'robust':
        if robust_threshold is None:
            raise ValueError('robust_threshold must be a float for this threshold method')
        pass
    else:
        raise ValueError('Threshold method not supported')
