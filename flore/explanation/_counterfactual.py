import numpy as np


def get_counterfactual_FID3(factual, counter_rules):
    min_rule_distance = np.inf
    best_cr = []
    for counter_rule in counter_rules:
        rule_distance = _compare_rules_FID3(factual, counter_rule)

        if rule_distance < min_rule_distance:
            min_rule_distance = rule_distance
            best_cr = [counter_rule]

        elif rule_distance == min_rule_distance:
            best_cr += [counter_rule]

    return best_cr, min_rule_distance


def _compare_rules_FID3(factual, counter_rule):
    ex = {key: val for key, val in factual.antecedent}
    cr = {key: val for key, val in counter_rule.antecedent}

    diffs = set([])

    for elem in ex:
        if elem in cr and ex[elem] == cr[elem]:
            pass
        else:
            diffs.add(elem)

    for elem in cr:
        if elem in ex and ex[elem] == cr[elem]:
            pass
        else:
            diffs.add(elem)
    return len(diffs)
