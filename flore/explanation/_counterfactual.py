import numpy as np
from ._factual import _get_maximum_weight_rules


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


def get_instance_counterfactual(instance, rule_list, prediction, df_numerical_columns):
    counterfactual = []
    max_weight_rules = _get_maximum_weight_rules(rule_list)
    counter_rules = [rule for rule in max_weight_rules if rule.consequent != prediction]
    for class_val in np.unique([rule.consequent for rule in counter_rules]):
        possible_cf = [(rule, _cf_dist_instance(instance, rule, df_numerical_columns))
                       for rule in counter_rules if rule.consequent == class_val]
        counterfactual.append((min(possible_cf, key=lambda rule: rule[1])))

    return counterfactual


def _cf_dist_instance(instance, cf_rule, df_numerical_columns):
    fuzzified_instance = {feat: max(fuzz_sets, key=lambda x: fuzz_sets[x]) for feat, fuzz_sets in instance.items()}
    cf_dict = {key: val for key, val in cf_rule.antecedent}
    num_keys = [x for x in df_numerical_columns if x in fuzzified_instance and x in cf_dict]

    rule_distance = _get_categorical_cf_distance(fuzzified_instance, cf_dict, df_numerical_columns)
    for key in num_keys:
        rule_distance += _weighted_literal_distance(instance[key], fuzzified_instance[key], cf_dict[key])

    return rule_distance


def _get_categorical_cf_distance(fuzzy_element, cf_dict, df_numerical_columns):
    common_keys = set([])
    common_keys.update([x for x in fuzzy_element if x in cf_dict])
    common_keys.update([x for x in cf_dict if x in fuzzy_element])

    cat_keys = [x for x in common_keys if x not in df_numerical_columns]

    rule_distance = 0
    for key in cat_keys:
        if cf_dict[key] is not fuzzy_element[key]:
            rule_distance += 1

    return rule_distance


def _weighted_literal_distance(fuzzy_clause, fuzzy_value, cf_value):
    return _literal_distance(fuzzy_clause, fuzzy_value, cf_value) * (1 - fuzzy_clause[cf_value])


def _literal_distance(fuzzy_clause, fuzzy_value, cf_value):
    skip = abs(list(fuzzy_clause).index(fuzzy_value) - list(fuzzy_clause).index(cf_value))
    distance = skip / len(fuzzy_clause)
    return distance
