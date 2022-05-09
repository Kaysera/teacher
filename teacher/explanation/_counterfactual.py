from copy import deepcopy

import numpy as np
from teacher.tree import Rule


def _compare_rules_FID3(factual, counter_rule):
    fact_ante = {key: val for key, val in factual.antecedent}
    counter_rule_ante = {key: val for key, val in counter_rule.antecedent}

    diffs = set([])

    for elem in fact_ante:
        if not (elem in counter_rule_ante and fact_ante[elem] == counter_rule_ante[elem]):
            diffs.add(elem)

    for elem in counter_rule_ante:
        if not (elem in fact_ante and fact_ante[elem] == counter_rule_ante[elem]):
            diffs.add(elem)
    return len(diffs)


def _cf_dist_instance(cf_rule, instance, df_numerical_columns):
    fuzzified_instance = {feat: max(fuzz_sets, key=lambda x: fuzz_sets[x]) for feat, fuzz_sets in instance.items()}
    cf_dict = {key: val for key, val in cf_rule.antecedent}
    num_keys = [x for x in df_numerical_columns if x in fuzzified_instance and x in cf_dict]

    rule_distance = _get_categorical_cf_distance(fuzzified_instance, cf_dict, df_numerical_columns)
    for key in num_keys:
        rule_distance += _weighted_literal_distance(instance[key], fuzzified_instance[key], cf_dict[key])

    return rule_distance


def _cf_dist_rule(cf_rule, rule, instance, df_numerical_columns, tau=0.5):
    rule_dict = {key: val for key, val in rule.antecedent}
    cf_dict = {key: val for key, val in cf_rule.antecedent}
    num_keys = [x for x in df_numerical_columns if x in rule_dict and x in cf_dict]

    rule_distance = _get_categorical_cf_distance(rule_dict, cf_dict, df_numerical_columns)
    for key in num_keys:
        rule_distance += _literal_distance(instance[key], rule_dict[key], cf_dict[key])

    only_instance = [x for x in rule_dict if x not in cf_dict]
    only_cf = [x for x in cf_dict if x not in rule_dict]
    simmetric_distance = len(only_instance) + len(only_cf)
    return tau * simmetric_distance + (1 - tau) * rule_distance


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
    """Distance between a point and a fuzzy set of a fuzzy variable"""
    return _literal_distance(fuzzy_clause, fuzzy_value, cf_value) * (1 - fuzzy_clause[cf_value])


def _literal_distance(fuzzy_clause, fuzzy_value, cf_value):
    """Distance between two fuzzy sets of a fuzzy variable"""
    skip = abs(list(fuzzy_clause).index(fuzzy_value) - list(fuzzy_clause).index(cf_value))
    distance = skip / (len(fuzzy_clause) - 1)
    return distance


def FID3_counterfactual(factual, counter_rules):
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


def i_counterfactual(instance, rule_list, class_val, df_numerical_columns, new=False):
    """Returns a list that contains the counterfactual with respect to the instance
    for each of the different class values not predicted, as explained in [ref]

    Parameters
    ----------
    instance : dict, {feature: {set_1: pert_1, set_2: pert_2, ...}, ...}
        Fuzzy representation of the instance with all the features and pertenence
        degrees to each fuzzy set
    rule_list : list(Rule)
        List of candidate rules to form part of the counterfactual
    class_val : str
        Predicted value that the factual will explain
    df_numerical_columns : list
        List of the numerical columns of the instance, used to compute the distance

    Returns
    -------
    list(Rule)
        List of counterfactual rules
    """
    counter_rules = [rule for rule in rule_list if rule.consequent != class_val]
    possible_cf = [(rule, _cf_dist_instance(rule, instance, df_numerical_columns))
                   for rule in counter_rules]
    sorted_cf = sorted(possible_cf, key=lambda rule: rule[1])
    if not new:
        return sorted_cf

    for cf in sorted_cf:
        new_instance, changes = _apply_changes(cf[0], instance)
        new_class_val = Rule.weighted_vote(rule_list, new_instance)
        if new_class_val != class_val:
            return changes

    return None


def f_counterfactual(factual, instance, rule_list, class_val, df_numerical_columns, tau=0.5, new=False):
    """Returns a list that contains the counterfactual with respect to the factual
    for each of the different class values not predicted, as explained in [ref]

    Parameters
    ----------
    factual : list(Rule)
        List of rules that correspond to a factual explanation of the
        instance for the class value class_val
    instance : dict, {feature: {set_1: pert_1, set_2: pert_2, ...}, ...}
        Fuzzy representation of the instance with all the features and pertenence
        degrees to each fuzzy set
    rule_list : list(Rule)
        List of candidate rules to form part of the counterfactual
    class_val : str
        Predicted value that the factual will explain
    df_numerical_columns : list
        List of the numerical columns of the instance, used to compute the distance
    tau : float, optional
        Importance degree of new elements added or substracted from a rule
        in contrast to existing elements that have been modified, used
        to compute the distance , by default 0.5

    Returns
    -------
    list(Rule)
        List of counterfactual rules
    """
    counterfactual = []
    counter_rules = [rule for rule in rule_list if rule.consequent != class_val]
    for cf_rule in counter_rules:
        cf_dist = 0
        for fact_rule in factual:
            AD = fact_rule.matching(instance)
            cf_dist += AD * _cf_dist_rule(cf_rule, fact_rule, instance, df_numerical_columns, tau)
        if cf_dist > 0:
            counterfactual.append((cf_rule, cf_dist))
    sorted_cf = sorted(counterfactual, key=lambda rule: rule[1])
    if not new:
        return sorted_cf
    for cf in sorted_cf:
        new_instance, changes = _apply_changes(cf[0], instance)
        new_class_val = Rule.weighted_vote(rule_list, new_instance)
        if new_class_val != class_val:
            return changes

    return None


def _apply_changes(rule, instance):
    changes = set([])
    rule_changes = {a[0]: a[1] for a in rule.antecedent}
    new_instance = deepcopy(instance)
    for fuzzy_var in new_instance:
        max_pert_value = max(new_instance[fuzzy_var], key=lambda fuzzy_set: new_instance[fuzzy_var][fuzzy_set])
        if fuzzy_var in rule_changes and max_pert_value != rule_changes[fuzzy_var]:
            changes.add((fuzzy_var, rule_changes[fuzzy_var]))
            for fuzzy_set in new_instance[fuzzy_var]:
                if fuzzy_set == rule_changes[fuzzy_var]:
                    new_instance[fuzzy_var][fuzzy_set] = 1
                else:
                    new_instance[fuzzy_var][fuzzy_set] = 0

    return new_instance, changes
