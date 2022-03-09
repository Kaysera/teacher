import numpy as np
from ._factual import _get_maximum_weight_rules


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
    distance = skip / len(fuzzy_clause)
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


def i_counterfactual(instance, rule_list, class_val, df_numerical_columns):
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
    counterfactual = []
    max_weight_rules = _get_maximum_weight_rules(rule_list)
    counter_rules = [rule for rule in max_weight_rules if rule.consequent != class_val]
    for class_val in np.unique([rule.consequent for rule in counter_rules]):
        possible_cf = [(rule, _cf_dist_instance(rule, instance, df_numerical_columns))
                       for rule in counter_rules if rule.consequent == class_val]
        counterfactual.append((min(possible_cf, key=lambda rule: rule[1])))

    return counterfactual


def f_counterfactual(factual, instance, rule_list, class_val, df_numerical_columns, tau=0.5):
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
    max_weight_rules = _get_maximum_weight_rules(rule_list)
    counter_rules = [rule for rule in max_weight_rules if rule.consequent != class_val]
    for class_val in np.unique([rule.consequent for rule in counter_rules]):
        possible_cf = []
        for cf_rule in (rule for rule in counter_rules if rule.consequent == class_val):
            cf_dist = 0
            for fact_rule in factual:
                AD = fact_rule.matching(instance) * fact_rule.weight
                cf_dist += (1 - AD) * _cf_dist_rule(cf_rule, fact_rule, instance, df_numerical_columns, tau)
            possible_cf.append((cf_rule, cf_dist))
        counterfactual.append((min(possible_cf, key=lambda rule: rule[1])))

    return counterfactual
