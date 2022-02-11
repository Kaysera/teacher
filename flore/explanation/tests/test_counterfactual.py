from flore.tree import ID3, ID3_dev, FDT, FDT_dev
from pytest import fixture

from sklearn.model_selection import train_test_split

from flore.fuzzy import get_fuzzy_triangle, get_fuzzy_set_dataframe, get_fuzzy_points
from flore.datasets import load_german, load_beer, load_compas
from flore.explanation import (FID3_factual, FID3_counterfactual, i_counterfactual,
                               mr_factual, f_counterfactual)

from .test_factual import (_get_fuzzy_element, _get_categorical_fuzzy,
                           _fuzzify_dataset, _alpha_factual_robust, _get_best_rule)
import numpy as np
import random


@fixture
def set_random():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    return seed


@fixture
def prepare_beer_fdt(set_random):
    dataset = load_beer()

    df = dataset['df']
    class_name = dataset['class_name']
    X = df.drop(class_name, axis=1)
    y = df[class_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=set_random)

    df_categorical_columns = dataset['discrete']
    class_name = dataset['class_name']
    df_categorical_columns.remove(class_name)
    df_numerical_columns = dataset['continuous']

    df_train = df.loc[X_train.index]
    df_test = df.loc[X_test.index]

    fuzzy_points = get_fuzzy_points(df_train, 'entropy', df_numerical_columns, class_name=class_name)
    fuzzy_set_df_train = get_fuzzy_set_dataframe(df_train, get_fuzzy_triangle, fuzzy_points,
                                                 df_numerical_columns, df_categorical_columns)
    fuzzy_set_df_test = get_fuzzy_set_dataframe(df_test, get_fuzzy_triangle, fuzzy_points,
                                                df_numerical_columns, df_categorical_columns)

    fuzzy_element = _get_fuzzy_element(fuzzy_set_df_test, 48)
    all_classes = dataset['possible_outcomes']
    return [fuzzy_set_df_train, fuzzy_set_df_test, X_train, y_train,
            X_test, y_test, fuzzy_element, all_classes, df_numerical_columns]


@fixture
def prepare_german_fdt(set_random):
    dataset = load_german()

    df = dataset['df']
    class_name = dataset['class_name']
    X = df.drop(class_name, axis=1)
    y = df[class_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=set_random)

    df_categorical_columns = dataset['discrete']
    class_name = dataset['class_name']
    df_categorical_columns.remove(class_name)
    df_numerical_columns = dataset['continuous']

    df_train = df.loc[X_train.index]
    df_test = df.loc[X_test.index]

    fuzzy_points = get_fuzzy_points(df_train, 'entropy', df_numerical_columns, class_name=class_name)
    fuzzy_set_df_train = get_fuzzy_set_dataframe(df_train, get_fuzzy_triangle, fuzzy_points,
                                                 df_numerical_columns, df_categorical_columns)
    fuzzy_set_df_test = get_fuzzy_set_dataframe(df_test, get_fuzzy_triangle, fuzzy_points,
                                                df_numerical_columns, df_categorical_columns)

    fuzzy_element = _get_fuzzy_element(fuzzy_set_df_test, 3)
    all_classes = dataset['possible_outcomes']
    return [fuzzy_set_df_train, fuzzy_set_df_test, X_train, y_train,
            X_test, y_test, fuzzy_element, all_classes, df_numerical_columns]


def _compare_rule(explanation, counter_rule):
    similarities = 0
    ex = {}
    cr = {}

    for elem in explanation:
        ex[elem[0]] = elem[1]

    for elem in counter_rule:
        cr[elem[0]] = elem[1]

    diffs = set([])

    for elem in ex:
        if elem in cr and ex[elem] == cr[elem]:
            similarities += 1
        else:
            diffs.add(elem)

    for elem in cr:
        if elem in ex and ex[elem] == cr[elem]:
            similarities += 1
        else:
            diffs.add(elem)
    return len(diffs)


def _FID3_counterfactual(all_rules, explanation):
    target = explanation[1]
    counter_rules = []

    for rule in all_rules:
        if rule[1] != target:
            counter_rules += [rule]
    min_rule_distance = np.inf
    best_cr = []

    for counter_rule in counter_rules:
        rule_distance = _compare_rule(explanation[0], counter_rule[0])

        if rule_distance < min_rule_distance:
            min_rule_distance = rule_distance
            best_cr = [counter_rule]

        elif rule_distance == min_rule_distance:
            best_cr += [counter_rule]

    return best_cr, min_rule_distance


def test_counterfactual_id3(set_random):
    dataset = load_compas()

    df = dataset['df']
    class_name = dataset['class_name']
    X = df.drop(class_name, axis=1)
    y = df[class_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=set_random)

    idx_record2explain = 1

    fuzzy_labels = ['low', 'mid']

    discrete = dataset['discrete']
    discrete.remove(class_name)
    continuous = dataset['continuous']

    # Dataset Preprocessing
    instance = X_train.iloc[idx_record2explain]

    fuzzy_points = get_fuzzy_points(X_train, 'equal_width', continuous, len(fuzzy_labels))
    fuzzy_set = get_fuzzy_set_dataframe(X_train, get_fuzzy_triangle, fuzzy_points, continuous, discrete)
    fuzzy_set_test = get_fuzzy_set_dataframe(X_test, get_fuzzy_triangle, fuzzy_points, continuous, discrete)
    fuzzy_X = _fuzzify_dataset(X_train, fuzzy_set, _get_categorical_fuzzy)

    X_np = fuzzy_X.values
    y_np = y_train.values

    id3_class = ID3(fuzzy_X.columns, X_np, y_np, max_depth=5, min_num_examples=10, prunning=True, th=0.00000001)
    id3_class.fit(X_np, y_np)

    explanation = id3_class.explainInstance(instance, idx_record2explain, fuzzy_set_test, discrete, verbose=False)

    operator = min
    cf_explanation = _get_best_rule(explanation, operator)[0]
    all_rules = id3_class.exploreTreeFn(verbose=False)
    exp_cf, _ = _FID3_counterfactual(all_rules, cf_explanation)

    new_id3 = ID3_dev(fuzzy_X.columns, max_depth=5, min_num_examples=10, prunning=True, th=0.00000001)
    new_id3.fit(X_np, y_np)
    f_instance = _get_fuzzy_element(fuzzy_set, idx_record2explain)
    rules = new_id3.to_rule_based_system()
    new_id3_pred = new_id3.predict(instance)[0]
    factual = FID3_factual(f_instance, rules)
    obt_cf, _ = FID3_counterfactual(factual, [rule for rule in rules if rule.consequent != new_id3_pred])

    for exp_rule, fact_rule in zip(exp_cf, obt_cf):
        for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]
        assert fact_rule.consequent == exp_rule[1]


def test_i_counterfactual_fdt(prepare_beer_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, all_classes, df_numerical_columns = prepare_beer_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    other_classes = [cv for cv in all_classes if cv != fdt_predict]
    exp_cf = fdt.get_counterfactual(fuzzy_element, other_classes, df_numerical_columns)

    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    obt_cf = i_counterfactual(fuzzy_element, rules, new_fdt_predict, df_numerical_columns)

    exp_cf_dict = {}
    for class_val, (rule, distance) in exp_cf:
        exp_cf_dict[class_val] = (tuple(rule), distance)

    obt_cf_dict = {}
    for rule, distance in obt_cf:
        obt_cf_dict[rule.consequent] = (rule.antecedent, distance)

    assert exp_cf_dict == obt_cf_dict


def test_i_counterfactual_compas_fdt(prepare_german_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, all_classes, df_numerical_columns = prepare_german_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    other_classes = [cv for cv in all_classes if cv != fdt_predict]
    exp_cf = fdt.get_counterfactual(fuzzy_element, other_classes, df_numerical_columns)

    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    obt_cf = i_counterfactual(fuzzy_element, rules, new_fdt_predict, df_numerical_columns)

    exp_cf_dict = {}
    for class_val, (rule, distance) in exp_cf:
        exp_cf_dict[class_val] = (tuple(rule), distance)

    obt_cf_dict = {}
    for rule, distance in obt_cf:
        obt_cf_dict[rule.consequent] = (rule.antecedent, distance)

    assert exp_cf_dict == obt_cf_dict


def test_f_counterfactual_fdt(prepare_beer_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, all_classes, df_numerical_columns = prepare_beer_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    other_classes = [cv for cv in all_classes if cv != fdt_predict]
    fdt_rob_thres = fdt.robust_threshold(fuzzy_element, other_classes)
    fdt_factual = _alpha_factual_robust(predicted_best_rules, fdt_rob_thres)
    exp_cf = fdt.get_alpha_counterfactual(fuzzy_element, other_classes, df_numerical_columns, fdt_factual)

    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    new_fdt_mr_factual = mr_factual(fuzzy_element, rules, new_fdt_predict)
    obt_cf = f_counterfactual(new_fdt_mr_factual, fuzzy_element, rules, new_fdt_predict, df_numerical_columns)

    exp_cf_dict = {}
    for class_val, (rule, distance) in exp_cf:
        exp_cf_dict[class_val] = (tuple(rule), distance)

    obt_cf_dict = {}
    for rule, distance in obt_cf:
        obt_cf_dict[rule.consequent] = (rule.antecedent, distance)

    assert exp_cf_dict == obt_cf_dict
