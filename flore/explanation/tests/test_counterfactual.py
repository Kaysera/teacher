from flore.tree import ID3, ID3_dev, FDT, FDT_dev
from pytest import fixture

from sklearn import datasets
from sklearn.model_selection import train_test_split

from flore.fuzzy import get_fuzzy_triangle, get_fuzzy_set_dataframe, get_fuzzy_points
from flore.datasets import load_compas, load_beer
from flore.explanation import (get_factual_FID3, get_counterfactual_FID3, get_instance_counterfactual,
                               get_threshold_factual, get_factual_counterfactual)

from .test_factual import _get_fuzzy_element, _get_categorical_fuzzy, _fuzzify_dataset, _alpha_factual_robust
import numpy as np
import random


@fixture
def set_random():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    return seed


@fixture
def prepare_iris_fdt(set_random):
    iris = datasets.load_iris(as_frame=True)

    df_numerical_columns = iris.feature_names
    df_categorical_columns = []
    class_name = 'target'

    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.33,
                                                        random_state=set_random)

    df_train = iris.frame.loc[X_train.index]
    df_test = iris.frame.loc[X_test.index]

    fuzzy_points = get_fuzzy_points(df_train, 'entropy', df_numerical_columns, class_name=class_name)
    fuzzy_set_df_train = get_fuzzy_set_dataframe(df_train, get_fuzzy_triangle, fuzzy_points,
                                                 df_numerical_columns, df_categorical_columns)
    fuzzy_set_df_test = get_fuzzy_set_dataframe(df_test, get_fuzzy_triangle, fuzzy_points,
                                                df_numerical_columns, df_categorical_columns)

    fuzzy_element = _get_fuzzy_element(fuzzy_set_df_test, 27)
    all_classes = np.unique(iris.target)
    return [fuzzy_set_df_train, fuzzy_set_df_test, X_train, y_train, X_test, y_test, fuzzy_element, all_classes]


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


def _compare_rule(explanation, counter_rule):
    # TODO REFACTOR SOME DAY
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


def _counterfactual_FID3(all_rules, explanation):
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


def _get_best_rule(rules, op, target=None):
    best_rule = []
    best_score = 0

    for rule in rules:
        rule_score = 1
        if target is None or target == rule[1]:
            for clause in rule[0]:
                rule_score = op([rule_score, clause[2]])

            if rule_score > best_score:
                best_score = rule_score
                best_rule = rule

    return (best_rule, best_score)


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
    cf_exp = _get_best_rule(explanation, operator)[0]
    all_rules = id3_class.exploreTreeFn(verbose=False)
    cf, cf_len = _counterfactual_FID3(all_rules, cf_exp)

    new_id3 = ID3_dev(fuzzy_X.columns, max_depth=5, min_num_examples=10, prunning=True, th=0.00000001)
    new_id3.fit(X_np, y_np)
    f_instance = _get_fuzzy_element(fuzzy_set, idx_record2explain)
    rules = new_id3.to_rule_based_system()
    new_id3_pred = new_id3.predict(instance)[0]
    factual = get_factual_FID3(f_instance, rules)
    best_rule = max(factual, key=lambda r: r.matching(f_instance))
    cf_new, brdist = get_counterfactual_FID3(best_rule, [rule for rule in rules if rule.consequent != new_id3_pred])

    for exp_rule, fact_rule in zip(cf, cf_new):
        for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]
        assert fact_rule.consequent == exp_rule[1]


def test_instance_cf_fdt(prepare_beer_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, all_classes, df_numerical_columns = prepare_beer_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    other_classes = [cv for cv in all_classes if cv != fdt_predict]
    fdt_cf = fdt.get_counterfactual(fuzzy_element, other_classes, df_numerical_columns)

    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    new_fdt_cf = get_instance_counterfactual(fuzzy_element, rules, new_fdt_predict, df_numerical_columns)

    fdt_cf_dict = {}
    for class_val, (rule, distance) in fdt_cf:
        fdt_cf_dict[class_val] = (tuple(rule), distance)

    new_fdt_cf_dict = {}
    for rule, distance in new_fdt_cf:
        new_fdt_cf_dict[rule.consequent] = (rule.antecedent, distance)

    assert fdt_cf_dict == new_fdt_cf_dict


def test_factual_cf_fdt(prepare_beer_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, all_classes, df_numerical_columns = prepare_beer_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    other_classes = [cv for cv in all_classes if cv != fdt_predict]
    fdt_rob_thres = fdt.robust_threshold(fuzzy_element, other_classes)
    alpha_factuals = _alpha_factual_robust(predicted_best_rules, fdt_rob_thres)
    fdt_cf = fdt.get_alpha_counterfactual(fuzzy_element, other_classes, df_numerical_columns, alpha_factuals)

    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    factual = get_threshold_factual(fuzzy_element, rules, new_fdt_predict, 'robust')
    new_fdt_cf = get_factual_counterfactual(factual, fuzzy_element, rules, new_fdt_predict, df_numerical_columns)

    fdt_cf_dict = {}
    for class_val, (rule, distance) in fdt_cf:
        fdt_cf_dict[class_val] = (tuple(rule), distance)

    new_fdt_cf_dict = {}
    for rule, distance in new_fdt_cf:
        new_fdt_cf_dict[rule.consequent] = (rule.antecedent, distance)

    assert fdt_cf_dict == new_fdt_cf_dict
