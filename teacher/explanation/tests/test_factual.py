from functools import reduce
from teacher.tree import ID3, FDT
from teacher.tree.tests.fdt_legacy_tree import FDT_Legacy
from teacher.tree.tests.id3_legacy_tree import ID3_Legacy
from pytest import fixture

from sklearn import datasets
from sklearn.model_selection import train_test_split

from teacher.fuzzy import get_fuzzy_variables, get_fuzzy_points, dataset_membership
from teacher.datasets import load_compas, load_beer
from teacher.explanation import FID3_factual, m_factual, mr_factual, c_factual
import numpy as np
import pandas as pd
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

    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.33,
                                                        random_state=set_random)

    X_num = X_train
    num_cols = X_num.columns
    fuzzy_points = get_fuzzy_points('entropy', num_cols, X_num, y_train)

    discrete_fuzzy_values = {col: X_train[col].unique() for col in df_categorical_columns}
    fuzzy_variables_order = {col: i for i, col in enumerate(X_train.columns)}
    fuzzy_variables = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, fuzzy_variables_order)

    df_train_membership = dataset_membership(X_train, fuzzy_variables)
    df_test_membership = dataset_membership(X_test, fuzzy_variables)

    fuzzy_element_idx = 48
    fuzzy_element = _get_fuzzy_element(df_test_membership, fuzzy_element_idx)

    all_classes = np.unique(iris.target)
    return [df_train_membership, df_test_membership, X_train, y_train,
            X_test, y_test, fuzzy_element, fuzzy_element_idx, all_classes, df_numerical_columns, fuzzy_variables]


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

    X_num = X_train[dataset['continuous']]
    num_cols = X_num.columns
    fuzzy_points = get_fuzzy_points('entropy', num_cols, X_num, y_train)

    discrete_fuzzy_values = {col: df[col].unique() for col in df_categorical_columns}
    fuzzy_variables_order = {col: i for i, col in enumerate(X.columns)}
    fuzzy_variables = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, fuzzy_variables_order)

    df_train_membership = dataset_membership(X_train, fuzzy_variables)
    df_test_membership = dataset_membership(X_test, fuzzy_variables)

    fuzzy_element_idx = 48
    fuzzy_element = _get_fuzzy_element(df_test_membership, fuzzy_element_idx)
    all_classes = dataset['possible_outcomes']
    return [df_train_membership, df_test_membership, X_train, y_train,
            X_test, y_test, fuzzy_element, fuzzy_element_idx, all_classes, df_numerical_columns, fuzzy_variables]


def _get_fuzzy_element(fuzzy_X, idx):
    element = {}
    for feat in fuzzy_X:
        element[feat] = {}
        for fuzzy_set in fuzzy_X[feat]:
            try:
                element[feat][str(fuzzy_set)] = pd.to_numeric(fuzzy_X[feat][fuzzy_set][idx])
            except ValueError:
                element[feat][str(fuzzy_set)] = fuzzy_X[feat][fuzzy_set][idx]

    return element


def _get_categorical_fuzzy(var):
    x = [var[k] for k in var]
    label = {i: j for i, j in enumerate(var)}
    return np.array([label[elem] for elem in np.argmax(x, axis=0)])


def _fuzzify_dataset(dataframe, fuzzy_set, fuzzify_variable):
    ndf = dataframe.copy()
    for k in fuzzy_set:
        try:
            ndf[str(k)] = pd.to_numeric(fuzzify_variable(fuzzy_set[k]))
        except ValueError:
            ndf[str(k)] = fuzzify_variable(fuzzy_set[k])
    return ndf


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


def _alpha_factual_avg(explanations):
    avg = reduce(lambda x, y: x + y[1], explanations, 0) / len(explanations)
    first_class_dict, first_matching, first_rule = explanations[0]
    alpha_factual = [(first_rule, first_matching)]
    for class_dict, matching, rule in explanations[1:]:
        if matching >= avg:
            alpha_factual += [rule]
        else:
            break

    return alpha_factual


def _alpha_factual_robust(explanations, threshold):
    # This is the cummulative mu of the
    # rules that will be selected
    first_class_dict, first_matching, first_rule = explanations[0]
    total_mu = first_matching
    alpha_factual = [(first_rule, first_matching)]
    for class_dict, matching, rule in explanations[1:]:
        if total_mu >= threshold:
            break
        total_mu += matching
        alpha_factual += [rule]

    return alpha_factual


def _alpha_factual_factor(explanations, alpha):
    first_class_dict, first_matching, first_rule = explanations[0]
    alpha_factual = [first_rule]
    prev_matching = first_matching
    for class_dict, matching, rule in explanations[1:]:
        factor = prev_matching / matching
        if factor <= 1 + alpha:
            prev_matching = matching
            alpha_factual += [rule]
        else:
            break

    return alpha_factual


def _alpha_factual_factor_sum(explanations, alpha, beta):
    first_class_dict, first_matching, first_rule = explanations[0]
    alpha_factual = [first_rule]
    prev_matching = first_matching
    total_mu = first_matching
    for class_dict, matching, rule in explanations[1:]:
        factor = prev_matching / matching
        if total_mu < beta or factor <= 1 + alpha:
            prev_matching = matching
            total_mu += matching
            alpha_factual += [rule]
        else:
            break

    return alpha_factual


def test_explain_id3(set_random):
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

    # Dataset Preprocessing
    instance = X_train.iloc[idx_record2explain]

    X_num = X_train[dataset['continuous']]
    num_cols = X_num.columns
    fuzzy_points = get_fuzzy_points('equal_width', num_cols, X_num, sets=len(fuzzy_labels))

    discrete_fuzzy_values = {col: df[col].unique() for col in discrete}
    fuzzy_variables_order = {col: i for i, col in enumerate(X.columns)}
    fuzzy_variables = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, fuzzy_variables_order)

    df_train_membership = dataset_membership(X_train, fuzzy_variables)
    df_test_membership = dataset_membership(X_test, fuzzy_variables)

    fuzzy_X = _fuzzify_dataset(X_train, df_train_membership, _get_categorical_fuzzy)

    X_np = fuzzy_X.values
    y_np = y_train.values

    id3_class = ID3_Legacy(fuzzy_X.columns, X_np, y_np, max_depth=5, min_num_examples=10, prunning=True, th=0.00000001)
    id3_class.fit(X_np, y_np)

    explanation = id3_class.explainInstance(instance, idx_record2explain, df_test_membership, discrete, verbose=False)

    operator = min
    best_rule = _get_best_rule(explanation, operator)[0]

    # CONVERTING DISCRETE VALUES TO STRING BECAUSE
    # ID3 TREE CASTS TO INT IF IT CAN
    for ante, conse in [best_rule]:
        for i, clause in enumerate(ante):
            if clause[0] in discrete:
                ante[i] = (clause[0], str(clause[1]))

    new_id3 = ID3(fuzzy_X.columns, max_depth=5, min_num_examples=10, prunning=True, th=0.00000001)
    new_id3.fit(X_np, y_np)
    f_instance = _get_fuzzy_element(df_train_membership, idx_record2explain)
    rules = new_id3.to_rule_based_system()

    # CONVERTING DISCRETE VALUES TO STRING BECAUSE
    # ID3 TREE CASTS TO INT IF IT CAN
    for rule in rules:
        for i, ante in enumerate(rule.antecedent):
            if ante[0] in discrete:
                ante_list = list(rule.antecedent)
                ante_list[i] = (ante[0], str(ante[1]))
                rule.antecedent = tuple(ante_list)

    factual = FID3_factual(f_instance, rules)

    for exp_rule, fact_rule in zip([best_rule], [factual]):
        for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]
        assert fact_rule.consequent == exp_rule[1]


def test_m_factual_fdt(prepare_iris_fdt):
    (df_train_membership, _, X_train, y_train,
     X_test, _, fuzzy_element, fuzzy_element_idx, _, _, fuzzy_variables) = prepare_iris_fdt

    fdt = FDT_Legacy(df_train_membership.keys(), df_train_membership)
    fdt.fit(X_train, y_train)

    new_fdt = FDT(fuzzy_variables)
    new_fdt.fit(X_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    expect_m_factual = _alpha_factual_avg(predicted_best_rules)

    new_fdt_predict = new_fdt.predict(X_test.iloc[fuzzy_element_idx].to_numpy().reshape(1, -1))
    rules = new_fdt.to_rule_based_system()
    ob_m_factual = m_factual(fuzzy_element, rules, new_fdt_predict)

    for exp_rule, fact_rule in zip(expect_m_factual, ob_m_factual):
        for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_r_factual_fdt(prepare_beer_fdt):
    (df_train_membership, _, X_train, y_train,
     X_test, _, fuzzy_element, fuzzy_element_idx, all_classes, _, fuzzy_variables) = prepare_beer_fdt

    fdt = FDT_Legacy(df_train_membership.keys(), df_train_membership)
    fdt.fit(X_train, y_train)

    new_fdt = FDT(fuzzy_variables)
    new_fdt.fit(X_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    other_classes = [cv for cv in all_classes if cv != fdt_predict]
    fdt_rob_thres = fdt.robust_threshold(fuzzy_element, other_classes)
    expect_r_factual = _alpha_factual_robust(predicted_best_rules, fdt_rob_thres)

    new_fdt_predict = new_fdt.predict(X_test.iloc[fuzzy_element_idx].to_numpy().reshape(1, -1))
    rules = new_fdt.to_rule_based_system()
    ob_r_factual = mr_factual(fuzzy_element, rules, new_fdt_predict)

    for exp_rule, fact_rule in zip(expect_r_factual, ob_r_factual):
        for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_c_lambda_factual(prepare_iris_fdt):
    (df_train_membership, _, X_train, y_train,
     X_test, _, fuzzy_element, fuzzy_element_idx, _, _, fuzzy_variables) = prepare_iris_fdt
    lam = 0.98

    fdt = FDT_Legacy(df_train_membership.keys(), df_train_membership)
    fdt.fit(X_train, y_train)

    new_fdt = FDT(fuzzy_variables)
    new_fdt.fit(X_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    expect_c_factual = _alpha_factual_factor(predicted_best_rules, lam)

    new_fdt_predict = new_fdt.predict(X_test.iloc[fuzzy_element_idx].to_numpy().reshape(1, -1))
    rules = new_fdt.to_rule_based_system()
    ob_c_factual = c_factual(fuzzy_element, rules, new_fdt_predict, lam)

    for exp_rule, fact_rule in zip(expect_c_factual, ob_c_factual):
        for exp_ante, fact_ante in zip(exp_rule, fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_c_lambda_factual_complex_fdt(prepare_beer_fdt):
    (df_train_membership, _, X_train, y_train,
     X_test, _, fuzzy_element, fuzzy_element_idx, _, _, fuzzy_variables) = prepare_beer_fdt
    lam = 0.5

    fdt = FDT_Legacy(df_train_membership.keys(), df_train_membership)
    fdt.fit(X_train, y_train)

    new_fdt = FDT(fuzzy_variables)
    new_fdt.fit(X_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    expect_c_factual = _alpha_factual_factor(predicted_best_rules, lam)

    new_fdt_predict = new_fdt.predict(X_test.iloc[fuzzy_element_idx].to_numpy().reshape(1, -1))
    rules = new_fdt.to_rule_based_system()
    ob_c_factual = c_factual(fuzzy_element, rules, new_fdt_predict, lam)

    for exp_rule, fact_rule in zip(expect_c_factual, ob_c_factual):
        for exp_ante, fact_ante in zip(exp_rule, fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_c_lambda_beta_factual_fdt(prepare_iris_fdt):
    (df_train_membership, _, X_train, y_train,
     X_test, _, fuzzy_element, fuzzy_element_idx, _, _, fuzzy_variables) = prepare_iris_fdt
    lam = 0.98
    beta = 0.5

    fdt = FDT_Legacy(df_train_membership.keys(), df_train_membership)
    fdt.fit(X_train, y_train)

    new_fdt = FDT(fuzzy_variables)
    new_fdt.fit(X_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    expect_c_factual = _alpha_factual_factor_sum(predicted_best_rules, lam, beta)

    new_fdt_predict = new_fdt.predict(X_test.iloc[fuzzy_element_idx].to_numpy().reshape(1, -1))
    rules = new_fdt.to_rule_based_system()
    ob_c_factual = c_factual(fuzzy_element, rules, new_fdt_predict, lam, beta)

    for exp_rule, fact_rule in zip(expect_c_factual, ob_c_factual):
        for exp_ante, fact_ante in zip(exp_rule, fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]
