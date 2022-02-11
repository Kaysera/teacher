from functools import reduce
from flore.tree import ID3, ID3_dev, FDT, FDT_dev
from pytest import fixture

from sklearn import datasets
from sklearn.model_selection import train_test_split

from flore.fuzzy import get_fuzzy_triangle, get_fuzzy_set_dataframe, get_fuzzy_points
from flore.datasets import load_compas, load_beer
from flore.explanation import FID3_factual, m_factual, mr_factual, c_factual
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
    return [fuzzy_set_df_train, fuzzy_set_df_test, X_train, y_train, X_test, y_test, fuzzy_element, all_classes]


def _get_fuzzy_element(fuzzy_X, idx):
    element = {}
    for feat in fuzzy_X:
        element[feat] = {}
        for fuzzy_set in fuzzy_X[feat]:
            element[feat][fuzzy_set] = fuzzy_X[feat][fuzzy_set][idx]

    return element


def _get_categorical_fuzzy(var):
    x = [var[k] for k in var]
    label = {i: j for i, j in enumerate(var)}
    return np.array([label[elem] for elem in np.argmax(x, axis=0)])


def _fuzzify_dataset(dataframe, fuzzy_set, fuzzify_variable):
    ndf = dataframe.copy()
    for k in fuzzy_set:
        ndf[k] = fuzzify_variable(fuzzy_set[k])
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
    continuous = dataset['continuous']

    # Dataset Preprocessing
    instance = X_test.iloc[idx_record2explain]

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
    best_rule = _get_best_rule(explanation, operator)[0]

    new_id3 = ID3_dev(fuzzy_X.columns, max_depth=5, min_num_examples=10, prunning=True, th=0.00000001)
    new_id3.fit(X_np, y_np)
    f_instance = _get_fuzzy_element(fuzzy_set_test, 1)
    rules = new_id3.to_rule_based_system()
    factual = FID3_factual(f_instance, rules)

    for exp_rule, fact_rule in zip([best_rule], [factual]):
        for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]
        assert fact_rule.consequent == exp_rule[1]


def test_m_factual_fdt(prepare_iris_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, _ = prepare_iris_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    expect_m_factual = _alpha_factual_avg(predicted_best_rules)

    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    ob_m_factual = m_factual(fuzzy_element, rules, new_fdt_predict)

    for exp_rule, fact_rule in zip(expect_m_factual, ob_m_factual):
        for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_r_factual_fdt(prepare_beer_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, all_classes = prepare_beer_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    other_classes = [cv for cv in all_classes if cv != fdt_predict]
    fdt_rob_thres = fdt.robust_threshold(fuzzy_element, other_classes)
    expect_r_factual = _alpha_factual_robust(predicted_best_rules, fdt_rob_thres)

    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    ob_r_factual = mr_factual(fuzzy_element, rules, new_fdt_predict)

    for exp_rule, fact_rule in zip(expect_r_factual, ob_r_factual):
        for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_c_lambda_factual(prepare_iris_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, _ = prepare_iris_fdt
    lam = 0.98

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    expect_c_factual = _alpha_factual_factor(predicted_best_rules, lam)

    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    ob_c_factual = c_factual(fuzzy_element, rules, new_fdt_predict, lam)

    for exp_rule, fact_rule in zip(expect_c_factual, ob_c_factual):
        for exp_ante, fact_ante in zip(exp_rule, fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_c_lambda_factual_complex_fdt(prepare_beer_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, _ = prepare_beer_fdt
    lam = 0.5

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    expect_c_factual = _alpha_factual_factor(predicted_best_rules, lam)

    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    ob_c_factual = c_factual(fuzzy_element, rules, new_fdt_predict, lam)

    for exp_rule, fact_rule in zip(expect_c_factual, ob_c_factual):
        for exp_ante, fact_ante in zip(exp_rule, fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_c_lambda_beta_factual_fdt(prepare_iris_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, _ = prepare_iris_fdt
    lam = 0.98
    beta = 0.5

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    expect_c_factual = _alpha_factual_factor_sum(predicted_best_rules, lam, beta)

    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    ob_c_factual = c_factual(fuzzy_element, rules, new_fdt_predict, lam, beta)

    for exp_rule, fact_rule in zip(expect_c_factual, ob_c_factual):
        for exp_ante, fact_ante in zip(exp_rule, fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]
