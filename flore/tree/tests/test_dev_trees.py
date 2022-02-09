from functools import reduce
from flore.tree import ID3, ID3_dev, FDT, FDT_dev
from pytest import fixture, raises

from sklearn import datasets
from sklearn.model_selection import train_test_split

from flore.fuzzy import get_fuzzy_triangle, get_fuzzy_set_dataframe, get_fuzzy_points
from flore.datasets import load_compas, load_beer
from flore.explanation import get_factual_FID3, get_factual_threshold, get_factual_difference
import numpy as np
import random


@fixture
def set_random():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    return seed


@fixture
def prepare_wine(set_random):
    wine = datasets.load_wine(as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                        wine.target,
                                                        test_size=0.33,
                                                        random_state=set_random)

    return [wine.feature_names, X_train, X_test, y_train, y_test]


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


def _alpha_factual_avg(explanations, alpha, debug=False):
    avg = reduce(lambda x, y: x + y[1], explanations, 0) / len(explanations)
    first_class_dict, first_matching, first_rule = explanations[0]
    alpha_factual = [(first_rule, first_matching)]
    for class_dict, matching, rule in explanations[1:]:
        if matching >= avg:
            if debug:  # pragma: no cover
                alpha_factual += [(rule, matching)]
            else:
                alpha_factual += [rule]
        else:
            break

    if debug:  # pragma: no cover
        total_mu = 0
        for rule, matching in alpha_factual:
            total_mu += matching
        return alpha_factual, total_mu
    else:
        return alpha_factual


def _alpha_factual_robust(explanations, threshold, debug=False):
    # This is the cummulative mu of the
    # rules that will be selected
    first_class_dict, first_matching, first_rule = explanations[0]
    total_mu = first_matching
    alpha_factual = [(first_rule, first_matching)]
    for class_dict, matching, rule in explanations[1:]:
        if total_mu >= threshold:
            break
        total_mu += matching
        if debug:  # pragma: no cover
            alpha_factual += [(rule, matching)]
        else:
            alpha_factual += [rule]

    if debug:  # pragma: no cover
        return alpha_factual, total_mu
    else:
        return alpha_factual


def _alpha_factual_factor(explanations, alpha, debug=False):
    first_class_dict, first_matching, first_rule = explanations[0]
    if debug:  # pragma: no cover
        alpha_factual = [(first_rule, first_matching)]
    else:
        alpha_factual = [first_rule]
    prev_matching = first_matching
    for class_dict, matching, rule in explanations[1:]:
        factor = prev_matching / matching
        if factor <= 1 + alpha:
            prev_matching = matching
            if debug:  # pragma: no cover
                alpha_factual += [(rule, matching)]
            else:
                alpha_factual += [rule]
        else:
            break

    if debug:  # pragma: no cover
        total_mu = 0
        for rule, matching in alpha_factual:
            total_mu += matching
        return alpha_factual, total_mu
    else:
        return alpha_factual


def _alpha_factual_factor_sum(explanations, alpha, beta, debug=False):
    first_class_dict, first_matching, first_rule = explanations[0]
    if debug:  # pragma: no cover
        alpha_factual = [(first_rule, first_matching)]
    else:
        alpha_factual = [first_rule]
    prev_matching = first_matching
    total_mu = first_matching
    for class_dict, matching, rule in explanations[1:]:
        factor = prev_matching / matching
        if total_mu < beta or factor <= 1 + alpha:
            prev_matching = matching
            total_mu += matching
            if debug:  # pragma: no cover
                alpha_factual += [(rule, matching)]
            else:
                alpha_factual += [rule]
        else:
            break

    if debug:  # pragma: no cover
        return alpha_factual, total_mu
    else:
        return alpha_factual


def test_wine_id3(prepare_wine):
    feature_names, X_train, X_test, y_train, y_test = prepare_wine

    id3 = ID3(feature_names, X_train.values, y_train)
    id3.fit(X_train.values, y_train)

    new_id3 = ID3_dev(feature_names)
    new_id3.fit(X_train.values, y_train)

    assert id3.score(X_test.values, y_test) == new_id3.score(X_test.values, y_test)
    assert id3.tree == new_id3.tree_


def test_rules_id3(prepare_wine):
    feature_names, X_train, X_test, y_train, y_test = prepare_wine

    id3 = ID3(feature_names, X_train.values, y_train)
    id3.fit(X_train.values, y_train)

    new_id3 = ID3_dev(feature_names)
    new_id3.fit(X_train.values, y_train)

    rules = []
    new_id3.tree_._get_rules(new_id3.tree_, rules, [])
    assert id3.exploreTreeFn() == rules


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

    new_id3 = ID3_dev(fuzzy_X.columns, max_depth=5, min_num_examples=10, prunning=True, th=0.00000001)
    new_id3.fit(X_np, y_np)
    f_instance = _get_fuzzy_element(fuzzy_set_test, 1)
    rules = new_id3.to_rule_based_system()

    factual = get_factual_FID3(f_instance, rules)
    for exp_rule, fact_rule in zip(explanation, factual):
        for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]
        assert fact_rule.consequent == exp_rule[1]


def test_factual_mean_fdt(prepare_iris_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, _ = prepare_iris_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    alpha_factuals = _alpha_factual_avg(predicted_best_rules, None)

    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    factual = get_factual_threshold(fuzzy_element, rules, new_fdt_predict, 'mean')

    for exp_rule, fact_rule in zip(alpha_factuals, factual):
        for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_factual_robust_fdt(prepare_beer_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, all_classes = prepare_beer_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    other_classes = [cv for cv in all_classes if cv != fdt_predict]
    fdt_rob_thres = fdt.robust_threshold(fuzzy_element, other_classes)

    alpha_factuals = _alpha_factual_robust(predicted_best_rules, fdt_rob_thres)
    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    factual = get_factual_threshold(fuzzy_element, rules, new_fdt_predict, 'robust')
    for exp_rule, fact_rule in zip(alpha_factuals, factual):
        for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_factual_threshold_not_supported_fdt(prepare_iris_fdt):
    with raises(ValueError):
        fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, _ = prepare_iris_fdt

        fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
        fdt.fit(X_train, y_train)

        new_fdt = FDT_dev(fuzzy_set_df_train.keys())
        new_fdt.fit(fuzzy_set_df_train, y_train)

        fdt_predict = fdt.predict(fuzzy_element)[0]
        predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
        alpha_factuals = _alpha_factual_avg(predicted_best_rules, None)

        new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
        rules = new_fdt.to_rule_based_system()
        factual = get_factual_threshold(fuzzy_element, rules, new_fdt_predict, 'Unsupported')

        for exp_rule, fact_rule in zip(alpha_factuals, factual):
            for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
                assert exp_ante[0] == fact_ante[0]
                assert exp_ante[1] == fact_ante[1]


def test_lambda_factual_fdt(prepare_iris_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, all_classes = prepare_iris_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    lam = 0.98

    alpha_factuals = _alpha_factual_factor(predicted_best_rules, lam)
    print(alpha_factuals)
    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    factual = get_factual_difference(fuzzy_element, rules, new_fdt_predict, lam)

    for exp_rule, fact_rule in zip(alpha_factuals, factual):
        for exp_ante, fact_ante in zip(exp_rule, fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_lambda_factual_complex_fdt(prepare_beer_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, _ = prepare_beer_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    lam = 0.5

    alpha_factuals = _alpha_factual_factor(predicted_best_rules, lam)

    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    factual = get_factual_difference(fuzzy_element, rules, new_fdt_predict, lam)
    for exp_rule, fact_rule in zip(alpha_factuals, factual):
        for exp_ante, fact_ante in zip(exp_rule, fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_lambda_beta_factual_fdt(prepare_iris_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, fuzzy_element, all_classes = prepare_iris_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    fdt_predict = fdt.predict(fuzzy_element)[0]
    predicted_best_rules = fdt.explain(fuzzy_element, fdt_predict)
    lam = 0.98
    beta = 0.5

    alpha_factuals = _alpha_factual_factor_sum(predicted_best_rules, lam, beta)
    new_fdt_predict = new_fdt.predict(fuzzy_element)[0]
    rules = new_fdt.to_rule_based_system()
    factual = get_factual_difference(fuzzy_element, rules, new_fdt_predict, lam, beta)

    for exp_rule, fact_rule in zip(alpha_factuals, factual):
        for exp_ante, fact_ante in zip(exp_rule, fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]


def test_rules_fdt(prepare_iris_fdt):
    fuzzy_set_df_train, _, X_train, y_train, _, _, _, all_classes = prepare_iris_fdt

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    all_rules = set((str(tuple(rule)) for rule in fdt.get_all_rules(all_classes)))
    new_rules = new_fdt.to_rule_based_system()
    new_rule_set = set([])
    for rule in new_rules:
        new_rule_set.add(str(rule.antecedent))
    assert new_rule_set == all_rules


def test_iris_fdt(prepare_iris_fdt):
    fuzzy_set_df_train, fuzzy_set_df_test, X_train, y_train, X_test, y_test, fuzzy_element, _ = prepare_iris_fdt

    fuzzy_test = [_get_fuzzy_element(fuzzy_set_df_test, i) for i in range(len(X_test.index))]

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)
    fdt_score = fdt.score(fuzzy_set_df_test, y_test)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)
    new_fdt_score = new_fdt.score(fuzzy_test, y_test)

    np.testing.assert_almost_equal(fdt.predict(fuzzy_set_df_test), new_fdt.predict(fuzzy_test))

    assert fdt.predict(fuzzy_element) == new_fdt.predict(fuzzy_element)
    assert new_fdt_score == fdt_score
    assert new_fdt.tree_ == fdt.tree


def test_iris_min_examples_fdt(prepare_iris_fdt):
    fuzzy_set_df_train, fuzzy_set_df_test, X_train, y_train, X_test, y_test, fuzzy_element, _ = prepare_iris_fdt

    fuzzy_test = [_get_fuzzy_element(fuzzy_set_df_test, i) for i in range(len(X_test.index))]

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train, min_num_examples=50)
    fdt.fit(X_train, y_train)
    fdt_score = fdt.score(fuzzy_set_df_test, y_test)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys(), min_num_examples=50)
    new_fdt.fit(fuzzy_set_df_train, y_train)
    new_fdt_score = new_fdt.score(fuzzy_test, y_test)

    np.testing.assert_almost_equal(fdt.predict(fuzzy_set_df_test), new_fdt.predict(fuzzy_test))

    assert fdt.predict(fuzzy_element) == new_fdt.predict(fuzzy_element)
    assert new_fdt_score == fdt_score
    assert new_fdt.tree_ == fdt.tree


def test_iris_max_depth_fdt(prepare_iris_fdt):
    fuzzy_set_df_train, fuzzy_set_df_test, X_train, y_train, X_test, y_test, fuzzy_element, _ = prepare_iris_fdt

    fuzzy_test = [_get_fuzzy_element(fuzzy_set_df_test, i) for i in range(len(X_test.index))]

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train, max_depth=2)
    fdt.fit(X_train, y_train)
    fdt_score = fdt.score(fuzzy_set_df_test, y_test)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys(), max_depth=2)
    new_fdt.fit(fuzzy_set_df_train, y_train)
    new_fdt_score = new_fdt.score(fuzzy_test, y_test)

    np.testing.assert_almost_equal(fdt.predict(fuzzy_set_df_test), new_fdt.predict(fuzzy_test))

    assert fdt.predict(fuzzy_element) == new_fdt.predict(fuzzy_element)
    assert new_fdt_score == fdt_score
    assert new_fdt.tree_ == fdt.tree


def test_iris_max_match_fdt(prepare_iris_fdt):
    fuzzy_set_df_train, fuzzy_set_df_test, X_train, y_train, X_test, y_test, fuzzy_element, _ = prepare_iris_fdt

    fuzzy_test = [_get_fuzzy_element(fuzzy_set_df_test, i) for i in range(len(X_test.index))]

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train, voting='max_match')
    fdt.fit(X_train, y_train)
    fdt_score = fdt.score(fuzzy_set_df_test, y_test)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys(), voting='max_match')
    new_fdt.fit(fuzzy_set_df_train, y_train)
    new_fdt_score = new_fdt.score(fuzzy_test, y_test)

    np.testing.assert_almost_equal(fdt.predict(fuzzy_set_df_test), new_fdt.predict(fuzzy_test))

    assert fdt.predict(fuzzy_element) == new_fdt.predict(fuzzy_element)
    assert new_fdt_score == fdt_score
    assert new_fdt.tree_ == fdt.tree


def test_iris_invalid_voting_fdt(prepare_iris_fdt):
    with raises(ValueError):
        fuzzy_set_df_train, fuzzy_set_df_test, X_train, y_train, X_test, y_test, fuzzy_element, _ = prepare_iris_fdt
        FDT_dev(fuzzy_set_df_train.keys(), voting='invalid')


def test_not_instance_tree_fdt(prepare_iris_fdt):
    fuzzy_set_df_train, fuzzy_set_df_test, X_train, y_train, X_test, y_test, fuzzy_element, _ = prepare_iris_fdt
    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    assert new_fdt.tree_ != 10


def test_not_instance_tree_id3(prepare_wine):
    feature_names, X_train, X_test, y_train, y_test = prepare_wine

    new_id3 = ID3_dev(feature_names)
    new_id3.fit(X_train.values, y_train)

    assert 10 != new_id3.tree_
