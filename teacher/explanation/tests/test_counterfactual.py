from teacher.tree import ID3, Rule
from teacher.tree.tests.id3_legacy_tree import ID3_Legacy
from pytest import fixture

from sklearn.model_selection import train_test_split

from teacher.fuzzy import get_fuzzy_variables, get_fuzzy_points, dataset_membership
from teacher.datasets import load_compas
from teacher.explanation import (FID3_factual, FID3_counterfactual, i_counterfactual,
                                 f_counterfactual)

from .test_factual import (_get_fuzzy_element, _get_categorical_fuzzy,
                           _fuzzify_dataset, _get_best_rule)
import numpy as np
import random


@fixture
def set_random():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    return seed


@fixture
def prepare_dummy():
    instance = {
        'feat1': {
            'low': 0.7,
            'mid': 0.3,
            'high': 0
        },
        'feat2': {
            'low': 0,
            'mid': 0.2,
            'high': 0.8
        },
        'feat3': {
            'r': 0,
            'g': 1,
            'b': 0
        }
    }

    rule_list = [
        Rule([('feat1', 'low'), ('feat2', 'high'), ('feat3', 'g')], 1, 1),
        Rule([('feat1', 'low'), ('feat2', 'mid'), ('feat3', 'g')], 0, 1),
        Rule([('feat1', 'high'), ('feat2', 'high'), ('feat3', 'r')], 0, 1),
        Rule([('feat1', 'mid'), ('feat2', 'mid'), ('feat3', 'b')], 0, 1),
    ]

    class_val = 1

    df_numerical_columns = ['feat1', 'feat2']

    return [instance, rule_list, class_val, df_numerical_columns]


@fixture
def prepare_dummy_no_cf():
    instance = {
        'feat1': {
            'low': 0.7,
            'mid': 0.3,
            'high': 0
        },
        'feat2': {
            'low': 0,
            'mid': 0.2,
            'high': 0.8
        },
        'feat3': {
            'r': 0,
            'g': 1,
            'b': 0
        }
    }

    rule_list = [
        Rule([('feat1', 'low'), ('feat2', 'high'), ('feat3', 'g')], 1, 1),
        Rule([('feat1', 'mid'), ('feat2', 'mid'), ('feat3', 'g')], 1, 0.7),
        Rule([('feat1', 'mid'), ('feat2', 'mid'), ('feat3', 'g')], 0, 0.3),
    ]

    class_val = 1

    df_numerical_columns = ['feat1', 'feat2']

    return [instance, rule_list, class_val, df_numerical_columns]


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


def test_i_counterfactual(prepare_dummy):
    instance, rule_list, class_val, df_numerical_columns = prepare_dummy
    i_cf = i_counterfactual(instance, rule_list, class_val, df_numerical_columns)
    assert i_cf == {('feat2', 'mid')}


def test_i_counterfactual_no_cf(prepare_dummy_no_cf):
    instance, rule_list, class_val, df_numerical_columns = prepare_dummy_no_cf
    i_cf = i_counterfactual(instance, rule_list, class_val, df_numerical_columns)
    assert i_cf is None


def test_f_counterfactual(prepare_dummy):
    instance, rule_list, class_val, df_numerical_columns = prepare_dummy
    factual = [Rule([('feat1', 'low'), ('feat2', 'high'), ('feat3', 'g')], 1, 1)]
    f_cf = f_counterfactual(factual, instance, rule_list, class_val, df_numerical_columns)
    assert f_cf == {('feat2', 'mid')}


def test_f_counterfactual_no_cf(prepare_dummy_no_cf):
    instance, rule_list, class_val, df_numerical_columns = prepare_dummy_no_cf
    factual = [Rule([('feat1', 'low'), ('feat2', 'high'), ('feat3', 'g')], 1, 1)]
    f_cf = f_counterfactual(factual, instance, rule_list, class_val, df_numerical_columns)
    assert f_cf is None


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
    cf_explanation = _get_best_rule(explanation, operator)[0]
    all_rules = id3_class.exploreTreeFn(verbose=False)
    exp_cf, _ = _FID3_counterfactual(all_rules, cf_explanation)

    # CONVERTING DISCRETE VALUES TO STRING BECAUSE
    # ID3 TREE CASTS TO INT IF IT CAN
    for ante, conse in exp_cf:
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

    new_id3_pred = new_id3.predict(instance.to_numpy().reshape(1, -1))[0]
    factual = FID3_factual(f_instance, rules)
    obt_cf, _ = FID3_counterfactual(factual, [rule for rule in rules if rule.consequent != new_id3_pred])

    for exp_rule, fact_rule in zip(exp_cf, obt_cf):
        for exp_ante, fact_ante in zip(exp_rule[0], fact_rule.antecedent):
            assert exp_ante[0] == fact_ante[0]
            assert exp_ante[1] == fact_ante[1]
        assert fact_rule.consequent == exp_rule[1]
