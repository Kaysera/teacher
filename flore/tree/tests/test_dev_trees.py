from flore.tree import ID3, ID3_dev, FDT, FDT_dev, Rule
from pytest import fixture, raises

from sklearn import datasets
from sklearn.model_selection import train_test_split

from flore.fuzzy import get_fuzzy_triangle, get_fuzzy_set_dataframe, get_fuzzy_points
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


def _get_fuzzy_element(fuzzy_X, idx):
    element = {}
    for feat in fuzzy_X:
        element[feat] = {}
        for fuzzy_set in fuzzy_X[feat]:
            element[feat][fuzzy_set] = fuzzy_X[feat][fuzzy_set][idx]

    return element


def test_rule_equal():
    r1 = Rule(('ante'), 'conse', 0.5)
    r2 = Rule(('ante'), 'conse', 0.5)
    assert r1 == r2


def test_rule_different():
    r1 = Rule(('ante'), 'conse', 0.5)
    r2 = Rule(('ante'), 'conse', 0.3)
    assert r1 != r2


def test_rule_different_not_rule():
    r1 = Rule(('ante'), 'conse', 0.5)
    r2 = 7
    assert r1 != r2


def test_rule_matching():
    r1 = Rule((('ante1', 'ok'), ('ante2', 'nook')), 'conse', 0.5)
    fuzzy_instance = {'ante1': {'ok': 0.6, 'nook': 0.4}, 'ante2': {'ok': 0.6, 'nook': 0.4}}
    assert r1.matching(fuzzy_instance) == 0.4


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
    assert id3.exploreTreeFn(verbose=False) == rules


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
