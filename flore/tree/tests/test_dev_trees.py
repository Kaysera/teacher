from flore.tree import ID3, FDT, Rule
from flore.tree.tests.fdt_legacy_tree import FDT_Legacy
from flore.tree.tests.id3_legacy_tree import ID3_Legacy
from pytest import fixture, raises

from sklearn import datasets
from sklearn.model_selection import train_test_split

from flore.fuzzy import get_fuzzy_points, get_fuzzy_variables, get_dataset_membership
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
    discrete_fuzzy_values = {col: df_train[col].unique() for col in df_categorical_columns}
    fuzzy_variables = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values)
    df_train_membership = get_dataset_membership(df_train, fuzzy_variables)
    df_test_membership = get_dataset_membership(df_test, fuzzy_variables)
    fuzzy_element = _get_fuzzy_element(df_test_membership, 27)
    all_classes = np.unique(iris.target)
    return [df_train_membership, df_test_membership, X_train, y_train, X_test, y_test, fuzzy_element, all_classes]


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
    instance_membership = {'ante1': {'ok': 0.6, 'nook': 0.4}, 'ante2': {'ok': 0.6, 'nook': 0.4}}
    assert r1.matching(instance_membership) == 0.4


def test_wine_id3(prepare_wine):
    feature_names, X_train, X_test, y_train, y_test = prepare_wine

    id3 = ID3_Legacy(feature_names, X_train.values, y_train)
    id3.fit(X_train.values, y_train)

    new_id3 = ID3(feature_names)
    new_id3.fit(X_train.values, y_train)
    assert id3.score(X_test.values, y_test) == new_id3.score(X_test.values, y_test)
    assert id3.tree == new_id3.tree_


def test_rules_id3(prepare_wine):
    feature_names, X_train, X_test, y_train, y_test = prepare_wine

    id3 = ID3_Legacy(feature_names, X_train.values, y_train)
    id3.fit(X_train.values, y_train)

    new_id3 = ID3(feature_names)
    new_id3.fit(X_train.values, y_train)

    rules = []
    new_id3.tree_._get_rules(new_id3.tree_, rules, [])
    assert id3.exploreTreeFn(verbose=False) == rules


def test_rules_fdt(prepare_iris_fdt):
    df_train_membership, _, X_train, y_train, _, _, _, all_classes = prepare_iris_fdt

    fdt = FDT_Legacy(df_train_membership.keys(), df_train_membership)
    fdt.fit(X_train, y_train)

    new_fdt = FDT(df_train_membership.keys())
    new_fdt.fit(df_train_membership, y_train.to_numpy())

    all_rules = set((str(tuple(rule)) for rule in fdt.get_all_rules(all_classes)))
    new_rules = new_fdt.to_rule_based_system()
    new_rule_set = set([])
    for rule in new_rules:
        new_rule_set.add(str(rule.antecedent))
    assert new_rule_set == all_rules


def test_iris_fdt(prepare_iris_fdt):
    df_train_membership, df_test_membership, X_train, y_train, X_test, y_test, fuzzy_element, _ = prepare_iris_fdt

    fuzzy_test = [_get_fuzzy_element(df_test_membership, i) for i in range(len(X_test.index))]

    fdt = FDT_Legacy(df_train_membership.keys(), df_train_membership)
    fdt.fit(X_train, y_train)
    fdt_score = fdt.score(df_test_membership, y_test)

    new_fdt = FDT(df_train_membership.keys())
    new_fdt.fit(df_train_membership, y_train.to_numpy())
    new_fdt_score = new_fdt.score(fuzzy_test, y_test)

    np.testing.assert_almost_equal(fdt.predict(df_test_membership), new_fdt.predict(fuzzy_test))

    assert fdt.predict(fuzzy_element) == new_fdt.predict(fuzzy_element)
    assert new_fdt_score == fdt_score
    assert new_fdt.tree_ == fdt.tree


def test_iris_min_examples_fdt(prepare_iris_fdt):
    df_train_membership, df_test_membership, X_train, y_train, X_test, y_test, fuzzy_element, _ = prepare_iris_fdt

    fuzzy_test = [_get_fuzzy_element(df_test_membership, i) for i in range(len(X_test.index))]

    fdt = FDT_Legacy(df_train_membership.keys(), df_train_membership, min_num_examples=50)
    fdt.fit(X_train, y_train)
    fdt_score = fdt.score(df_test_membership, y_test)

    new_fdt = FDT(df_train_membership.keys(), min_num_examples=50)
    new_fdt.fit(df_train_membership, y_train.to_numpy())
    new_fdt_score = new_fdt.score(fuzzy_test, y_test)

    np.testing.assert_almost_equal(fdt.predict(df_test_membership), new_fdt.predict(fuzzy_test))

    assert fdt.predict(fuzzy_element) == new_fdt.predict(fuzzy_element)
    assert new_fdt_score == fdt_score
    assert new_fdt.tree_ == fdt.tree


def test_iris_max_depth_fdt(prepare_iris_fdt):
    df_train_membership, df_test_membership, X_train, y_train, X_test, y_test, fuzzy_element, _ = prepare_iris_fdt

    fuzzy_test = [_get_fuzzy_element(df_test_membership, i) for i in range(len(X_test.index))]

    fdt = FDT_Legacy(df_train_membership.keys(), df_train_membership, max_depth=2)
    fdt.fit(X_train, y_train)
    fdt_score = fdt.score(df_test_membership, y_test)

    new_fdt = FDT(df_train_membership.keys(), max_depth=2)
    new_fdt.fit(df_train_membership, y_train.to_numpy())
    new_fdt_score = new_fdt.score(fuzzy_test, y_test)

    np.testing.assert_almost_equal(fdt.predict(df_test_membership), new_fdt.predict(fuzzy_test))

    assert fdt.predict(fuzzy_element) == new_fdt.predict(fuzzy_element)
    assert new_fdt_score == fdt_score
    assert new_fdt.tree_ == fdt.tree


def test_iris_max_match_fdt(prepare_iris_fdt):
    df_train_membership, df_test_membership, X_train, y_train, X_test, y_test, fuzzy_element, _ = prepare_iris_fdt

    fuzzy_test = [_get_fuzzy_element(df_test_membership, i) for i in range(len(X_test.index))]

    fdt = FDT_Legacy(df_train_membership.keys(), df_train_membership, voting='max_match')
    fdt.fit(X_train, y_train)
    fdt_score = fdt.score(df_test_membership, y_test)

    new_fdt = FDT(df_train_membership.keys(), voting='max_match')
    new_fdt.fit(df_train_membership, y_train.to_numpy())
    new_fdt_score = new_fdt.score(fuzzy_test, y_test)

    np.testing.assert_almost_equal(fdt.predict(df_test_membership), new_fdt.predict(fuzzy_test))

    assert fdt.predict(fuzzy_element) == new_fdt.predict(fuzzy_element)
    assert new_fdt_score == fdt_score
    assert new_fdt.tree_ == fdt.tree


def test_iris_invalid_voting_fdt(prepare_iris_fdt):
    with raises(ValueError):
        df_train_membership, df_test_membership, X_train, y_train, X_test, y_test, fuzzy_element, _ = prepare_iris_fdt
        FDT(df_train_membership.keys(), voting='invalid')


def test_not_instance_tree_fdt(prepare_iris_fdt):
    df_train_membership, df_test_membership, X_train, y_train, X_test, y_test, fuzzy_element, _ = prepare_iris_fdt
    new_fdt = FDT(df_train_membership.keys())
    assert new_fdt.tree_ != 10


def test_not_instance_tree_id3(prepare_wine):
    feature_names, X_train, X_test, y_train, y_test = prepare_wine

    new_id3 = ID3(feature_names)
    new_id3.fit(X_train.values, y_train)

    assert 10 != new_id3.tree_
