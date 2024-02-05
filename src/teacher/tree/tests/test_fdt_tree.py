import numpy as np

from teacher.fuzzy import FuzzyVariable
from teacher.fuzzy.fuzzy_set import FuzzyContinuousSet
from teacher.tree import FDT, Rule

import pytest
import random

from sklearn import datasets
from sklearn.model_selection import train_test_split

from teacher.fuzzy import get_fuzzy_variables, get_fuzzy_points
from teacher.tree.fdt_tree import TreeFDT


@pytest.fixture
def set_random():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def prepare_iris_fdt(set_random):
    iris = datasets.load_iris(as_frame=True)

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

    return [X_train, y_train, X_test, y_test, fuzzy_variables]


@pytest.fixture
def iris_rules():
    return [
        Rule((('petal length (cm)', '1.1'),), 0, 1.0),
        Rule((('petal length (cm)', '1.7'), ('petal width (cm)', '0.1')), 0, 1.0),
        Rule((('petal length (cm)', '1.7'), ('petal width (cm)', '0.5')), 0, 0.9269256089532588),
        Rule((('petal length (cm)', '1.7'), ('petal width (cm)', '0.5')), 1, 0.07307439104674127),
        Rule((('petal length (cm)', '1.7'), ('petal width (cm)', '1.3')), 1, 1.0),
        Rule((('petal length (cm)', '1.7'), ('petal width (cm)', '2.5')), 0, 0.9462365591397849),
        Rule((('petal length (cm)', '1.7'), ('petal width (cm)', '2.5')), 1, 0.053763440860215034),
        Rule((('petal length (cm)', '3.9'),), 1, 0.9623529411764704),
        Rule((('petal length (cm)', '3.9'),), 2, 0.037647058823529395),
        Rule((('petal length (cm)', '5.0'),), 1, 0.3323927765237021),
        Rule((('petal length (cm)', '5.0'),), 2, 0.6676072234762981),
        Rule((('petal length (cm)', '6.9'),), 2, 1.0)
    ]


@pytest.fixture
def mock_fuzzy_variables():
    fuzzy_sets = [FuzzyContinuousSet('fs1', [1, 2, 3])]
    fuzzy_variables = [FuzzyVariable('fv1', fuzzy_sets)]
    return fuzzy_variables


def test_build_fdt(mock_fuzzy_variables):
    fdt = FDT(mock_fuzzy_variables,
              fuzzy_threshold=0.0001,
              th=0.0001,
              max_depth=10,
              min_num_examples=1,
              prunning=True,
              t_norm=np.minimum,
              voting='agg_vote')

    assert fdt.fuzzy_variables == mock_fuzzy_variables
    assert fdt.fuzzy_threshold == 0.0001
    assert fdt.th == 0.0001
    assert fdt.max_depth == 10
    assert fdt.min_num_examples == 1
    assert fdt.prunning
    assert fdt.tree_ == TreeFDT(None, np.minimum, 'agg_vote')


def test_fit_predict_fdt(prepare_iris_fdt):
    [X_train, y_train, X_test, y_test, fuzzy_variables] = prepare_iris_fdt

    fdt = FDT(fuzzy_variables)
    fdt.fit(X_train, y_train)

    assert fdt.predict(X_test.iloc[48].to_numpy().reshape(1, -1)) == [[2]]


def test_score_fdt(prepare_iris_fdt):
    [X_train, y_train, X_test, y_test, fuzzy_variables] = prepare_iris_fdt

    fdt = FDT(fuzzy_variables)
    fdt.fit(X_train, y_train)
    score = fdt.score(X_test, y_test)

    assert score == 0.96


def test_score_max_match_fdt(prepare_iris_fdt):
    [X_train, y_train, X_test, y_test, fuzzy_variables] = prepare_iris_fdt

    fdt = FDT(fuzzy_variables, voting='max_match')
    fdt.fit(X_train, y_train)
    score = fdt.score(X_test, y_test)

    assert score == 0.88


def test_score_min_num_examples_fdt(prepare_iris_fdt):
    [X_train, y_train, X_test, y_test, fuzzy_variables] = prepare_iris_fdt

    fdt = FDT(fuzzy_variables, min_num_examples=50)
    fdt.fit(X_train, y_train)
    score = fdt.score(X_test, y_test)

    assert score == 0.96


def test_score_max_depth_fdt(prepare_iris_fdt):
    [X_train, y_train, X_test, y_test, fuzzy_variables] = prepare_iris_fdt

    fdt = FDT(fuzzy_variables, max_depth=1)
    fdt.fit(X_train, y_train)
    score = fdt.score(X_test, y_test)

    assert score == 0.96


def test_rules_fdt(prepare_iris_fdt, iris_rules):
    [X_train, y_train, X_test, y_test, fuzzy_variables] = prepare_iris_fdt

    fdt = FDT(fuzzy_variables)
    fdt.fit(X_train, y_train)

    rules = fdt.to_rule_based_system()
    assert rules == iris_rules


def test_invalid_voting_fdt(mock_fuzzy_variables):
    with pytest.raises(ValueError):
        FDT(mock_fuzzy_variables, voting='invalid')


def test_not_tree_instance_fdt(mock_fuzzy_variables):
    fdt = FDT(mock_fuzzy_variables)
    assert fdt.tree_ != 10
