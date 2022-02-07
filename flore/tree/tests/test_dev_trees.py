from flore.tree import ID3, ID3_dev, FDT, FDT_dev

from sklearn import datasets
from sklearn.model_selection import train_test_split

from flore.fuzzy import get_fuzzy_points_entropy, get_fuzzy_triangle, get_fuzzy_set_dataframe
import numpy as np


def test_wine_id3():
    wine = datasets.load_wine(as_frame=True)

    df_numerical_columns = wine.feature_names
    print(df_numerical_columns)

    X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                        wine.target,
                                                        test_size=0.33,
                                                        random_state=42)

    id3 = ID3(wine.feature_names, X_train.values, y_train)
    id3.fit(X_train.values, y_train)

    new_id3 = ID3_dev(wine.feature_names)
    new_id3.fit(X_train.values, y_train)

    assert id3.score(X_test.values, y_test) == new_id3.score(X_test.values, y_test)
    assert id3.tree == new_id3.tree_


def test_rules_id3():
    wine = datasets.load_wine(as_frame=True)

    df_numerical_columns = wine.feature_names
    print(df_numerical_columns)

    X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                        wine.target,
                                                        test_size=0.33,
                                                        random_state=42)

    id3 = ID3(wine.feature_names, X_train.values, y_train)
    id3.fit(X_train.values, y_train)

    new_id3 = ID3_dev(wine.feature_names)
    new_id3.fit(X_train.values, y_train)

    rules = []
    new_id3.tree_._get_rules(new_id3.tree_, rules, [])
    assert id3.exploreTreeFn() == rules


def test_rules_fdt():
    iris = datasets.load_iris(as_frame=True)

    df_numerical_columns = iris.feature_names
    df_categorical_columns = []
    class_name = 'target'

    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.33,
                                                        random_state=42)

    df_train = iris.frame.loc[X_train.index]

    fuzzy_points = get_fuzzy_points_entropy(df_train, df_numerical_columns, class_name)
    fuzzy_set_df_train = get_fuzzy_set_dataframe(df_train, get_fuzzy_triangle, fuzzy_points,
                                                 df_numerical_columns, df_categorical_columns)

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    all_rules = set([str(rule) for rule in fdt.get_all_rules(np.unique(iris.target))])
    new_rules = new_fdt.to_rule_based_system()
    new_rule_set = set([])
    for rule in new_rules:
        new_rule_set.add(str(rule.antecedent))
    assert new_rule_set == all_rules


def _get_fuzzy_element(fuzzy_X, idx):
    element = {}
    for feat in fuzzy_X:
        element[feat] = {}
        for fuzzy_set in fuzzy_X[feat]:
            element[feat][fuzzy_set] = [fuzzy_X[feat][fuzzy_set][idx]]

    return element


def test_iris_fdt():
    iris = datasets.load_iris(as_frame=True)

    df_numerical_columns = iris.feature_names
    df_categorical_columns = []
    class_name = 'target'

    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.33,
                                                        random_state=42)

    df_train = iris.frame.loc[X_train.index]
    df_test = iris.frame.loc[X_test.index]

    fuzzy_points = get_fuzzy_points_entropy(df_train, df_numerical_columns, class_name)
    fuzzy_set_df_train = get_fuzzy_set_dataframe(df_train, get_fuzzy_triangle, fuzzy_points,
                                                 df_numerical_columns, df_categorical_columns)
    fuzzy_set_df_test = get_fuzzy_set_dataframe(df_test, get_fuzzy_triangle, fuzzy_points,
                                                df_numerical_columns, df_categorical_columns)
    fuzzy_test = [_get_fuzzy_element(fuzzy_set_df_test, i) for i in range(len(X_test.index))]

    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    fdt_score = fdt.score(fuzzy_set_df_test, y_test)

    new_fdt = FDT_dev(fuzzy_set_df_train.keys())
    new_fdt.fit(fuzzy_set_df_train, y_train)

    new_fdt_score = new_fdt.score(fuzzy_test, y_test)
    np.testing.assert_almost_equal(fdt.predict(fuzzy_set_df_test), new_fdt.predict(fuzzy_test))
    fuzzy_element = _get_fuzzy_element(fuzzy_set_df_test, 1)
    assert fdt.predict(fuzzy_element) == new_fdt.predict(fuzzy_element)
    assert new_fdt_score == fdt_score
