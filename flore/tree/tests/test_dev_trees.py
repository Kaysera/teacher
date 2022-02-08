from flore.tree import ID3, ID3_dev, FDT, FDT_dev

from sklearn import datasets
from sklearn.model_selection import train_test_split

from flore.fuzzy import get_fuzzy_triangle, get_fuzzy_set_dataframe, get_fuzzy_points
from flore.datasets import load_compas
from flore.explanation import get_factual_FID3
import numpy as np
import random


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


def _get_categorical_fuzzy(var):
    x = [var[k] for k in var]
    label = {i: j for i, j in enumerate(var)}
    return np.array([label[elem] for elem in np.argmax(x, axis=0)])


def _fuzzify_dataset(dataframe, fuzzy_set, fuzzify_variable):
    ndf = dataframe.copy()
    for k in fuzzy_set:
        ndf[k] = fuzzify_variable(fuzzy_set[k])
    return ndf


def test_explain_id3():
    # TODO: FINISH
    seed = 0

    random.seed(seed)
    np.random.seed(seed)

    dataset = load_compas()

    df = dataset['df']
    class_name = dataset['class_name']
    X = df.drop(class_name, axis=1)
    y = df[class_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

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
    fuzzy_points = get_fuzzy_points(df_train, 'entropy', df_numerical_columns, class_name=class_name)
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
            element[feat][fuzzy_set] = fuzzy_X[feat][fuzzy_set][idx]

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

    fuzzy_points = get_fuzzy_points(df_train, 'entropy', df_numerical_columns, class_name=class_name)
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
