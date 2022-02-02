import numpy as np
import pandas as pd

from flore.fuzzy import get_fuzzy_points_entropy, get_fuzzy_triangle
from flore.tree import FDT
from flore.explanation import alpha_factual_avg

from sklearn import datasets
from sklearn.model_selection import train_test_split

from functools import reduce


def test_tree():
    theory = np.array([0, 0, 3, 3, 7, 7, 9])
    practice = np.array([0, 3, 3, 9, 1, 4, 9])

    df = pd.DataFrame(([i, j, i + j >= 10] for i, j in zip(theory, practice)), columns=['theory', 'practice', 'class'])
    print(df)

    df_numerical_columns = ['theory', 'practice']
    class_name = 'class'

    X = df[df_numerical_columns]
    y = df[class_name]

    fuzzy_points = get_fuzzy_points_entropy(df, df_numerical_columns, class_name)

    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_dataframe = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_dataframe[column] = get_fuzzy_triangle(df[column].to_numpy(),
                                                         list(zip(labels, fuzzy_points[column])),
                                                         False)
    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('\n--------------------------------')
    print('Begin tree')
    print('--------------------------------')
    fdt = FDT(df_numerical_columns, fuzzy_set_dataframe)
    fdt.fit(X, y)

    print(fdt.tree)
    prediction = fdt.predict(fuzzy_set_dataframe)
    results = [False, False, False, True, False, False, True]

    assert prediction == results


def test_inference():
    df = pd.DataFrame(
        [
            [2, 5, False],
            [3, 8, False],
            [5, 4, False],
            [7, 8, True],
            [7, 3, False],
            [6, 9, True]
        ],
        columns=['theory', 'practice', 'class']
    )

    df_numerical_columns = ['theory', 'practice']
    class_name = 'class'

    fuzzy_points = {'theory': [0, 5, 10], 'practice': [0, 5, 10]}

    X = df[df_numerical_columns]
    y = df[class_name]

    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_dataframe = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_dataframe[column] = get_fuzzy_triangle(df[column].to_numpy(),
                                                         list(zip(labels, fuzzy_points[column])),
                                                         False)

    print(fuzzy_set_dataframe)
    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('\n--------------------------------')
    print('Begin tree')
    print('--------------------------------')
    fdt = FDT(df_numerical_columns, fuzzy_set_dataframe)
    fdt.fit(X, y)

    print(fdt.tree)
    score = fdt.score(fuzzy_set_dataframe, y)
    expected_score = 0.83333333

    np.testing.assert_almost_equal(score, expected_score)


def test_iris():
    iris = datasets.load_iris(as_frame=True)

    df_numerical_columns = iris.feature_names
    class_name = 'target'

    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.33,
                                                        random_state=42)

    df_train = iris.frame.loc[X_train.index]
    df_test = iris.frame.loc[X_test.index]
    print('\n----------')
    print('Getting fuzzy points')
    print('----------')

    fuzzy_points = get_fuzzy_points_entropy(df_train, df_numerical_columns, class_name)

    print('----------')
    print('Get fuzzy set train')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_train = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_train[column] = get_fuzzy_triangle(df_train[column].to_numpy(),
                                                        list(zip(labels, fuzzy_points[column])),
                                                        False)

    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Training tree')
    print('----------')
    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train)
    fdt.fit(X_train, y_train)

    print(fdt.tree)

    print('----------')
    print('Get fuzzy set test')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_test = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_test[column] = get_fuzzy_triangle(df_test[column].to_numpy(),
                                                       list(zip(labels, fuzzy_points[column])),
                                                       False)
    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Computing score')
    print('----------')
    score = fdt.score(fuzzy_set_df_test, y_test)
    expected_score = 1

    np.testing.assert_almost_equal(score, expected_score)


def test_wine():
    wine = datasets.load_wine(as_frame=True)

    df_numerical_columns = wine.feature_names
    class_name = 'target'

    X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                        wine.target,
                                                        test_size=0.33,
                                                        random_state=42)

    df_train = wine.frame.loc[X_train.index]
    df_test = wine.frame.loc[X_test.index]
    print('\n----------')
    print('Getting fuzzy points')
    print('----------')

    fuzzy_points = get_fuzzy_points_entropy(df_train, df_numerical_columns, class_name)

    print('----------')
    print('Get fuzzy set train')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_train = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_train[column] = get_fuzzy_triangle(df_train[column].to_numpy(),
                                                        list(zip(labels, fuzzy_points[column])),
                                                        False)

    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Training tree')
    print('----------')
    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train, voting='max_match')
    fdt.fit(X_train, y_train)

    print(fdt.tree)

    print('----------')
    print('Get fuzzy set test')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_test = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_test[column] = get_fuzzy_triangle(df_test[column].to_numpy(),
                                                       list(zip(labels, fuzzy_points[column])),
                                                       False)
    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Computing score')
    print('----------')
    score = fdt.score(fuzzy_set_df_test, y_test)
    expected_score = 0.9322033898305084

    np.testing.assert_almost_equal(score, expected_score)


def test_explain_all_rules():
    wine = datasets.load_wine(as_frame=True)

    df_numerical_columns = wine.feature_names
    class_name = 'target'

    X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                        wine.target,
                                                        test_size=0.33,
                                                        random_state=42)

    df_train = wine.frame.loc[X_train.index]
    df_test = wine.frame.loc[X_test.index[0:1]]
    print('\n----------')
    print('Getting fuzzy points')
    print('----------')

    fuzzy_points = get_fuzzy_points_entropy(df_train, df_numerical_columns, class_name)

    print('----------')
    print('Get fuzzy set train')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_train = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_train[column] = get_fuzzy_triangle(df_train[column].to_numpy(),
                                                        list(zip(labels, fuzzy_points[column])),
                                                        False)

    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Training tree')
    print('----------')
    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train, voting='max_match')
    fdt.fit(X_train, y_train)

    print(fdt.tree)

    print('----------')
    print('Get fuzzy set test')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_test = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_test[column] = get_fuzzy_triangle(df_test[column].to_numpy(),
                                                       list(zip(labels, fuzzy_points[column])),
                                                       False)
    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Explaining instance')
    print('----------')
    prediction = fdt.predict(fuzzy_set_df_test)[0]
    explanation = fdt.explain(fuzzy_set_df_test, prediction)
    all_rules = [(0.6156156156156157, [('flavanoids', '1.75'), ('alcohol', '14.83')]),
                 (0.6010101010101007, [('flavanoids', '1.75'), ('alcohol', '12.85'), ('proline', '750.0')]),
                 (0.3843843843843843, [('flavanoids', '5.08'), ('alcohol', '12.85')]),
                 (0.3843843843843843, [('flavanoids', '5.08'), ('alcohol', '14.83')]),
                 (0.1191969887076537, [('flavanoids', '1.75'), ('alcohol', '12.85'),
                                       ('proline', '1547.0'), ('alcalinity_of_ash', '10.6')]),
                 (0.1191969887076537, [('flavanoids', '1.75'), ('alcohol', '12.85'), ('proline', '1547.0'),
                                       ('alcalinity_of_ash', '30.0')])]
    assert explanation == all_rules


def test_explain_single_rule():
    wine = datasets.load_wine(as_frame=True)

    df_numerical_columns = wine.feature_names
    class_name = 'target'

    X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                        wine.target,
                                                        test_size=0.33,
                                                        random_state=42)

    df_train = wine.frame.loc[X_train.index]
    df_test = wine.frame.loc[X_test.index[0:1]]
    print('\n----------')
    print('Getting fuzzy points')
    print('----------')

    fuzzy_points = get_fuzzy_points_entropy(df_train, df_numerical_columns, class_name)

    print('----------')
    print('Get fuzzy set train')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_train = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_train[column] = get_fuzzy_triangle(df_train[column].to_numpy(),
                                                        list(zip(labels, fuzzy_points[column])),
                                                        False)

    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Training tree')
    print('----------')
    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train, voting='max_match')
    fdt.fit(X_train, y_train)

    print(fdt.tree)

    print('----------')
    print('Get fuzzy set test')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_test = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_test[column] = get_fuzzy_triangle(df_test[column].to_numpy(),
                                                       list(zip(labels, fuzzy_points[column])),
                                                       False)
    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Explaining instance')
    print('----------')
    prediction = fdt.predict(fuzzy_set_df_test)[0]
    explanation = fdt.explain(fuzzy_set_df_test, prediction, n_rules=1)
    rule = [(0.6156156156156157, [('flavanoids', '1.75'), ('alcohol', '14.83')])]
    assert explanation == rule


def test_get_best_rule():
    wine = datasets.load_wine(as_frame=True)

    df_numerical_columns = wine.feature_names
    class_name = 'target'

    X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                        wine.target,
                                                        test_size=0.33,
                                                        random_state=42)

    df_train = wine.frame.loc[X_train.index]
    df_test = wine.frame.loc[X_test.index[0:1]]
    print('\n----------')
    print('Getting fuzzy points')
    print('----------')

    fuzzy_points = get_fuzzy_points_entropy(df_train, df_numerical_columns, class_name)

    print('----------')
    print('Get fuzzy set train')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_train = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_train[column] = get_fuzzy_triangle(df_train[column].to_numpy(),
                                                        list(zip(labels, fuzzy_points[column])),
                                                        False)

    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Training tree')
    print('----------')
    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train, voting='max_match')
    fdt.fit(X_train, y_train)

    print(fdt.tree)

    print('----------')
    print('Get fuzzy set test')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_test = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_test[column] = get_fuzzy_triangle(df_test[column].to_numpy(),
                                                       list(zip(labels, fuzzy_points[column])),
                                                       False)
    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Obtaining best matching rule')
    print('----------')
    class_value, explanation = fdt.get_best_rule(fuzzy_set_df_test)
    expected_class_value = 0
    expected_explanation = [(0.6010101010101007, [('flavanoids', '1.75'), ('alcohol', '12.85'), ('proline', '750.0')])]
    assert class_value == expected_class_value
    assert expected_explanation == explanation


def test_get_counterfactual():
    wine = datasets.load_wine(as_frame=True)

    df_numerical_columns = wine.feature_names
    class_name = 'target'

    X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                        wine.target,
                                                        test_size=0.33,
                                                        random_state=42)

    df_train = wine.frame.loc[X_train.index]
    df_test = wine.frame.loc[X_test.index[15:16]]
    print('\n----------')
    print('Getting fuzzy points')
    print('----------')

    fuzzy_points = get_fuzzy_points_entropy(df_train, df_numerical_columns, class_name)

    print('----------')
    print('Get fuzzy set train')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_train = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_train[column] = get_fuzzy_triangle(df_train[column].to_numpy(),
                                                        list(zip(labels, fuzzy_points[column])),
                                                        False)

    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Training tree')
    print('----------')
    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train, voting='max_match')
    fdt.fit(X_train, y_train)

    print(fdt.tree)

    print('----------')
    print('Get fuzzy set test')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_test = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_test[column] = get_fuzzy_triangle(df_test[column].to_numpy(),
                                                       list(zip(labels, fuzzy_points[column])),
                                                       False)
    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Obtaining best matching rule')
    print('----------')
    # print(fuzzy_set_df_test)
    prediction = fdt.predict(fuzzy_set_df_test)[0]
    other_classes = ([x for x in np.unique(y_train) if prediction != x])
    # print(other_classes)
    print(fdt.get_counterfactual(fuzzy_set_df_test, other_classes, df_numerical_columns))


def get_fuzzy_element(fuzzy_X, idx):
    element = {}
    for feat in fuzzy_X:
        element[feat] = {}
        for fuzzy_set in fuzzy_X[feat]:
            element[feat][fuzzy_set] = [fuzzy_X[feat][fuzzy_set][idx]]

    return element


def test_get_alpha_counterfactual():
    wine = datasets.load_wine(as_frame=True)

    df_numerical_columns = wine.feature_names
    class_name = 'target'

    X_train, X_test, y_train, y_test = train_test_split(wine.data,
                                                        wine.target,
                                                        test_size=0.33,
                                                        random_state=42)

    df_train = wine.frame.loc[X_train.index]
    df_test = wine.frame.loc[X_test.index]
    print('\n----------')
    print('Getting fuzzy points')
    print('----------')

    fuzzy_points = get_fuzzy_points_entropy(df_train, df_numerical_columns, class_name)

    print('----------')
    print('Get fuzzy set train')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_train = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_train[column] = get_fuzzy_triangle(df_train[column].to_numpy(),
                                                        list(zip(labels, fuzzy_points[column])),
                                                        False)

    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Training tree')
    print('----------')
    fdt = FDT(fuzzy_set_df_train.keys(), fuzzy_set_df_train, voting='max_match')
    fdt.fit(X_train, y_train)

    print(fdt.tree)

    print('----------')
    print('Get fuzzy set test')
    print('----------')
    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_df_test = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_df_test[column] = get_fuzzy_triangle(df_test[column].to_numpy(),
                                                       list(zip(labels, fuzzy_points[column])),
                                                       False)
    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('----------')
    print('Obtaining best matching rule')
    print('----------')
    # print(fuzzy_set_df_test)
    # for idx in range(len(df_test)):
    idx = 13

    element = get_fuzzy_element(fuzzy_set_df_test, idx)
    print('----------')
    print('Element to predict')
    print(element)
    print('----------')
    prediction = fdt.predict(element)[0]
    print(f'Prediction: {prediction}')
    other_classes = ([x for x in np.unique(y_train) if prediction != x])
    print('')
    print(element)
    alpha = 0.5
    predicted_best_rules = fdt.explain(element, prediction)
    print('Best rules: ')
    for rule in predicted_best_rules:
        print(rule)
    alpha_factuals, alpha_mu = alpha_factual_avg(predicted_best_rules, alpha, debug=True)
    # if len(alpha_factuals) > 1:
    #     print(f'Element {idx} has multiple alpha factuals')
    print(f'Alpha factuals: ({alpha_mu}) ')
    for factual in alpha_factuals:
        print(factual)

    for class_val in other_classes:
        class_explanation = fdt.explain(element, class_val)
        if len(class_explanation) > 0:
            total_mu = reduce(lambda x, y: x + y[1], class_explanation, 0)
            print(f'\tClass {class_val}, N-rules: {len(class_explanation)}, Total-mu: {total_mu}')
            if total_mu > predicted_best_rules[0][1]:
                print('\tHey, Im unrobust')
            if total_mu > alpha_mu:
                print('\tHey, Im alpha unrobust')

    print('Alpha counterfactual')

    acf = fdt.get_alpha_counterfactual(fuzzy_set_df_test, other_classes, df_numerical_columns, alpha_factuals, n_cf='all', stats=True)
    for key, value in acf.items():
        print(f'Counterfactuals for class {key}:')
        for cf in value:
            print(f'\t{cf}')
