from flore.tree import ID3, ID3_dev

from sklearn import datasets
from sklearn.model_selection import train_test_split


def test_iris_id3():
    iris = datasets.load_wine(as_frame=True)

    df_numerical_columns = iris.feature_names
    print(df_numerical_columns)

    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.33,
                                                        random_state=42)

    id3 = ID3(iris.feature_names, X_train.values, y_train)
    id3.fit(X_train.values, y_train)

    new_id3 = ID3_dev(iris.feature_names)
    new_id3.fit(X_train.values, y_train)

    assert id3.score(X_test.values, y_test) == new_id3.score(X_test.values, y_test)
    assert id3.tree == new_id3.tree_
