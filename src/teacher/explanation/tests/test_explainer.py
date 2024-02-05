import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from teacher.explanation import FID3Explainer, FDTExplainer
from teacher.datasets import load_compas
from teacher.tree import Rule
from teacher.neighbors import LoreNeighborhood, NotFittedError
from .._base_explainer import BaseExplainer
from .._factual_local_explainer import FactualLocalExplainer


class MockBaseExplainer(BaseExplainer):
    """Mock Base Explainer, not intended for use"""
    def fit(self):
        """Mock fit method that does nothing"""


class MockFactualLocalExplainer(FactualLocalExplainer):
    """Mock Base Explainer, not intended for use"""
    def fit(self):
        """Mock fit method that does nothing"""
        self.exp_value = [1]
        self.target = [1]
        factual = Rule([('feat1', 'val1')], 1, 1)
        counterfactual = {('feat1', 'val2')}
        self.explanation = (factual, counterfactual)


def test_explainer_not_fitted():
    with pytest.raises(NotFittedError):
        mbe = MockBaseExplainer()
        mbe.explain()


def test_explainer_hit_not_fitted():
    with pytest.raises(NotFittedError):
        mbe = MockFactualLocalExplainer()
        mbe.hit()


def test_explainer_hit():
    mbe = MockFactualLocalExplainer()
    mbe.fit()
    assert mbe.hit() is True


def test_write_explanation():
    mbe = MockFactualLocalExplainer()
    mbe.fit()
    expected_explanation = 'The element is 1 because feat1: val1 => 1 (Weight: 1)\n'
    expected_explanation += 'Otherwise, you would need feat1 = val2'
    assert expected_explanation == mbe.write_explanation()


@pytest.mark.skip(reason="LoreNeighborhood is obsolete")
def test_FID3Explainer(set_random):
    dataset = load_compas()

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=set_random)

    idx_record2explain = 3
    instance = X_test[idx_record2explain]
    size = 300
    class_name = dataset['class_name']
    get_division = 'equal_freq'

    df_numerical_columns = [col for col in dataset['continuous'] if col != class_name]
    df_categorical_columns = [col for col in dataset['discrete'] if col != class_name]

    blackbox = RandomForestClassifier(n_estimators=20, random_state=set_random)
    blackbox.fit(X_train, y_train)
    target = blackbox.predict(instance.reshape(1, -1))

    neighborhood = LoreNeighborhood(instance, size, class_name, blackbox, dataset, X_test, idx_record2explain)
    neighborhood.fit()
    neighborhood.fuzzify(get_division,
                         sets=3,
                         class_name=class_name,
                         df_numerical_columns=df_numerical_columns,
                         df_categorical_columns=df_categorical_columns)

    explainer = FID3Explainer()
    explainer.fit(instance, target, neighborhood)

    factual, counterfactual = explainer.explain()

    expected_fact = Rule((('days_b_screening_arrest', '1.0'), ('priors_count', '1.0')), 1, 1)
    expected_cf = [Rule((('days_b_screening_arrest', '1.0'), ('priors_count', '1.0')), 1, 1)]

    assert factual == expected_fact
    assert counterfactual == expected_cf


@pytest.mark.parametrize(
    "f_method, cf_method, lam, beta, expected_fact, expected_cf",
    [
        ('m_factual', 'i_counterfactual', None, None, [Rule((('priors_count', '2'), ('two_year_recid', '1')), 0, 1.0)],
         {('priors_count', '1')}),
        ('mr_factual', 'i_counterfactual', None, None, [Rule((('priors_count', '2'), ('two_year_recid', '1')), 0, 1.0)],
         {('priors_count', '1')}),
        ('c_factual', 'i_counterfactual', 0.9, None, [Rule((('priors_count', '2'), ('two_year_recid', '1')), 0, 1.0)],
         {('priors_count', '1')}),
        ('c_factual', 'i_counterfactual', 0.9, 0.5, [Rule((('priors_count', '2'), ('two_year_recid', '1')), 0, 1.0)],
         {('priors_count', '1')}),
        ('m_factual', 'f_counterfactual', None, None, [Rule((('priors_count', '2'), ('two_year_recid', '1')), 0, 1.0)],
         {('two_year_recid', '0')}),
        ('mr_factual', 'f_counterfactual', None, None, [Rule((('priors_count', '2'), ('two_year_recid', '1')), 0, 1.0)],
         {('two_year_recid', '0')})
    ]
    )
@pytest.mark.skip(reason="LoreNeighborhood is obsolete")
def test_FDTExplainer(prepare_compas, f_method, cf_method, lam, beta, expected_fact, expected_cf):
    [instance, target, neighborhood, df_numerical_columns] = prepare_compas

    explainer = FDTExplainer()
    if f_method == 'c_factual':
        explainer.fit(instance, target, neighborhood, df_numerical_columns, f_method, cf_method, lam=lam, beta=beta)
    else:
        explainer.fit(instance, target, neighborhood, df_numerical_columns, f_method, cf_method)

    factual, counterfactual = explainer.explain()

    assert factual == expected_fact
    assert counterfactual == expected_cf


@pytest.mark.skip(reason="LoreNeighborhood is obsolete")
def test_FDTExplainer_invalid_factual(prepare_compas):
    with pytest.raises(ValueError):
        [instance, target, neighborhood, df_numerical_columns] = prepare_compas
        f_method = None
        cf_method = 'i_counterfactual'
        explainer = FDTExplainer()
        explainer.fit(instance, target, neighborhood, df_numerical_columns, f_method, cf_method)


@pytest.mark.skip(reason="LoreNeighborhood is obsolete")
def test_FDTExplainer_invalid_counterfactual(prepare_compas):
    with pytest.raises(ValueError):
        [instance, target, neighborhood, df_numerical_columns] = prepare_compas
        f_method = 'm_factual'
        cf_method = None
        explainer = FDTExplainer()
        explainer.fit(instance, target, neighborhood, df_numerical_columns, f_method, cf_method)


@pytest.mark.skip(reason="LoreNeighborhood is obsolete")
def test_FDTExplainer_no_lambda(prepare_compas):
    with pytest.raises(ValueError):
        [instance, target, neighborhood, df_numerical_columns] = prepare_compas
        f_method = 'c_factual'
        cf_method = 'i_counterfactual'
        explainer = FDTExplainer()
        explainer.fit(instance, target, neighborhood, df_numerical_columns, f_method, cf_method)
