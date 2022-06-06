from teacher.tree import Rule
from teacher.explanation import i_counterfactual, f_counterfactual, FID3_counterfactual


def test_i_counterfactual(mock_fired_rules):
    instance, rule_list, class_val = mock_fired_rules
    df_numerical_columns = ['color', 'strength', 'bitterness']
    assert i_counterfactual(instance, rule_list, class_val, df_numerical_columns) == {('color', 'high')}


def test_f_counterfactual(mock_fired_rules):
    instance, rule_list, class_val = mock_fired_rules
    df_numerical_columns = ['color', 'strength', 'bitterness']

    rule_one = Rule([('color', 'low'), ('strength', 'high')], 'IPA', 0.93)
    rule_two = Rule([('color', 'high'), ('strength', 'high')], 'IPA', 0.65)
    rule_three = Rule([('color', 'low'), ('strength', 'very high')], 'IPA', 0.44)
    factual = [rule_one, rule_two, rule_three]

    assert f_counterfactual(factual, instance, rule_list, class_val, df_numerical_columns) == {('color', 'high')}


def test_FID3_counterfactual():
    factual = Rule([('color', 'low'), ('strength', 'very high')], 'IPA', 0.44)

    rule_four = Rule([('color', 'low'), ('strength', 'very high')], 'Barleywine', 0.28)
    rule_five = Rule([('color', 'high'), ('strength', 'very high')], 'Barleywine', 0.88)
    rule_six = Rule([('color', 'low'), ('strength', 'very high')], 'Barleywine', 0.28)
    rule_seven = Rule([('color', 'high'), ('strength', 'high')], 'Barleywine', 0.35)
    rule_eight = Rule([('color', 'low'), ('strength', 'high')], 'Barleywine', 0.07)
    counter_rules = [rule_four, rule_five, rule_six, rule_seven, rule_eight]

    expected_counterfactual = ([Rule((('color', 'low'), ('strength', 'very high')), 'Barleywine', 0.28),
                                Rule((('color', 'low'), ('strength', 'very high')), 'Barleywine', 0.28)], 0)

    assert FID3_counterfactual(factual, counter_rules) == expected_counterfactual


def test_i_counterfactual_no_cf(prepare_dummy_no_cf):
    instance, rule_list, class_val, df_numerical_columns = prepare_dummy_no_cf
    i_cf = i_counterfactual(instance, rule_list, class_val, df_numerical_columns)
    assert i_cf is None


def test_i_counterfactual_categorical_variables(prepare_dummy):
    instance, rule_list, class_val, df_numerical_columns = prepare_dummy
    i_cf = i_counterfactual(instance, rule_list, class_val, df_numerical_columns)
    assert i_cf == {('feat2', 'mid')}
