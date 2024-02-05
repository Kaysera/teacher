from teacher.tree import Rule
from teacher.explanation import m_factual, c_factual, mr_factual, FID3_factual


def test_m_factual(mock_fired_rules):
    instance, rule_list, class_val = mock_fired_rules
    rule_one = Rule([('color', 'low'), ('strength', 'high')], 'IPA', 0.93)
    rule_two = Rule([('color', 'high'), ('strength', 'high')], 'IPA', 0.65)
    expected_factual = [rule_one, rule_two]
    assert m_factual(instance, rule_list, class_val) == expected_factual


def test_c_factual(mock_fired_rules):
    instance, rule_list, class_val = mock_fired_rules
    rule_one = Rule([('color', 'low'), ('strength', 'high')], 'IPA', 0.93)
    expected_factual = [rule_one]
    assert c_factual(instance, rule_list, class_val, 0.1) == expected_factual


def test_c_beta_factual(mock_fired_rules):
    instance, rule_list, class_val = mock_fired_rules
    rule_one = Rule([('color', 'low'), ('strength', 'high')], 'IPA', 0.93)
    rule_two = Rule([('color', 'high'), ('strength', 'high')], 'IPA', 0.65)
    expected_factual = [rule_one, rule_two]
    assert c_factual(instance, rule_list, class_val, 0.1, 0.51) == expected_factual


def test_mr_factual(mock_fired_rules):
    instance, rule_list, class_val = mock_fired_rules
    rule_one = Rule([('color', 'low'), ('strength', 'high')], 'IPA', 0.93)
    rule_two = Rule([('color', 'high'), ('strength', 'high')], 'IPA', 0.65)
    rule_three = Rule([('color', 'low'), ('strength', 'very high')], 'IPA', 0.44)
    expected_factual = [rule_one, rule_two, rule_three]
    assert mr_factual(instance, rule_list, class_val) == expected_factual


def test_FID3_factual(mock_fired_rules):
    instance, rule_list, class_val = mock_fired_rules
    rule_three = Rule([('color', 'low'), ('strength', 'very high')], 'IPA', 0.44)
    expected_factual = rule_three
    assert FID3_factual(instance, rule_list) == expected_factual
