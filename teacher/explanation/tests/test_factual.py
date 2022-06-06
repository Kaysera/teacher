import pytest
from teacher.tree import Rule
from teacher.explanation import m_factual, c_factual, mr_factual, FID3_factual


@pytest.fixture
def mock_fired_rules():

    instance = {
        'color': {
            'low': 0.5,
            'high': 0.5
        },
        'bitterness': {
            'very high': 0.8,
            'extremely high': 0.2
        },
        'strength': {
            'high': 0.47,
            'very high': 0.53
        }
    }

    class_val = 'IPA'

    rule_one = Rule([('color', 'low'), ('strength', 'high')], 'IPA', 0.93)
    rule_two = Rule([('color', 'high'), ('strength', 'high')], 'IPA', 0.65)
    rule_three = Rule([('color', 'low'), ('strength', 'very high')], 'IPA', 0.44)
    rule_four = Rule([('color', 'high'), ('strength', 'very high')], 'IPA', 0.12)
    rule_five = Rule([('color', 'high'), ('strength', 'very high')], 'Barleywine', 0.88)
    rule_six = Rule([('color', 'low'), ('strength', 'very high')], 'Barleywine', 0.28)
    rule_seven = Rule([('color', 'high'), ('strength', 'high')], 'Barleywine', 0.35)
    rule_eight = Rule([('color', 'low'), ('strength', 'high')], 'Barleywine', 0.07)

    rule_list = [rule_one, rule_two, rule_three, rule_four, rule_five, rule_six, rule_seven, rule_eight]

    return instance, rule_list, class_val


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
