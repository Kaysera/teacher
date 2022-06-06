import pytest
from teacher.tree import Rule


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


@pytest.fixture
def prepare_dummy_no_cf():
    instance = {
        'feat1': {
            'low': 0.7,
            'mid': 0.3,
            'high': 0
        },
        'feat2': {
            'low': 0,
            'mid': 0.2,
            'high': 0.8
        },
        'feat3': {
            'r': 0,
            'g': 1,
            'b': 0
        }
    }

    rule_list = [
        Rule([('feat1', 'low'), ('feat2', 'high'), ('feat3', 'g')], 1, 1),
        Rule([('feat1', 'mid'), ('feat2', 'mid'), ('feat3', 'g')], 1, 0.7),
        Rule([('feat1', 'mid'), ('feat2', 'mid'), ('feat3', 'g')], 0, 0.3),
    ]

    class_val = 1

    df_numerical_columns = ['feat1', 'feat2']

    return [instance, rule_list, class_val, df_numerical_columns]


@pytest.fixture
def prepare_dummy():
    instance = {
        'feat1': {
            'low': 0.7,
            'mid': 0.3,
            'high': 0
        },
        'feat2': {
            'low': 0,
            'mid': 0.2,
            'high': 0.8
        },
        'feat3': {
            'r': 0,
            'g': 1,
            'b': 0
        }
    }

    rule_list = [
        Rule([('feat1', 'low'), ('feat2', 'high'), ('feat3', 'g')], 1, 1),
        Rule([('feat1', 'low'), ('feat2', 'mid'), ('feat3', 'g')], 0, 1),
        Rule([('feat1', 'high'), ('feat2', 'high'), ('feat3', 'r')], 0, 1),
        Rule([('feat1', 'mid'), ('feat2', 'mid'), ('feat3', 'b')], 0, 1),
    ]

    class_val = 1

    df_numerical_columns = ['feat1', 'feat2']

    return [instance, rule_list, class_val, df_numerical_columns]
