from teacher.fuzzy.fuzzy_set import FuzzyContinuousSet, FuzzyDiscreteSet
from teacher.fuzzy.fuzzy_variable import FuzzyVariable
from teacher.tree import Rule
import pytest


@pytest.fixture
def get_fuzzy_sets():
    fuzzy_sets_height_1 = [
        FuzzyContinuousSet('low', [0, 1, 2]),
        FuzzyContinuousSet('mid', [1, 2, 3]),
        FuzzyContinuousSet('high', [2, 3, 4]),
    ]

    fuzzy_sets_height_2 = [
        FuzzyContinuousSet('low', [2.5, 3.5, 4.5]),
        FuzzyContinuousSet('mid', [3.5, 5, 7]),
        FuzzyContinuousSet('high', [5, 6, 9]),
    ]

    fuzzy_sets_color_1 = [
        FuzzyDiscreteSet('red', 'red'),
        FuzzyDiscreteSet('green', 'green'),
        FuzzyDiscreteSet('blue', 'blue'),
    ]

    fuzzy_sets_color_2 = [
        FuzzyDiscreteSet('red', 'red'),
        FuzzyDiscreteSet('green', 'green'),
        FuzzyDiscreteSet('blue', 'blue'),
        FuzzyDiscreteSet('yellow', 'yellow'),
        FuzzyDiscreteSet('cyan', 'cyan'),
        FuzzyDiscreteSet('magenta', 'magenta'),
    ]

    return fuzzy_sets_color_1, fuzzy_sets_color_2, fuzzy_sets_height_1, fuzzy_sets_height_2


@pytest.fixture
def mock_rule():
    ante = [('height', 'high'), ('color', 'red')]
    conse = 'conse'
    weight = 1
    rule = Rule(ante, conse, weight)
    return rule


def test_build_rule(mock_rule):
    ante = [('height', 'high'), ('color', 'red')]
    conse = 'conse'
    weight = 1
    assert mock_rule.antecedent == tuple(ante)
    assert mock_rule.consequent == conse
    assert mock_rule.weight == weight


def test_rule_equal(mock_rule):
    ante = [('height', 'high'), ('color', 'red')]
    conse = 'conse'
    conse_two = 'noconse'
    weight = 1
    rule_two = Rule(ante, conse, weight)
    rule_three = Rule(ante, conse_two, weight)

    assert mock_rule == rule_two
    assert mock_rule != rule_three


def test_rule_no_equal_class(mock_rule):
    assert mock_rule != 7


def test_rule_matching(mock_rule):
    instance_membership = {
        'height': {'low': 0.3, 'high': 0.7},
        'color': {'red': 1, 'blue': 0}
    }

    assert mock_rule.matching(instance_membership) == 0.7


def test_rule_matching_keyerror(mock_rule):
    instance_membership = {
        'height': {'low': 0.3, 'high': 0.7},
    }

    assert mock_rule.matching(instance_membership) == 0


def test_weighted_vote():
    instance_membership = {
        'height': {'low': 0.3, 'high': 0.7},
        'color': {'red': 1, 'blue': 0}
    }

    rule_one = Rule([('height', 'high'), ('color', 'red')], 'conse', 0.4)
    rule_two = Rule([('height', 'low'), ('color', 'red')], 'conse_two', 1)
    rule_three = Rule([('height', 'high'), ('color', 'blue')], 'conse', 0.7)
    rule_list = [rule_one, rule_two, rule_three]

    assert Rule.weighted_vote(rule_list, instance_membership) == 'conse_two'


def test_map_rule_variables_different_universe(get_fuzzy_sets):
    fuzzy_sets_color_1, fuzzy_sets_color_2, fuzzy_sets_height_1, fuzzy_sets_height_2 = get_fuzzy_sets
    with pytest.raises(ValueError):
        fuzzy_vars_1 = [
            FuzzyVariable('height', fuzzy_sets_height_1),
            FuzzyVariable('color', fuzzy_sets_color_1)
        ]

        fuzzy_vars_2 = [
            FuzzyVariable('height', fuzzy_sets_height_2),
        ]

        original_rule = Rule((('height', 'high'), ('color', 'red')), 'conse', 1)
        dest_rule = Rule((('height', 'low'), ('color', 'red')), 'conse', 1)

        assert Rule.map_rule_variables(original_rule, fuzzy_vars_1, fuzzy_vars_2) == dest_rule


def test_map_rule_variables(get_fuzzy_sets):
    fuzzy_sets_color_1, fuzzy_sets_color_2, fuzzy_sets_height_1, fuzzy_sets_height_2 = get_fuzzy_sets

    fuzzy_vars_1 = [
        FuzzyVariable('height', fuzzy_sets_height_1),
        FuzzyVariable('color', fuzzy_sets_color_1)
    ]

    fuzzy_vars_2 = [
        FuzzyVariable('height', fuzzy_sets_height_2),
        FuzzyVariable('color', fuzzy_sets_color_2)
    ]

    original_rule = Rule((('height', 'high'), ('color', 'red')), 'conse', 1)
    dest_rule = Rule((('height', 'low'), ('color', 'red')), 'conse', 1)

    assert Rule.map_rule_variables(original_rule, fuzzy_vars_1, fuzzy_vars_2) == dest_rule
