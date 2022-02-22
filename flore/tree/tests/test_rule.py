from flore.fuzzy.fuzzy_set import FuzzyContinuousSet, FuzzyDiscreteSet
from flore.fuzzy.fuzzy_variable import FuzzyVariable
from flore.tree import Rule


def test_map_rule_variables():
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
