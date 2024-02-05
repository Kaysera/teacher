from teacher.fuzzy import FuzzyContinuousSet, FuzzyDiscreteSet
import pytest


def test_continuous_intersection():
    f1 = FuzzyContinuousSet('set1', [1, 2, 3])
    f2 = FuzzyContinuousSet('set2', [2, 2, 3])
    f3 = FuzzyContinuousSet('set3', [2, 3, 4])
    f4 = FuzzyContinuousSet('set3', [3, 4, 5])

    assert f1.intersection(f2) == 1
    assert f1.intersection(f3) == 0.5
    assert f1.intersection(f4) == 0

    assert f1.intersection(f2) == f2.intersection(f1)
    assert f1.intersection(f3) == f3.intersection(f1)
    assert f1.intersection(f4) == f4.intersection(f1)


def test_discrete_intersection():
    f1 = FuzzyDiscreteSet('set1', 'val1')
    f2 = FuzzyDiscreteSet('set1', 'val1')
    f3 = FuzzyDiscreteSet('set1', 'val2')
    f4 = FuzzyDiscreteSet('set2', 'val2')

    assert f1.intersection(f2) == 1
    assert f1.intersection(f3) == 0
    assert f1.intersection(f4) == 0

    assert f1.intersection(f2) == f2.intersection(f1)
    assert f1.intersection(f3) == f3.intersection(f1)
    assert f1.intersection(f4) == f4.intersection(f1)


def test_continuous_value_error():
    with pytest.raises(ValueError):
        f1 = FuzzyContinuousSet('set1', [1, 2, 3])
        f2 = FuzzyDiscreteSet('set1', 'val1')
        f1.intersection(f2)


def test_discrete_value_error():
    with pytest.raises(ValueError):
        f1 = FuzzyContinuousSet('set1', [1, 2, 3])
        f2 = FuzzyDiscreteSet('set1', 'val1')
        f2.intersection(f1)
