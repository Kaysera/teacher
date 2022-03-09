from teacher.neighbors import (genetic_algorithm, get_feature_values, random_init, informed_init, uniform_crossover,
                               tournament_selection, replacement)
import pandas as pd
import numpy as np
from random import seed


def test_get_feature_values():
    np.random.seed(42)
    test_df = pd.DataFrame(
        [
            [5, 'a', 1.1, 'j'],
            [10, 'b', 2.2, 'k'],
            [15, 'c', 3.3, 'l']
        ],
        columns=['continuous', 'discrete', 'float', 'second_discrete']
    )

    feat_values = get_feature_values(test_df)
    assert feat_values[0]() == 12.483570765056164
    assert feat_values[1]() == 'c'
    assert feat_values[2]() == 2.0479092687116967
    assert feat_values[3]() == 'j'


def test_random_individual():
    np.random.seed(42)
    seed(42)
    test_df = pd.DataFrame(
        [
            [5, 'a', 1.1, 'j'],
            [10, 'b', 2.2, 'k'],
            [15, 'c', 3.3, 'l']
        ],
        columns=['continuous', 'discrete', 'float', 'second_discrete']
    )

    feat_values = get_feature_values(test_df)
    ind = random_init(feat_values)
    assert ind == [12.483570765056164, 'c', 2.0479092687116967, 'j']


def test_informed_individual():
    np.random.seed(42)
    test_df = pd.DataFrame(
        [
            [5, 'a', 1.1, 'j'],
            [10, 'b', 2.2, 'k'],
            [15, 'c', 3.3, 'l']
        ],
        columns=['continuous', 'discrete', 'float', 'second_discrete']
    )

    original = [7, 'a', 1.75, 'k']

    feat_values = get_feature_values(test_df)
    ind = informed_init(feat_values, original, 0.4)
    assert ind == [4.440599409765397, 'c', 1.75, 'l']


def test_uniform_crossover():
    np.random.seed(42)
    first = [7, 'a', 1.75, 'k']
    second = [10, 'b', 2.2, 'j']
    new_first, new_second = uniform_crossover(first, second, 0.5)
    assert new_first == [10, 'a', 1.75, 'k']
    assert new_second == [7, 'b', 2.2, 'j']


def test_tournament_selection():
    np.random.seed(42)
    population = [
        [5, 'a', 1.1, 'j'],
        [10, 'b', 2.2, 'k'],
        [15, 'c', 3.3, 'l'],
        [7, 'a', 1.75, 'k'],
        [10, 'b', 2.2, 'j']
    ]

    def fitness(x):
        val_first = {'a': 1, 'b': 2, 'c': 3}
        val_second = {'j': 1, 'k': 2, 'l': 3}

        return x[0] + val_first[x[1]] + x[2] + val_second[x[3]]

    expected_tournament = [[15, 'c', 3.3, 'l'], [10, 'b', 2.2, 'k'],
                           [15, 'c', 3.3, 'l'], [15, 'c', 3.3, 'l'], [10, 'b', 2.2, 'k']]

    assert tournament_selection(population, 3, fitness) == expected_tournament


def test_replacement():
    np.random.seed(42)
    population = [
        [5, 'a', 1.1, 'j'],
        [10, 'b', 2.2, 'k'],
        [15, 'c', 3.3, 'l'],
        [7, 'a', 1.75, 'k'],
        [10, 'b', 2.2, 'j']
    ]

    def fitness(x):
        val_first = {'a': 1, 'b': 2, 'c': 3}
        val_second = {'j': 1, 'k': 2, 'l': 3}

        return x[0] + val_first[x[1]] + x[2] + val_second[x[3]]

    tournament = tournament_selection(population, 3, fitness)
    new_population = replacement(population, tournament, fitness)
    expected_new_pop = [[15, 'c', 3.3, 'l'], [15, 'c', 3.3, 'l'],
                        [15, 'c', 3.3, 'l'], [15, 'c', 3.3, 'l'], [10, 'b', 2.2, 'k']]
    assert new_population == expected_new_pop


def test_genetic_algorithm():
    np.random.seed(42)
    test_df = pd.DataFrame(
        [
            [5, 'a', 1.1, 'j'],
            [10, 'b', 2.2, 'k'],
            [15, 'c', 3.3, 'l']
        ],
        columns=['continuous', 'discrete', 'float', 'second_discrete']
    )

    original = [7, 'a', 1.75, 'k']
    feat_values = get_feature_values(test_df)

    perturbation_prob = 0.4
    cxprob = 0.3
    crossprob = 0.3
    mutprob = 0.4
    mut_ext_prob = 0.5
    tournament_k = 5
    rounds = 10

    initial_population = []
    for i in range(10):
        ind = random_init(feat_values)
        initial_population += [ind]

    for i in range(10):
        ind = informed_init(feat_values, original, perturbation_prob)
        initial_population += [ind]

    def fitness(x):
        val_first = {'a': 1, 'b': 2, 'c': 3}
        val_second = {'j': 1, 'k': 2, 'l': 3}

        return x[0] + val_first[x[1]] + x[2] + val_second[x[3]]

    final_pop = genetic_algorithm(initial_population, cxprob, crossprob,
                                  mutprob, mut_ext_prob, feat_values, tournament_k, fitness, rounds)

    expected_final_pop = [
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l'],
        [22.361397311363298, 'c', 4.199821568757226, 'l']
        ]

    assert final_pop == expected_final_pop
