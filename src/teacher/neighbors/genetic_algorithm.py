import numpy as np
from functools import partial


def get_feature_values(dataset):
    feature_values = {}
    num_cols = set(dataset._get_numeric_data().columns)
    cat_cols = (set(dataset.columns) - num_cols)

    for i, col in enumerate(dataset.columns):
        if col in num_cols:
            mean = dataset[col].mean()
            std = dataset[col].std()
            gen_function = partial(np.random.normal, loc=mean, scale=std)

        elif col in cat_cols:
            gen_function = partial(np.random.choice, dataset[col].unique())

        feature_values[i] = gen_function

    return feature_values


def random_init(feature_values):
    individual = list()
    for feature_idx in feature_values:
        gen_value = feature_values[feature_idx]
        val = gen_value()
        individual.append(val)
    return individual


def informed_init(feature_values, original, prob):
    individual = original.copy()
    for feature_idx in feature_values:
        if np.random.random() < prob:
            gen_value = feature_values[feature_idx]
            val = gen_value()
            individual[feature_idx] = val
    return individual


def uniform_crossover(first, second, prob):
    new_first = first.copy()
    new_second = second.copy()

    for idx in range(len(first)):
        if np.random.random() < prob:
            new_first[idx] = second[idx]
            new_second[idx] = first[idx]

    return new_first, new_second


def tournament_selection(population, k, fitness_function):
    tournament = []
    for i in range(len(population)):
        candidates_idx = np.random.choice(len(population), k)
        candidates = [(population[i], fitness_function(population[i])) for i in candidates_idx]
        tournament += [max(candidates, key=lambda k: k[1])[0]]

    return tournament


def replacement(old_population, new_individuals, fitness_function):
    big_population = old_population + new_individuals
    big_population = sorted(big_population, key=lambda k: fitness_function(k), reverse=True)
    return big_population[:len(old_population)]


def genetic_algorithm(init_population, cx_prob, cross_prob, mut_prob,
                      mut_ext_prob, feature_values, tournament_k, fitness_function, rounds):
    population = init_population
    for i in range(rounds):
        # Selection
        new_population = tournament_selection(population, tournament_k, fitness_function)

        # Crossover
        for i in range(len(new_population) - 1):
            if np.random.random() < cross_prob:
                first, second = uniform_crossover(new_population[i], new_population[i+1], cx_prob)
                new_population[i] = first
                new_population[i + 1] = second

        # Mutation
        for i in range(len(new_population)):
            if np.random.random() < mut_ext_prob:
                mut_ind = informed_init(feature_values, new_population[i], mut_prob)
                new_population[i] = mut_ind

        # Replacement
        new_population = replacement(population, new_population, fitness_function)
        population = new_population

    return population
