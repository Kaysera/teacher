import numpy as np
from functools import partial


def get_feature_values(dataset, cat_cols):
    feature_values = {}
    num_cols = (set(dataset.columns) - set(cat_cols))

    for i, col in enumerate(dataset.columns):
        if col in num_cols:
            mean = dataset[col].mean()
            std = dataset[col].std()
            gen_function = partial(np.random.normal, loc=mean, scale=std)

        elif col in cat_cols:
            # Assuming the categorical columns are label encoded
            gen_function = partial(np.random.randint, low=0, high=len(dataset[col].unique()))

        feature_values[i] = gen_function

    return feature_values


def gen_random_individiual(feature_values, original):
    n_mutated_gens = np.random.randint(1, 4)  # Number of mutated gens between one and three
    mutated_gens = np.random.randint(len(original), size=n_mutated_gens)  # Which gens mutate
    individual = original.copy()
    for feature_idx in mutated_gens:
        gen_value = feature_values[feature_idx]
        val = gen_value()
        individual[feature_idx] = val
    return individual


def gen_random_population(feature_values, original, n, bb):
    positive = []
    negative = []

    while len(positive) < n and len(negative) < n:
        new_individual = gen_random_individiual(feature_values, original)
        prediction = bb.predict([new_individual])[0]

        if prediction == 0:
            negative += [new_individual]
        else:
            positive += [new_individual]

    print(len(positive))
    print(len(negative))

    extra_positive = []
    extra_negative = []

    while len(positive) < n:
        mut = 1
        new_individual = gen_random_individiual(feature_values, original)
        prediction = bb.predict([new_individual])[0]

        while prediction != 1:
            mut += 1
            new_individual = gen_random_individiual(feature_values, new_individual)
            prediction = bb.predict([new_individual])[0]

        positive += [new_individual]
        extra_positive += [mut]

    while len(negative) < n:
        mut = 1
        new_individual = gen_random_individiual(feature_values, original)
        prediction = bb.predict([new_individual])[0]

        while prediction != 0:
            mut += 1
            new_individual = gen_random_individiual(feature_values, new_individual)
            prediction = bb.predict([new_individual])[0]

        negative += [new_individual]
        extra_negative += [mut]

    # print(f'{len(positive)} - {np.mean(extra_positive)}')
    # print(f'{len(negative)} - {np.mean(extra_negative)}')

    population = positive + negative
    return population, np.mean(extra_positive), np.mean(extra_negative)


def random_init(feature_values):
    individual = np.zeros(len(feature_values))
    for feature_idx in feature_values:
        gen_value = feature_values[feature_idx]
        val = gen_value()
        individual[feature_idx] = val
    return individual


def informed_init(feature_values, original, prob, fitness_function=False):
    if fitness_function:
        individual = original[0].copy()
    else:
        individual = original.copy()
    for feature_idx in feature_values:
        if np.random.random() < prob:
            gen_value = feature_values[feature_idx]
            val = gen_value()
            individual[feature_idx] = val
    if fitness_function:
        return individual, fitness_function(individual)
    else:
        return individual


def uniform_crossover(first, second, prob, fitness_function):
    new_first = first[0].copy()
    new_second = second[0].copy()

    for idx in range(len(first)):
        if np.random.random() < prob:
            new_first[idx] = second[0][idx]
            new_second[idx] = first[0][idx]

    return (new_first, fitness_function(new_first)), (new_second, fitness_function(new_second))


def tournament_selection(population, k):
    tournament = []
    pop_fit = [j for i, j in population]

    for i in range(len(population)):
        candidates_idx = np.random.choice(len(population), k)
        candidates = [population[i] for i in candidates_idx]
        candidates_np = [pop_fit[i] for i in candidates_idx]
        # tourn_cand = max(candidates, key=lambda k: k[1])[0]
        tourn_cand_np = candidates[np.argmax(candidates_np)]
        tournament += [tourn_cand_np]

    return tournament


def replacement(old_population, new_individuals):
    # TODO: OPTIMIZE
    big_population = old_population + new_individuals
    # print(big_population)
    # big_population_sorted = sorted(big_population, key=lambda k: fitness_function(k)[0], reverse=True)
    # big_population_fitness = [fitness_function(k)[0] for k in big_population_sorted]
    big_pop_fitness = [j[0] for i, j in big_population]
    big_pop_ags = np.argsort(-np.array(big_pop_fitness))
    big_population_np = [big_population[i] for i in big_pop_ags]
    # big_pop_fitness_sorted = [big_pop_fitness[i] for i in big_pop_ags]
    # print(big_pop_fitness_sorted == big_population_fitness)

    # print(np.array(big_population_sorted) == big_population_np)
    return big_population_np[:len(old_population)]


def old_replacement(old_population, new_individuals, fitness_function):
    # TODO: OPTIMIZE
    big_population = old_population + new_individuals
    big_population_sorted = sorted(big_population, key=lambda k: fitness_function(k)[0], reverse=True)
    return big_population_sorted[:len(old_population)]


def genetic_algorithm(init_population, cxprob, crossprob, mutprob,
                      mut_ext_prob, feature_values, tournament_k, fitness_function, rounds):
    population = init_population

    for i in range(rounds):
        # print(f'Round {i}')
        # Selection
        # print('Selection')
        new_population = tournament_selection(population, tournament_k)
        # Crossover
        # print('Crossover')
        for i in range(len(new_population) - 1):
            if np.random.random() < crossprob:
                first, second = uniform_crossover(new_population[i], new_population[i + 1], cxprob, fitness_function)
                new_population[i] = first
                new_population[i + 1] = second
        # Mutation
        # print('Mutation')
        for i in range(len(new_population)):
            if np.random.random() < mut_ext_prob:
                mut_ind = informed_init(feature_values, new_population[i], mutprob, fitness_function)
                new_population[i] = mut_ind
        # Replacement
        # print('Replacement')
        # start = time.time()
        # old_replacement(population, new_population, fitness_function)
        # end = time.time()
        # print(f'Old replacement: {end - start}')

        new_population = replacement(population, new_population)

        # replacement(population, new_population, fitness_function)
        population = new_population

    return population
