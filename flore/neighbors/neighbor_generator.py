from .gpdatagenerator import *
from .genetic_algorithm import *
from flore.utils import *
from functools import partial

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import CondensedNearestNeighbour


def genetic_neighborhood_old(dfZ, x, blackbox, dataset):
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    class_name = dataset['class_name']
    idx_features = dataset['idx_features']
    feature_values = dataset['feature_values']

    discrete_no_class = list(discrete)
    discrete_no_class.remove(class_name)

    def distance_function(x0, x1, discrete, continuous, class_name):
        return mixed_distance(x0, x1, discrete, continuous, class_name,
                              ddist=simple_match_distance,
                              cdist=normalized_euclidean_distance)

    Z = generate_data(x, feature_values, blackbox, discrete_no_class, continuous, class_name, idx_features,
                      distance_function, neigtype={'ss': 0.5, 'sd': 0.5}, population_size=1000, halloffame_ratio=0.1,
                      alpha1=0.5, alpha2=0.5, eta1=1.0, eta2=0.0,  tournsize=3, cxpb=0.5, mutpb=0.2, ngen=10)
    dfZ = build_df2explain(blackbox, Z, dataset)
    return dfZ, Z


def gen_initial_population(feat_values, original, perturbation_prob, random_size, informed_size):
    initial_population = []
    for i in range(random_size):
        ind = random_init(feat_values)
        initial_population += [ind]

    for i in range(informed_size):
        ind = informed_init(feat_values, original, perturbation_prob)
        initial_population += [ind]

    return initial_population


def genetic_neighborhood_flore(dfZ, x, blackbox, dataset, params):
    feat_values = dataset['feat_values']
    perturbation_prob = params['perturbation_prob']
    cxprob = params['cxprob']
    crossprob = params['crossprob']
    mutprob = params['mutprob']
    mut_ext_prob = params['mut_ext_prob']
    tournament_k = params['tournament_k']
    rounds = params['rounds']
    random_size = params['random_size']
    informed_size = params['informed_size']

    discrete = dataset['discrete']
    continuous = dataset['continuous']
    class_name = dataset['class_name']
    idx_features = dataset['idx_features']

    discrete_no_class = list(discrete)
    discrete_no_class.remove(class_name)

    def distance_function(x0, x1, discrete, continuous, class_name):
        return mixed_distance(x0, x1, discrete, continuous, class_name,
                              ddist=simple_match_distance,
                              cdist=normalized_euclidean_distance)

    gd = partial(distance_function, discrete=discrete, continuous=continuous, class_name=class_name)


    initial_population = gen_initial_population(feat_values, x, perturbation_prob, random_size, informed_size)
    fit_sso = partial(fitness_sso, bb=blackbox, alpha1=0.5, alpha2=0.5, eta=0.3, discrete=discrete_no_class, continuous=continuous, class_name=class_name, idx_features=idx_features, distance_function=distance_function, x1=x)
    fit_sdo = partial(fitness_sdo, bb=blackbox, alpha1=0.5, alpha2=0.5, eta=0.3, discrete=discrete_no_class, continuous=continuous, class_name=class_name, idx_features=idx_features, distance_function=distance_function, x1=x)

    sso_pop = genetic_algorithm(initial_population, cxprob, crossprob, mutprob, mut_ext_prob, feat_values, tournament_k, fit_sso, rounds)
    sdo_pop = genetic_algorithm(initial_population, cxprob, crossprob, mutprob, mut_ext_prob, feat_values, tournament_k, fit_sdo, rounds)
    
    Z = sso_pop + sdo_pop
    Z = np.array(list(list(zip(*Z))[0]))

    mean_gd, std_gd = genetic_distance(Z, gd, idx_features)
    
    # print(mean_gd, std_gd)
    zy = blackbox.predict(Z)

    if len(np.unique(zy)) == 1:
        # print('qui')
        label_encoder = dataset['label_encoder']
        dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
        neig_indexes = get_closest_diffoutcome(dfZ, dfx, discrete, continuous, class_name,
                                               blackbox, label_encoder, distance_function, k=100)
        Zn, _ = label_encode(dfZ, discrete, label_encoder)
        Zn = Zn.iloc[neig_indexes, Z.columns != class_name].values
        Z = np.concatenate((Z, Zn), axis=0)
    # print(np.unique(zy, return_counts=True))


    dfZ = build_df2explain(blackbox, Z, dataset)
    return dfZ, Z, mean_gd, std_gd, len(dfZ.drop_duplicates())

def gen_random_individuals(dfZ, x, blackbox, dataset):
    feat_values = dataset['feat_values']
    
    Z, pos_mut, neg_mut = gen_random_population(feat_values, x, 500, blackbox)
    Z = np.array(Z)
    
    dfZ = build_df2explain(blackbox, Z, dataset)
    return dfZ, Z, pos_mut, neg_mut


def genetic_distance(Z, distance, idx_features):
    dists = []
    for x0 in Z:
        d = 0
        for x1 in Z:
            x0d = {idx_features[i]: val for i, val in enumerate(x0)}
            x1d = {idx_features[i]: val for i, val in enumerate(x1)}
            d += distance(x0d, x1d)
        d /= len(Z)
        dists += [d]
    return np.mean(dists), np.std(dists)


def genetic_neighborhood(dfZ, x, blackbox, dataset):
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    class_name = dataset['class_name']
    idx_features = dataset['idx_features']
    feature_values = dataset['feature_values']

    discrete_no_class = list(discrete)
    discrete_no_class.remove(class_name)

    def distance_function(x0, x1, discrete, continuous, class_name):
        return mixed_distance(x0, x1, discrete, continuous, class_name,
                              ddist=simple_match_distance,
                              cdist=normalized_euclidean_distance)

    Z = generate_data(x, feature_values, blackbox, discrete_no_class, continuous, class_name, idx_features,
                      distance_function, neigtype={'ss': 0.5, 'sd': 0.5}, population_size=1000, halloffame_ratio=0.1,
                      alpha1=0.5, alpha2=0.5, eta1=1.0, eta2=0.0,  tournsize=3, cxpb=0.5, mutpb=0.2, ngen=10)

    zy = blackbox.predict(Z)
    # print(np.unique(zy, return_counts=True))
    if len(np.unique(zy)) == 1:
        # print('qui')
        label_encoder = dataset['label_encoder']
        dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
        neig_indexes = get_closest_diffoutcome(dfZ, dfx, discrete, continuous, class_name,
                                               blackbox, label_encoder, distance_function, k=100)
        Zn, _ = label_encode(dfZ, discrete, label_encoder)
        Zn = Zn.iloc[neig_indexes, Z.columns != class_name].values
        Z = np.concatenate((Z, Zn), axis=0)

    dfZ = build_df2explain(blackbox, Z, dataset)
    return dfZ, Z




def real_data(dfZ, x, blackbox, dataset):
    discrete = dataset['discrete']
    label_encoder = dataset['label_encoder']
    class_name = dataset['class_name']

    dfZ = dfZ
    Z, _ = label_encode(dfZ, discrete, label_encoder)
    Z = Z.iloc[:, Z.columns != class_name].values
    dfZ = build_df2explain(blackbox, Z, dataset)

    return dfZ, Z


def closed_real_data(dfZ, x, blackbox, dataset):
    discrete = dataset['discrete']
    label_encoder = dataset['label_encoder']
    class_name = dataset['class_name']
    continuous = dataset['continuous']

    def distance_function(x0, x1, discrete, continuous, class_name):
        return mixed_distance(x0, x1, discrete, continuous, class_name,
                              ddist=simple_match_distance,
                              cdist=normalized_euclidean_distance)

    dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
    neig_indexes = get_closest_diffoutcome(dfZ, dfx, discrete, continuous, class_name,
                                           blackbox, label_encoder, distance_function, k=100)

    dfZ = dfZ
    Z, _ = label_encode(dfZ, discrete, label_encoder)
    Z = Z.iloc[neig_indexes, Z.columns != class_name].values
    dfZ = build_df2explain(blackbox, Z, dataset)

    return dfZ, Z


def random_neighborhood(dfZ, x, blackbox, dataset, stratified=True):
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    label_encoder = dataset['label_encoder']
    class_name = dataset['class_name']
    columns = dataset['columns']
    features_type = dataset['features_type']

    if stratified:

        def distance_function(x0, x1, discrete, continuous, class_name):
            return mixed_distance(x0, x1, discrete, continuous, class_name,
                                  ddist=simple_match_distance,
                                  cdist=normalized_euclidean_distance)

        dfx = build_df2explain(blackbox, x.reshape(1, -1), dataset).to_dict('records')[0]
        neig_indexes = get_closest_diffoutcome(dfZ, dfx, discrete, continuous, class_name,
                                               blackbox, label_encoder, distance_function, k=100)

        Z, _ = label_encode(dfZ, discrete, label_encoder)
        Z = Z.iloc[neig_indexes, Z.columns != class_name].values
        Z = generate_random_data(Z, class_name, columns, discrete, continuous, features_type, size=1000, uniform=True)
        dfZ = build_df2explain(blackbox, Z, dataset)

        return dfZ, Z

    else:

        Z, _ = label_encode(dfZ, discrete, label_encoder)
        Z = Z.iloc[:, Z.columns != class_name].values
        Z = generate_random_data(Z, class_name, columns, discrete, continuous, features_type, size=1000, uniform=True)
        dfZ = build_df2explain(blackbox, Z, dataset)

        return dfZ, Z


def generate_random_data(X, class_name, columns, discrete, continuous, features_type, size=1000, uniform=True):
    if isinstance(X, pd.DataFrame):
        X = X.values
    X1 = list()
    columns1 = list(columns)
    columns1.remove(class_name)
    for i, col in enumerate(columns1):
        values = X[:, i]
        diff_values = np.unique(values)
        prob_values = [1.0 * list(values).count(val) / len(values) for val in diff_values]
        if col in discrete:
            if uniform:
                new_values = np.random.choice(diff_values, size)
            else:
                new_values = np.random.choice(diff_values, size, prob_values)
        elif col in continuous:
            mu = np.mean(values)
            sigma = np.std(values)
            if sigma <= 0.0:
                new_values = np.array([values[0]] * size)
            else:
                new_values = np.random.normal(mu, sigma, size)
        if features_type[col] == 'integer':
            new_values = new_values.astype(int)
        X1.append(new_values)
    X1 = np.concatenate((X, np.column_stack(X1)), axis=0).tolist()
    if isinstance(X, pd.DataFrame):
        X1 = pd.DataFrame(data=X1, columns=columns1)
    return X1


def random_oversampling(dfZ, x, blackbox, dataset):
    dfZ, Z = random_neighborhood(dfZ, x, blackbox, dataset)
    y = blackbox.predict(Z)

    oversampler = RandomOverSampler()
    Z, _ = oversampler.fit_sample(Z, y)
    dfZ = build_df2explain(blackbox, Z, dataset)
    return dfZ, Z


def random_instance_selection(dfZ, x, blackbox, dataset):
    dfZ1, Z = random_neighborhood(dfZ, x, blackbox, dataset)
    y = blackbox.predict(Z)

    cnn = CondensedNearestNeighbour(return_indices=True)
    Z, _, _ = cnn.fit_sample(Z, y)
    dfZ = build_df2explain(blackbox, Z, dataset)
    return dfZ, Z

