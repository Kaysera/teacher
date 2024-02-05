
# =============================================================================
# Imports
# =============================================================================

# Standard
from math import inf

# Third party
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import _validate_vector
from scipy.stats import median_abs_deviation


# =============================================================================
# Functions
# =============================================================================

def mad_cityblock(u, v, mad):
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = abs(u - v)
    l1_diff_mad = l1_diff / mad
    return l1_diff_mad.sum()


def _continuous_distance(x, cf_list, continuous_features, metric='euclidean', X=None, agg=None):

    if metric == 'mad':
        mad = median_abs_deviation(X[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)
        dist = cdist(x.reshape(1, -1)[:, continuous_features], cf_list.reshape(1, -1)[:, continuous_features],
                     metric=_mad_cityblock)
    else:
        dist = cdist(x.reshape(1, -1)[:, continuous_features], cf_list.reshape(1, -1)[:, continuous_features],
                     metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def _categorical_distance(x, cf_list, categorical_features, metric='jaccard', agg=None):

    dist = cdist(x.reshape(1, -1)[:, categorical_features], cf_list.reshape(1, -1)[:, categorical_features],
                 metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def _mixed_distance(instance_a, instance_b, continuous, discrete, mad, ratio_cont=None, agg=None):
    nbr_features = instance_b.shape[0]
    dist_cont = _continuous_distance(instance_a, instance_b, continuous, metric='euclidean', X=None, agg=agg)
    dist_cate = _categorical_distance(instance_a, instance_b, discrete, metric='jaccard', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(continuous) / nbr_features
        ratio_categorical = len(discrete) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def _distance(instance_a, instance_b, continuous, discrete, mad):
    """Compute the distance between two instances of
    the same dataset

    Parameters
    ----------
    instance_a : array-like
        First instance
    instance_b : array-like
        Second instance
    continuous : array-like
        Indices of the continuous features
    discrete : array-like
        Indices of the discrete features
    mad : dict
        Median Absolute Distances of all the
        continuous features in the dataset, where
        the keys are the indices of the continuous features
    """
    cont = 0
    disc = 0
    diss = 0
    for i, (instance_var, cf_instance_var) in enumerate(zip(instance_a, instance_b)):
        if i in continuous:
            if abs(instance_var - cf_instance_var) == 0:
                cont += 0
            else:
                cont += abs(instance_var - cf_instance_var) / mad[i]
        else:
            disc += int(instance_var != cf_instance_var)

    # Avoid division by zero when there is no continuous or discrete variables
    if cont:
        diss += cont / len(continuous)
    if disc:
        diss += disc / len(discrete)

    return diss


DISTANCES = {
    'mixed': _mixed_distance,
    'moth': _distance
}


def _closest_instance(instance, dataset, continuous, discrete, mad, distance='mixed'):
    """Return the closest instance to a given one from a dataset

    Parameters
    ----------
    instance : array-like, 1D
        Reference instance
    dataset : array-like, 2D
        Dataset with the instances to measure
    continuous : array-like
        Indices of the continuous features
    discrete : array-like
        Indices of the discrete features
    mad : dict
        Median Absolute Distances of all the
        continuous features in the dataset, where
        the keys are the indices of the continuous features
    """

    closest_instance = []
    min_distance = inf

    for ds_instance in dataset:
        new_distance = DISTANCES[distance](instance, ds_instance, continuous, discrete, mad)

        if new_distance < min_distance and new_distance > 0:
            min_distance = new_distance
            closest_instance = ds_instance

    return closest_instance


def proximity_dissimilarity(instance, cf_instance, continuous, discrete, mad, distance='moth'):
    """Compute the proximity dissimilarity between an instance
    and the applied counterfactual instance

    Parameters
    ----------
    instance : array-like
        Original instance
    cf_instance : array-like
        Counterfactual applied instance
    continuous : array-like
        Indices of the continuous features
    discrete : array-like
        Indices of the discrete features
    mad : dict
        Median Absolute Distances of all the continuous features
        in the dataset, where the keys are the indices of the continuous features
    """
    return DISTANCES[distance](instance, cf_instance, continuous, discrete, mad)


def sparsity_dissimilarity(instance, cf_instance, distance='mismatch'):
    """Compute the sparsity dissimilarity between an instance
    and the applied counterfactual instance

    Parameters
    ----------
    instance : array-like
        Original instance
    cf_instance : array-like
        Counterfactual applied instance
    """
    if distance == 'mismatch':
        diss = 0
        for instance_var, cf_instance_var in zip(instance, cf_instance):
            diss += int(instance_var != cf_instance_var)

        return diss / len(instance)
    else:
        return cdist(instance.reshape(1, -1), cf_instance.reshape(1, -1), metric='jaccard')[0][0]


def implausibility(cf_instance, dataset, continuous, discrete, mad, distance='moth'):
    """Return the level of plausibility of a counterfactual instance
    in the dataset

    Parameters
    ----------
    cf_instance : array-like, 1D
        Reference instance
    dataset : array-like, 2D
        Dataset with the instances to measure
    continuous : array-like
        Indices of the continuous features
    discrete : array-like
        Indices of the discrete features
    mad : dict
        Median Absolute Distances of all the
        continuous features in the dataset, where
        the keys are the indices of the continuous features
    """
    closest_instance = _closest_instance(cf_instance, dataset, continuous, discrete, mad, distance)
    return DISTANCES[distance](cf_instance, closest_instance, continuous, discrete, mad)


def instability(instance,
                cf_instance,
                closest_instance,
                cf_closest_instance,
                continuous,
                discrete,
                mad,
                distance='moth'):
    """Return the level of stability of a counterfactual instance
    against the counterfactual of the closest instance to the original
    instance

    Parameters
    ----------
    instance : array-like, 1D
        Original instance
    cf_instance : array-like, 1D
        Counterfactual applied original instance
    closest_instance : array-like, 1D
        Closest instance to the original one
    cf_closest_instance : array-like, 1D
        Counterfactual applied to the closest instance
    continuous : array-like
        Indices of the continuous features
    discrete : array-like
        Indices of the discrete features
    mad : dict
        Median Absolute Distances of all the
        continuous features in the dataset, where
        the keys are the indices of the continuous features
    """
    cf_distance = DISTANCES[distance](cf_instance, cf_closest_instance, continuous, discrete, mad)
    i_distance = DISTANCES[distance](instance, closest_instance, continuous, discrete, mad)
    return cf_distance / (i_distance + 1)
