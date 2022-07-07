
# =============================================================================
# Imports
# =============================================================================

# Standard
from math import inf

# =============================================================================
# Functions
# =============================================================================

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
    for i, (instance_var, cf_instance_var) in enumerate(zip(instance_a, instance_b)):
        if i in continuous:
            if abs(instance_var - cf_instance_var) == 0:
                cont += 0
            else:
                cont += abs(instance_var - cf_instance_var) / mad[i]
        else:
            disc += int(instance_var != cf_instance_var)
    
    diss = 1 / len(continuous) * cont + 1 / len(discrete) * disc

    return diss


def _closest_instance(instance, dataset, continuous, discrete, mad):
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
        new_distance = _distance(instance, ds_instance, continuous, discrete, mad)

        if new_distance < min_distance and new_distance > 0:
            min_distance = new_distance
            closest_instance = ds_instance
    
    return closest_instance


def proximity_dissimilarity(instance, cf_instance, continuous, discrete, mad):
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
    return _distance(instance, cf_instance, continuous, discrete, mad)


def sparsity_dissimilarity(instance, cf_instance):
    """Compute the sparsity dissimilarity between an instance
    and the applied counterfactual instance

    Parameters
    ----------
    instance : array-like
        Original instance
    cf_instance : array-like
        Counterfactual applied instance
    """
    diss = 0
    for instance_var, cf_instance_var in zip(instance, cf_instance):
        diss += int(instance_var != cf_instance_var)
    
    return diss / len(instance)


def implausibility(cf_instance, dataset, continuous, discrete, mad):
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
    closest_instance =  _closest_instance(cf_instance, dataset, continuous, discrete, mad)
    return _distance(cf_instance, closest_instance, continuous, discrete, mad)


def instability(instance, cf_instance, closest_instance, cf_closest_instance, continuous, discrete, mad):
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

    return _distance(cf_instance, cf_closest_instance, continuous, discrete, mad) / (_distance(instance, closest_instance, continuous, discrete, mad) + 1)

