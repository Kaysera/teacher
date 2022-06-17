# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC, abstractmethod

# Local application
from ._exceptions import NotFittedError

# =============================================================================
# Classes
# =============================================================================


class BaseNeighborhood(ABC):
    """
    Base abstract neighborhood that provides the basic operations
    for all the other neighborhoods.
    """

    def __init__(self, instance, size, class_name, bb):
        """
        Parameters
        ----------
        instance : array-like, of shape (n_features)
            Instance to generate the neighborhood from
        size : int
            Size of the neighborhood
        class_name : str
            Name of the feature that is the class value
        bb : scikit-learn compatible predictor
            Black-box already fitted with the input data.
        """
        self._X = None
        self._y = None
        self._Xy = None
        self.instance = instance
        self.size = size
        self.class_name = class_name
        self.bb = bb

    @abstractmethod
    def fit(self):
        '''
        Perform the necessary operations to get
        a neighborhood from a given instance
        '''

    def get_X(self):
        """
        Return the neighborhood

        Returns
        -------
        array-like, of shape (size, n_features)
            Neighborhood

        Raises
        ------
        NotFittedError
            If the neighborhood is not fitted yet
        """
        if self._X is None:
            raise NotFittedError
        else:
            return self._X

    def get_y(self):
        """
        Return the black-box prediction of the neighborhood

        Returns
        -------
        array-like, of shape(size)
            Array of the predictions

        Raises
        ------
        NotFittedError
            If the neighborhood is not fitted yet
        """
        if self._y is None:
            raise NotFittedError
        else:
            return self._y
