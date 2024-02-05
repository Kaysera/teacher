# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC, abstractmethod

# Third party
import numpy as np
from sklearn.utils import check_array, check_X_y

# =============================================================================
# Classes
# =============================================================================


class BaseDecisionTree(ABC):
    """
    Base abstract decision tree that provides the basic methods
    to implement the rest of the decision trees.
    """
    def __init__(self, features, th=0.0001, max_depth=2, min_num_examples=1, prunning=True):
        """
        Parameters
        ----------
        features : list
            Sorted list of features as they will appear in the dataset
        th : float, optional
            Minimum gain threshold to keep branching the tree, by default 0.0001
        max_depth : int, optional
            Maximum depth of the tree, by default 2
        min_num_examples : int, optional
            Minimum number of examples per leaf, by default 1
        prunning : bool, optional
            Whether or not to prune the tree, by default True
        """
        self.max_depth = max_depth
        self.features = features
        self.th = th
        self.prunning = prunning
        self.min_num_examples = min_num_examples
        self.tree_ = None

    @abstractmethod
    def fit(self, X, y):
        """Build a decision tree classifier from the training set (X, y)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted
            to *dtype=np.float64* if possible or *dtype=object* otherwise.
        y : array-like of shape (n_samples,)
            The target values (class labels) as integers or strings.
        """

    def predict(self, X):
        """
        Predict class value for X.

        The predicted class for each sample in X is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted
            to *dtype=np.float64* if possible or *dtype=object* otherwise.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes
        """
        X = check_array(X, dtype=['float64', 'object'])
        return np.array([self.tree_.predict(X)])

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for *X*

        Returns
        -------
        score : float
            Mean accuracy of *self.predict(X)* wrt. *y*
        """
        X, y = check_X_y(X, y, dtype=['float64', 'object'])
        return np.sum(self.predict(X) == y)/y.shape[0]

    def to_rule_based_system(self, verbose=False, simplify=False):
        """
        Return the tree as a rule-based system

        Parameters
        ----------
        verbose : bool, optional
            debug flag, by default False
        simplify : bool, optional
            Whether or not to simplify the rules, by default False

        Returns
        -------
        rule_system : list[Rule]
            List of the rules extracted from the tree
        """
        return self.tree_.to_rule_based_system(verbose, simplify)
