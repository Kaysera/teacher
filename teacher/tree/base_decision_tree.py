from abc import ABC, abstractmethod
from sklearn.utils import check_array, check_X_y
import numpy as np


class BaseDecisionTree(ABC):
    def __init__(self, features, th=0.0001, max_depth=2, min_num_examples=1, prunning=True):
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
        X : [type]
            [description]
        y : [type]
            [description]
        """

    def predict(self, X):
        X = check_array(X)
        return np.array([self.tree_.predict(x) for x in X])

    def score(self, X, y):
        X, y = check_X_y(X, y)
        return np.sum(self.predict(X) == y)/y.shape[0]

    def to_rule_based_system(self, verbose=False):
        return self.tree_.to_rule_based_system(verbose=False)
