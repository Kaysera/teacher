from abc import ABC, abstractmethod


class NotFittedError(Exception):
    pass


class BaseNeighborhood(ABC):

    def __init__(self, instance, size, class_name, bb):
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
        Perform the operation necessary to get a neighborhood from a
        given instance
        '''

    def get_X(self):
        if self._X is None:
            raise NotFittedError
        else:
            return self._X

    def get_y(self):
        if self._y is None:
            raise NotFittedError
        else:
            return self._y
