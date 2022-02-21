from dataclasses import dataclass
import skfuzzy as fuzz
from abc import ABC, abstractmethod


@dataclass
class FuzzySet(ABC):
    name: str

    @abstractmethod
    def membership(self, variable):
        pass

    @abstractmethod
    def intersection(self, other):
        pass


@dataclass
class FuzzyContinuousSet(FuzzySet):
    fuzzy_points: list

    def membership(self, variable):
        return fuzz.trimf(variable, self.fuzzy_points)

    def intersection(self, other):
        pass


@dataclass
class FuzzyDiscreteSet(FuzzySet):
    value: str

    def membership(self, variable):
        try:
            return (variable == self.value).astype(int)
        except AttributeError:
            return variable == self.value

    def intersection(self, other):
        pass
