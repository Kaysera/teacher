from dataclasses import dataclass
import skfuzzy as fuzz
from abc import ABC, abstractmethod


@dataclass
class FuzzySet(ABC):
    name: str

    @abstractmethod
    def membership(self, variable):
        """Take the values of a variable
        and returns an array with the membership
        of those values to the Fuzzy Set

        Parameters
        ----------
        variable : NumPy array
            Array with values of the variable
        """

    @abstractmethod
    def intersection(self, other):
        """Intersect two fuzzy sets of the same
        type

        Parameters
        ----------
        other : FuzzySet
            Set to intersect with the current object

        Returns
        -------
        float
            Degree of intersection

        Raises
        ------
        ValueError
            If the set is not of the same subtype
        """


@dataclass
class FuzzyContinuousSet(FuzzySet):
    fuzzy_points: list

    def membership(self, variable):
        return fuzz.trimf(variable, self.fuzzy_points)

    def intersection(self, other):
        if not isinstance(other, FuzzyContinuousSet):
            raise ValueError('Intersection must be between two Fuzzy Sets of the same type')

        A = ((self.fuzzy_points[0], 0), (self.fuzzy_points[1], 1))
        B = ((self.fuzzy_points[1], 1), (self.fuzzy_points[2], 0))

        C = ((other.fuzzy_points[0], 0), (other.fuzzy_points[1], 1))
        D = ((other.fuzzy_points[1], 1), (other.fuzzy_points[2], 0))

        all_intersects = [
                            self._line_intersect(A, C),
                            self._line_intersect(A, D),
                            self._line_intersect(B, C),
                            self._line_intersect(B, D)
                         ]
        return max(all_intersects)

    def _line_intersect(self, A, B):
        if A == B:
            return 1

        ((Ax1, Ay1), (Ax2, Ay2)) = A
        ((Bx1, By1), (Bx2, By2)) = B

        denom = (Ax1 - Ax2) * (By1 - By2) - (Ay1 - Ay2) * (Bx1 - Bx2)
        if denom == 0:
            return 0

        y = (Ax1 * Ay2 - Ay1 * Ax2) * (By1 - By2) - (Ay1 - Ay2) * (Bx1 * By2 - By1 * Bx2)
        y /= denom

        if y < 0 or y > 1:
            return 0
        else:
            return y


@dataclass
class FuzzyDiscreteSet(FuzzySet):
    value: str

    def membership(self, variable):
        try:
            return (variable == self.value).astype(int)
        except AttributeError:
            return int(variable == self.value)

    def intersection(self, other):
        if not isinstance(other, FuzzyDiscreteSet):
            raise ValueError('Intersection must be between two Fuzzy Sets of the same type')

        return int(self == other)
