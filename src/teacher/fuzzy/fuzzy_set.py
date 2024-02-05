# =============================================================================
# Imports
# =============================================================================

# Standard
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third party
import skfuzzy as fuzz
import numpy as np

# =============================================================================
# Classes
# =============================================================================


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
        variable : numpy.ndarray
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

    @abstractmethod
    def simmilarity(self, other):
        """Compute the similarity between two fuzzy sets of the same
        type

        Parameters
        ----------
        other : FuzzySet
            Set to compute the similarity with the current object

        Returns
        -------
        float
            Degree of similarity

        Raises
        ------
        ValueError
            If the set is not of the same subtype
        """


@dataclass
class FuzzyContinuousSet(FuzzySet):
    """
    Dataclass that represents a fuzzy continuous set by assigning a name and a list of points that represent
    the triangles
    """
    fuzzy_points: list
    point_set: bool = False

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.fuzzy_points), self.point_set))

    def __lt__(self, other):
        return self.fuzzy_points < other.fuzzy_points

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

    def simmilarity(self, other):
        if not isinstance(other, FuzzyContinuousSet):
            raise ValueError('Intersection must be between two Fuzzy Sets of the same type')

        # Compute the range for an alpha cut of 0.5 because we assume fuzzy strong partitions
        min_self = (self.fuzzy_points[1] - self.fuzzy_points[0]) / 2
        max_self = (self.fuzzy_points[2] - self.fuzzy_points[1]) / 2

        min_other = (other.fuzzy_points[1] - other.fuzzy_points[0]) / 2
        max_other = (other.fuzzy_points[2] - other.fuzzy_points[1]) / 2

        # if the ranges don't intersect the simmilarity is zero
        if min_self >= max_other or min_other >= max_self:
            return 0

        # Else compute the intersection divided by the union of the ranges
        inters = min(max_self, max_other) - max(min_self, min_other)
        union = max(max_self, max_other) - min(min_self, min_other)
        return inters / union

    def alpha_cut(self, cut):
        '''Return the interval of the alpha cut'''

        if cut < 0 or cut > 1:
            raise ValueError('The alpha cut must be between 0 and 1')

        left_offset = (self.fuzzy_points[1] - self.fuzzy_points[0]) * cut
        right_offset = (self.fuzzy_points[2] - self.fuzzy_points[1]) * cut

        return (self.fuzzy_points[0] + left_offset, self.fuzzy_points[2] - right_offset)

    @staticmethod
    def merge(a, b):
        '''Merge two fuzzy sets the min and max of their fuzzy points and the
        mean of their middle points'''
        new_name = np.mean([float(a.name), float(b.name)])
        return FuzzyContinuousSet(str(new_name),
                                  [min(a.fuzzy_points[0], b.fuzzy_points[0]),
                                   np.mean([a.fuzzy_points[1], b.fuzzy_points[1]]),
                                   max(a.fuzzy_points[2], b.fuzzy_points[2])])

    @staticmethod
    def jaccard_similarity(a, b):
        '''Compute the Jaccard similarity between two fuzzy sets
        with triangular membership functions'''

        # Define the ranges for the common support
        common_support_start = max(a.fuzzy_points[0], b.fuzzy_points[0])
        common_support_end = min(a.fuzzy_points[2], b.fuzzy_points[2])

        if common_support_start >= common_support_end:
            return 0

        # Calculate the intersection and union
        intersection = common_support_end - common_support_start
        union = max(a.fuzzy_points[2], b.fuzzy_points[2]) - min(a.fuzzy_points[0], b.fuzzy_points[0])
        jaccard_similarity = intersection / union * a.intersection(b)

        return jaccard_similarity


@dataclass
class FuzzyDiscreteSet(FuzzySet):
    """
    Dataclass that represents a fuzzy continuous set by assigning a name that represents the value
    of this set.
    """
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

    def simmilarity(self, other):
        return self.intersection(other)
