# =============================================================================
# Imports
# =============================================================================

# Standard
from dataclasses import dataclass

# =============================================================================
# Classes
# =============================================================================


@dataclass
class FuzzyVariable():
    """
    Dataclass that represents a fuzzy variable by assigning a name and a list of fuzzy sets
    """
    name: str
    fuzzy_sets: list

    def membership(self, variable):
        """
        Compute the membership degree of the variable to the different fuzzy sets of the variable

        Parameters
        ----------
        variable : array-like, of shape (n_features)
            The variable to compute the membership degree

        Returns
        -------
        dict
            {set_1: pert_1, ...} for each fuzzy set of the variable
        """
        return {fs.name: fs.membership(variable) for fs in self.fuzzy_sets}
