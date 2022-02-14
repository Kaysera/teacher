class Rule:
    def __init__(self, antecedent, consequent, weight):
        """Constructor of rules

        Parameters
        ----------
        antecedent : list of tuples (feature, value)
        consequent : string or number
        weight: weight of the consequent in the tree
        """
        self.antecedent = tuple(antecedent)
        self.consequent = consequent
        self.weight = weight

    def __repr__(self):  # pragma: no cover
        return f'Rule({self.antecedent}, {self.consequent}, {self.weight})'

    def __str__(self):  # pragma: no cover
        antecedents = " AND ".join([f"{feat}: {val}"for (feat, val) in self.antecedent])
        return f'{antecedents} => {self.consequent} (Weight: {self.weight})'

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return (self.antecedent == other.antecedent and
                self.consequent == other.consequent and
                self.weight == other.weight)

    def matching(self, fuzzy_instance, t_norm=min):
        """Matching that an instance has with the rule
        If there is some feature or value not existing in the instance,
        its matching degree is zero

        Parameters
        ----------
        instance : dict
            Instance in fuzzy format {feature: {value: pertenence degree}}
        t_norm : function, optional
            Operation to use as tnorm to get the matching, by default min
        """
        try:
            return t_norm([fuzzy_instance[feature][value] for (feature, value) in self.antecedent])
        except KeyError:
            return 0
