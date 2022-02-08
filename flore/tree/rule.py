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

    def __repr__(self):
        return f'Rule({self.antecedent}, {self.consequent}, {self.weight})'

    def __str__(self):
        antecedents = " AND ".join([f"{feat}: {val}"for (feat, val) in self.antecedent])
        return f'{antecedents} => {self.consequent} (Weight: {self.weight})'

    def matching(self, fuzzy_instance, t_norm=min):
        """[summary]

        Parameters
        ----------
        instance : dict
            Instance in fuzzy format {feature: {value: pertenence degree}}
        t_norm : function, optional
            Operation to use as tnorm to get the matching, by default min
        """
        return t_norm([fuzzy_instance[feature][value] for (feature, value) in self.antecedent])
