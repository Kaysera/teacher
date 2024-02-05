# =============================================================================
# Imports
# =============================================================================

# Standard
from collections import defaultdict

# Local application
from teacher.fuzzy import FuzzyContinuousSet

# =============================================================================
# Classes
# =============================================================================


class Rule:
    def __init__(self, antecedent, consequent, weight, simplify=False, multiple_antecedents=False):
        """
        Parameters
        ----------
        antecedent : list of tuples (feature, value)
        consequent : string or number
        weight: weight of the consequent in the tree
        simplify : bool, optional
            Whether or not to simplify the rules, by default False
        multiple_antecedents : bool, optional
            Whether or not the rule has multiple antecedents, by default False
        """
        self.antecedent = tuple(antecedent)
        self.consequent = consequent
        self.weight = weight
        self.multiple_antecedents = multiple_antecedents
        if simplify:
            self.simplify()

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

    def __hash__(self) -> int:
        return hash((self.antecedent, self.consequent, self.weight))

    def simplify(self):
        """If there are repeated features in the antecedent, it simplifies the rule
        """
        new_antecedent = {}
        for (feature, value) in self.antecedent:
            if feature not in new_antecedent:
                new_antecedent[feature] = value
            else:
                if len(new_antecedent[feature]) > len(value):
                    new_antecedent[feature] = value
        self.antecedent = tuple(new_antecedent.items())

    def matching(self, instance_membership, t_norm=min):
        """Matching that an instance has with the rule
        If there is some feature or value not existing in the instance,
        its matching degree is zero

        Parameters
        ----------
        instance_membership : dict
            Membership of the instance with the format {feature: {value: pertenence degree}}
        t_norm : function, optional
            Operation to use as tnorm to get the matching, by default min
        """
        if self.multiple_antecedents:
            try:
                memberships = []
                for feature, values in self.antecedent:
                    m = 0
                    for value in values:
                        m += instance_membership[feature][value]
                    memberships.append(m)
                return t_norm(memberships)
            except KeyError:
                return 0
        else:
            try:
                return t_norm([instance_membership[feature][value] for (feature, value) in self.antecedent])
            except KeyError:
                return 0

    def to_json(self, fuzzy_variables):
        """Transform the rule to a json format

        Parameters
        ----------
        fuzzy_variables : list[FuzzyVariable]
            List with the fuzzy variables of the problem

        Returns
        -------
        dict
            Json with the rule
        """
        if self.multiple_antecedents:
            fuzzy_dict = {fv.name: fv.fuzzy_sets for fv in fuzzy_variables}
            json_antecedents = {feature: tuple(values) for (feature, values) in self.antecedent}
            fuzzy_things = []
            for feature, values in self.antecedent:
                fuzzy_sets = {fs.name: fs for fs in fuzzy_dict[feature]}
                for value in values:
                    fuzzy_set = fuzzy_sets[value]
                    if isinstance(fuzzy_set, FuzzyContinuousSet):
                        fuzzy_things.append((feature, fuzzy_set.name, fuzzy_set.fuzzy_points))
                    else:
                        fuzzy_things.append((feature, fuzzy_set.name))
        else:
            fuzzy_dict = {fv.name: fv.fuzzy_sets for fv in fuzzy_variables}
            json_antecedents = {feature: value for (feature, value) in self.antecedent}
            fuzzy_things = []
            for feature, value in self.antecedent:
                fuzzy_sets = {fs.name: fs for fs in fuzzy_dict[feature]}
                fuzzy_set = fuzzy_sets[value]
                if isinstance(fuzzy_set, FuzzyContinuousSet):
                    fuzzy_things.append((feature, fuzzy_set.name, fuzzy_set.fuzzy_points))
                else:
                    fuzzy_things.append((feature, fuzzy_set.name))

        return [json_antecedents, self.consequent, self.weight, fuzzy_things]

    def to_crisp(self, alpha_cut, fuzzy_variables):
        """Transform the rule to a crisp rule

        Parameters
        ----------
        alpha_cut : float
            Alpha cut to use to transform the rule

        Returns
        -------
        Rule
            Crisp rule
        """

        fuzzy_dict = {fv.name: fv.fuzzy_sets for fv in fuzzy_variables}
        new_antecedent = []
        for (feature, value) in self.antecedent:
            fuzzy_sets = fuzzy_dict[feature]
            fuzzy_sets_dict = {fs.name: fs for fs in fuzzy_sets}
            fuzzy_set = fuzzy_sets_dict[value]

            # Check if fuzzy set is FuzzyContinuousSet

            if isinstance(fuzzy_set, FuzzyContinuousSet):
                new_value = fuzzy_set.alpha_cut(alpha_cut)
            else:
                new_value = value

            new_antecedent.append((feature, new_value))

        return Rule(new_antecedent, self.consequent, self.weight)

    @staticmethod
    def weighted_vote(rule_list, instance_membership):
        """Use the weighted vote inference method to return the consequent
        associated to an instance and a rule list given the instance membership

        Parameters
        ----------
        rule_list : list[Rule]
            List with the rules that will be taken into account for the
            weighted vote method
        instance_membership : dict
            Membership of the instance with the format
            {feature: {value: pertenence degree}}

        Returns
        -------
        string or number
            consequent associated with the instance and the rule list
        """
        conse_dict = defaultdict(lambda: 0)
        for rule in rule_list:
            AD = rule.matching(instance_membership) * rule.weight
            conse_dict[rule.consequent] += AD
        return max(conse_dict, key=lambda conse: conse_dict[conse])

    @staticmethod
    def map_rule_variables(rule, origin_fuzzy_variables, dest_fuzzy_variables, map_function='intersection'):
        """Changes the fuzzy variables of the rule
        for ones that are defined in the same universe

        Parameters
        ----------
        rule : Rule
            Original rule to map to the new variables
        origin_fuzzy_variables : list[FuzzyVariable]
            List with the original fuzzy variables
        dest_fuzzy_variables : list[FuzzyVariable]
            List with the destination fuzzy variables
        map_function : str, {'intersection', 'simmilarity'}
            Method to check the best fuzzy set to change

        Returns
        -------
        Rule
            Rule with the new variables

        Raises
        ------
        ValueError
            If the universes of the variables are not the same
            it raises an error
        """

        origin_dict = {fv.name: fv.fuzzy_sets for fv in origin_fuzzy_variables}
        dest_dict = {fv.name: fv.fuzzy_sets for fv in dest_fuzzy_variables}

        if origin_dict.keys() != dest_dict.keys():
            raise ValueError('The universes of the fuzzy variables are not the same')

        new_antecedent = []
        for feat, value in rule.antecedent:
            origin_feat = origin_dict[feat]
            dest_feat = dest_dict[feat]

            origin_fuzzy_sets = {fs.name: fs for fs in origin_feat}
            origin_fs = origin_fuzzy_sets[value]

            if map_function == 'intersection':
                dest_fs = max(dest_feat, key=lambda fs: fs.intersection(origin_fs))
            elif map_function == 'simmilarity':
                dest_fs = max(dest_feat, key=lambda fs: fs.simmilarity(origin_fs))
            else:
                raise ValueError(f'Map function {map_function} not supported')
            new_antecedent.append((feat, dest_fs.name))

        return Rule(new_antecedent, rule.consequent, rule.weight)
