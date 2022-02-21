from dataclasses import dataclass


@dataclass
class FuzzyVariable():
    name: str
    fuzzy_sets: list

    def membership(self, variable):
        return {fs.name: fs.membership(variable) for fs in self.fuzzy_sets}
