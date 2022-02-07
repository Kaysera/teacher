from abc import ABC


class BaseExplainer(ABC):
    def __init__(self):
        pass

    def fit(self):
        pass

    def get_explanation(self):
        pass

    def get_score(self):
        pass

    def map_factual(self):
        pass

    def precision(self):
        pass

    def coverage(self):
        pass

    def hit(self):
        pass

    def fidelity(self):
        pass

    def l_fidelity(self):
        pass

    def cl_fidelity(self):
        pass
