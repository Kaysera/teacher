from collections import defaultdict
import numpy as np


def _aggregated_vote(self, all_classes):
    agg_vote = defaultdict(lambda: np.zeros(len(all_classes[0][1])))
    for leaf in all_classes:
        for key in leaf[0]:
            agg_vote[key] += leaf[0][key] * leaf[1]
    return agg_vote


def _maximum_matching(self, all_classes):
    max_match = defaultdict(lambda: np.zeros(len(all_classes[0][1])))
    for leaf in all_classes:
        for key in leaf[0]:
            max_match[key] = np.maximum(max_match[key], (leaf[0][key] * leaf[1]))
    return max_match
