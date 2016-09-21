from collections import defaultdict
from copy import deepcopy


class FeatureVector(defaultdict):
    def __init__(self):
        super(FeatureVector, self).__init__(float)
        return

    def __add__(self, otherFeatureVector):
        result = deepcopy(self)
        for key in otherFeatureVector:
            result[key] += otherFeatureVector[key]
        return result

    def __iadd__(self, otherFeatureVector):
        for key in otherFeatureVector:
            self[key] += otherFeatureVector[key]
        return self

    def __sub__(self, otherFeatureVector):
        result = deepcopy(self)
        for key in otherFeatureVector:
            result[key] -= otherFeatureVector[key]
        return result

    def __isub__(self, otherFeatureVector):
        for key in otherFeatureVector:
            self[key] -= otherFeatureVector[key]
        return self

    def __eq___(self, otherFeatureVector):
        if len(self) != len(otherFeatureVector):
            return False
        for key in otherFeatureVector:
            if self[key] != otherFeatureVector[key]:
                return False
        return True
