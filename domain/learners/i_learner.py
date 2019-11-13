# This class represents an interface that all LearnerInput classes must implement. Python does not support interfaces.
# This file exists for documentation and code completion purposes.


class ILearner:

    def cross_validate(self, i_learner_input, scoring_metric=None):
        pass

    def inspect(self, i_learner_input):
        pass
