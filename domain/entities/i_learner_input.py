# This class represents an interface that all LearnerInput classes must implement. Python does not support interfaces.
# This file exists for documentation and code completion purposes.


class ILearnerInput:

    @property
    def x_train(self):
        return

    @property
    def y_train(self):
        return

    @property
    def x_test(self):
        return

    @property
    def y_test(self):
        return

    @property
    def is_preprocessed(self):
        return

    def get_data_statistics(self):
        pass

    def preprocess(self):
        pass
