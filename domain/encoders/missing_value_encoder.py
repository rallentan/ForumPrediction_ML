import math
import numpy


class MissingValueEncoder:

    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self, samples):
        for sample in samples:
            for i in range(len(sample)):
                if sample[i] is None:
                    sample[i] = numpy.float(math.nan)  # NaN

    def fit_transform(self, samples):
        self.fit()
        self.transform(samples)
