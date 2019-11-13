
class BooleanEncoder:

    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self, samples):
        for sample in samples:
            for i in range(len(sample)):
                if type(sample[i]) is bool:
                    sample[i] = 1.0 if sample[i] else 0.0

    def fit_transform(self, samples):
        self.fit()
        self.transform(samples)
