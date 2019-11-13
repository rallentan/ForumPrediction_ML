
class FeatureImportances:

    def __init__(self, importances, indices, std, feature_count, feature_names):
        self.importances = importances
        self.indices = indices
        self.std = std
        self.feature_count = feature_count
        self.feature_names = feature_names
