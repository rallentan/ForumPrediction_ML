import matplotlib.pyplot as plt
from rfpimp import plot_corr_heatmap


class MetricDisplay:

    @staticmethod
    def print_data_statistics(stats):
        print("Data Statistics:")
        for metric, stat in stats.items():
            print("{}: {}".format(metric, stat))

    @staticmethod
    def print_evaluation_metrics(multi_scores):
        print("Model Metrics:")
        for metric, scores in multi_scores.items():
            print("%s: %0.2f (+/- %0.2f)" % (metric, scores.mean(), scores.std() * 2))

    @staticmethod
    def print_feature_importances(importances):
        imp = importances
        print("Feature ranking:")
        for f in range(imp.feature_count):
            # print("%d. feature %d (%f)" % (f + 1, imp.indices[f], imp.importances[imp.indices[f]]))
            print("%d. %s (%f)" % (f + 1, imp.feature_names[f], imp.importances[imp.indices[f]]))

    @staticmethod
    def plot_feature_importances(importances):
        imp = importances
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(imp.feature_count),
                imp.importances[imp.indices],
                color="r",
                yerr=imp.std[imp.indices],
                align="center")
        plt.xticks(range(imp.feature_count), imp.feature_names, rotation=60)
        plt.xlim([-1, imp.feature_count])
        plt.show()

    @staticmethod
    def print_feature_dependencies(dependency_matrix):
        print(dependency_matrix)

    @staticmethod
    def plot_feature_correlations(dependency_matrix):
        plot_corr_heatmap(dependency_matrix, label_fontsize=8, value_fontsize=6)
        return
