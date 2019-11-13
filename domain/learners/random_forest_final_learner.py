import math
import numpy
from rfpimp import feature_dependence_matrix

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from domain.value_objects.feature_importances import FeatureImportances


class RandomForestFinalLearner:

    def __init__(self, number_of_features, random_seed):
        self.__number_of_features = number_of_features

        # Documentation of function:
        # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html#sklearn.preprocessing.Imputer
        self.__imputer = preprocessing.Imputer(missing_values="NaN",
                                               strategy="median",
                                               verbose=1,
                                               axis=0,
                                               # Probably not necessary, but future-proof. Setting to false may improve
                                               # performance and reduce memory requirements.
                                               copy=True)

        # We want to manually control the number of features at each split in order to tune the algorithm for Spectrum
        # data.
        features_at_each_split = int(math.sqrt(number_of_features) + 0)

        # Unfortunately, this library doesn't allow early stopping when the next split has a higher impurity
        # than the current split.
        #
        # Documentation for this function is at:
        # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_log_proba
        self.__classifier = RandomForestClassifier(
            bootstrap=True,
            class_weight=None,
            criterion='gini',
            # Since early stopping options are limited, we might
            # need to tweak this number for short-term results.
            max_depth=None,
            max_features=features_at_each_split,
            # I don't think setting an arbitrary limit to leaf nodes is a good way to prevent over-
            # splitting, but this could potentially be tuned for early stopping purposes.
            max_leaf_nodes=None,
            # This is the option that lacks the early stopping method I want to use.
            # Since early stopping options are limited, this is another option that can be tuned to
            # avoid too many internal nodes.
            min_impurity_split=None,
            min_impurity_decrease=0.0,
            # 100 seems like a decent sample size, and may be a temporary solution to our early
            # stopping problem. Scratch that. Available data is much lower than previously mentioned.
            min_samples_split=10,
            min_samples_leaf=1,
            # Spectrum and recruiter data isn't weighted, so this should be zero.
            min_weight_fraction_leaf=0.0,
            # Seems like a good average number to start with according to this research paper:
            # https://www.researchgate.net/publication/230766603_How_Many_Trees_in_a_Random_Forest
            n_estimators=96,
            # Make sure this correctly detects the number of cores on the bare-metal
            # production server and runs on all them.
            n_jobs=-1,
            oob_score=False,
            random_state=random_seed,
            verbose=1,
            warm_start=False)

        # A pipeline is necessary when imputing missing values to avoid leaking statistics about the test data into
        # the model when cross-validating.
        #
        # Documentation of function:
        # http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline
        pipeline = Pipeline([("imputer", self.__imputer),
                             ("forest", self.__classifier)],
                            # Enabling this may improve performance by caching
                            memory=None)
        self.__pipeline = pipeline

    def cross_validate(self, learner_input, scoring_metrics='roc_auc'):
        # IMPORTANT: Do not shuffle the data before running it through the estimator/classifier because our samples are
        # not i.i.d. They were generated using a time-dependent process. That is, because our samples were collected
        # over a period of time using a recruitment process that evolved and had memory of past generated samples, there
        # are likely to be correlations between observations that are near in time, and therefore must be kept together
        # during training.
        if type(scoring_metrics) is str:
            scoring_metrics = [scoring_metrics]
        multi_scores = {}
        for scoring_metric in scoring_metrics:
            scores = cross_val_score(self.__pipeline, learner_input.x_train, numpy.array(learner_input.y_train).ravel(),
                                     scoring=scoring_metric,
                                     # Most authors (and empirical evidence) suggest that 5 to 10 folds are effective. I
                                     # select 5 folds because of our potentially small data set.
                                     #
                                     # Also, this will automatically select StratifiedKFold for our cross-validator
                                     # by default, preserving the percentage of samples for each class in each
                                     # subset. This is what we want because of the imbalance in the distribution of
                                     # target classes for our problem (i.e. joins to not-joins).
                                     #
                                     # TODO: Split the input data using a TimeSeriesSplit because our data is a
                                     # time-series.
                                     cv=5,
                                     # Make sure this correctly detects the number of cores on the bare-metal
                                     # production server and runs on all them.
                                     n_jobs=-1,
                                     verbose=1,
                                     # If memory consumption on the production server is too high, set this to "n_jobs",
                                     # though that will come at the cost of performance.
                                     pre_dispatch="2*n_jobs")
            multi_scores[scoring_metric] = scores
        return multi_scores

    # Scikit uses "mean decrease in impurity" (gini importance) to measure feature importances, which is not always
    # reliable. Use the more recent "permutation importance" method when you have split out-of-bag test data away from
    # the training data. See https://github.com/parrt/random-forest-importances for more information.

    def inspect(self, learner_input):
        importances = self.find_importances_by_gini_importance(learner_input)
        dependencies = self.generate_feature_dependency_matrix(learner_input)
        return {'importances': importances, 'dependencies': dependencies}

    def find_importances_by_gini_importance(self, learner_input):
        self.__pipeline.fit(learner_input.x_train, learner_input.y_train)
        importances = self.__classifier.feature_importances_
        std = numpy.std([tree.feature_importances_ for tree in self.__classifier.estimators_], axis=0)
        indices = numpy.argsort(importances)[::-1]
        feature_names = []
        for index in indices:
            feature_names.append(learner_input.index_to_feature_name(index))
        return FeatureImportances(importances, indices, std, self.__number_of_features, feature_names)

    def find_importances_by_permutation(self, X_test, y_test):
        # When you have split out test data and are ready to use this method, uncomment these lines and pip install
        # rfpimp
        # # imp = importances(rf, X_test, y_test, n_samples=-1)  # permutation
        # # plot_importances(imp)
        raise NotImplementedError

    def find_feature_dependencies(self, feature_index):
        # dependencies = oob_dependences(random_forest_model, X_train)
        # print(dependencies)
        raise NotImplementedError

    def generate_feature_dependency_matrix(self, learner_input):
        # print("...imputing missing values")
        x_imputed = self.__imputer.fit_transform(learner_input.x_train)
        # print("...converting to data frame")
        xy_data_frame = learner_input.get_full_xy_data_frame(alternative_x=x_imputed)
        del x_imputed
        # print("...computing dependencies")
        rfr = RandomForestRegressor(n_estimators=96, n_jobs=-1, oob_score=True)
        matrix = feature_dependence_matrix(rfr, xy_data_frame)
        return matrix
