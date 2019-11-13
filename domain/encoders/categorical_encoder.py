from sklearn.preprocessing import LabelEncoder


class CategoricalEncoder:

    def __init__(self):
        self.__country_encoder = LabelEncoder()
        self.__fluency_encoder = LabelEncoder()

        # It seems Scikit OrdinalEncoder hasn't been released to their stable version, even though they updated their
        # documentation. Use it instead of LabelEncoder when it is available.
        # self.__ordinal_encoder = OrdinalEncoder(categories=[
        #     None, None, None, None, None, None, None, None,
        #     country_categories,
        #     None, None,
        #     fluency_categories,
        #     fluency_categories,
        #     fluency_categories,
        #     fluency_categories,
        #     None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        # ])

    def fit(self, country_categories, fluency_categories):
        self.__country_encoder.fit(country_categories)
        self.__fluency_encoder.fit(fluency_categories)

    def transform(self, input_data):
        x = input_data.x_train
        for sample in x:
            for index in range(len(sample)):
                category_type = input_data.get_feature_category_type(index)
                if category_type == 'country':
                    sample[index] = self.__country_encoder.transform([sample[index]])[0]
                elif category_type == 'fluency':
                    sample[index] = self.__fluency_encoder.transform([sample[index]])[0]

        # NOTE: Scikit's Random Forest and Decision Tree implementations may not work well with categorical features
        # expressed as integers or otherwise continuous numbers. This is not a limitation of Random Forest, just this
        # library's implementation of it. Other implementations will work fine with categorical features expressed as
        # ordinals.
        #
        # Therefore, we may need to one-hot encode the categorical features, such as location and fluency. However,
        # since our categorical features have a high cardinality, I want experiment with ordinals first. The
        # following code may be useful if ordinals prove ineffective and we must one-hot encode the features. If
        # that's the case, reduce the cardinality by condensing categories that have a low correlation with the
        # output into catch-all buckets.
        #
        # Documentation for this function is at:
        # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
        # http://scikit-learn.org/stable/modules/preprocessing.html
        # one_hot_encoder = preprocessing.OneHotEncoder(
        #     n_values=[
        #         0,
        #         0,
        #         0,
        #         ...
        #     ],
        #     categorical_features=[
        #         False,
        #         False,
        #         False,
        #         ...
        #     ],
        #     handle_unknown='error')
        # one_hot_encoder.transform(X)

    def fit_transform(self, samples):
        raise Exception
