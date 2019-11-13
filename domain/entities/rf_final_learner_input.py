import csv
import math
import random

import numpy
import pandas

from domain.encoders.boolean_encoder import BooleanEncoder
from domain.encoders.categorical_encoder import CategoricalEncoder
from domain.encoders.missing_value_encoder import MissingValueEncoder
from domain.feature_extractors.word_extractor import WordExtractor


class RFFinalLearnerInput:

    @staticmethod
    def from_invites(invites, country_categories, fluency_categories, random_seed=0):
        x = []
        y = []
        for invite in invites:
            features = [
                # invite.forum_id,
                invite.spectrum_id,
                invite.citizen_id,
                invite.location.country,
                invite.location.gave_province,
                invite.fluencies.count,
                invite.fluencies.one,
                invite.fluencies.two,
                invite.fluencies.three,
                invite.fluencies.four,
                invite.has_custom_avatar,
                invite.enlisted.timestamp() if invite.enlisted is not None else None,
                invite.enlisted.year if invite.enlisted is not None else None,
                invite.enlisted.month if invite.enlisted is not None else None,
                # invite.enlisted.day if invite.enlisted is not None else None,
                # invite.enlisted.time().hour if invite.enlisted is not None else None,
                invite.enlisted.weekday() if invite.enlisted is not None else None,
                # invite.enlisted.weekday() >= 5 if invite.enlisted is not None else None,
                invite.invite_sent.timestamp() if invite.invite_sent is not None else None,
                # invite.invite_sent.month if invite.invite_sent is not None else None,
                # invite.invite_sent.day if invite.invite_sent is not None else None,
                invite.invite_sent.time().hour if invite.invite_sent is not None else None,
                invite.invite_sent.weekday() if invite.invite_sent is not None else None,
                # invite.invite_sent.weekday() >= 5 if invite.invite_sent is not None else None,
                # invite.chat_last_presence.timestamp() if invite.chat_last_presence is not None else None,
                # invite.chat_last_presence.month if invite.chat_last_presence is not None else None,
                # invite.chat_last_presence.day if invite.chat_last_presence is not None else None,
                # invite.chat_last_presence.time().hour if invite.chat_last_presence is not None else None,
                # invite.chat_last_presence.weekday() if invite.chat_last_presence is not None else None,
                # invite.chat_last_presence.weekday() >= 5 if invite.chat_last_presence is not None else None,
                # invite.forum_last_active.timestamp() if invite.forum_last_active is not None else None,
                # invite.forum_last_active.month if invite.forum_last_active is not None else None,
                # invite.forum_last_active.day if invite.forum_last_active is not None else None,
                # invite.forum_last_active.time().hour if invite.forum_last_active is not None else None,
                # invite.forum_last_active.weekday() if invite.forum_last_active is not None else None,
                # invite.forum_last_active.weekday() >= 5 if invite.forum_last_active is not None else None,
                True if invite.forum_id is not None else False,
                True if invite.spectrum_id is not None else False,
                WordExtractor.first_four_letters_match(invite.moniker, invite.handle),
                WordExtractor.has_number_in_first_four_letter(invite.moniker),
                WordExtractor.has_number_in_first_four_letter(invite.handle),
                WordExtractor.has_special_characters(invite.moniker),
                WordExtractor.has_special_characters(invite.handle),
                WordExtractor.damerau_levenshtein_distance(invite.moniker, invite.handle),
                # WordExtractor.damerau_levenshtein_distance(WordExtractor.nysiis(invite.moniker),
                #                                            WordExtractor.nysiis(invite.handle)),
            ]
            targets = [
                invite.recruit_status.joined
            ]
            x.append(features)
            y.append(targets)
        feature_info = [
            # {'name': "ForumId", 'category_type': None},
            {'name': "SpectrumId", 'category_type': None},
            {'name': "CitizenId", 'category_type': None},
            {'name': "Country", 'category_type': 'country'},
            {'name': "GaveProvince", 'category_type': None},
            {'name': "FluencyCount", 'category_type': None},
            {'name': "Fluency1", 'category_type': 'fluency'},
            {'name': "Fluency2", 'category_type': 'fluency'},
            {'name': "Fluency3", 'category_type': 'fluency'},
            {'name': "Fluency4", 'category_type': 'fluency'},
            {'name': "HasCustomAvatar", 'category_type': None},
            {'name': "EnlistedDate", 'category_type': None},
            {'name': "EnlistedYear", 'category_type': None},
            {'name': "EnlistedMonth", 'category_type': None},
            # {'name': "EnlistedDay", 'category_type': None},
            # {'name': "EnlistedTime", 'category_type': None},
            {'name': "EnlistedDayOfWeek", 'category_type': None},
            # {'name': "EnlistedIsWeekend", 'category_type': None},
            {'name': "InviteSentDate", 'category_type': None},
            # {'name': "InviteSentMonth", 'category_type': None},
            # {'name': "InviteSentDay", 'category_type': None},
            {'name': "InviteSentTime", 'category_type': None},
            {'name': "InviteSentDayOfWeek", 'category_type': None},
            # {'name': "InviteSentIsWeekend", 'category_type': None},
            # {'name': "ChatLastPresenceDate", 'category_type': None},
            # {'name': "ChatLastPresenceMonth", 'category_type': None},
            # {'name': "ChatLastPresenceDay", 'category_type': None},
            # {'name': "ChatLastPresenceTime", 'category_type': None},
            # {'name': "ChatLastPresenceDayOfWeek", 'category_type': None},
            # {'name': "ChatLastPresenceIsWeekend", 'category_type': None},
            # {'name': "ForumLastActiveDate", 'category_type': None},
            # {'name': "ForumLastActiveMonth", 'category_type': None},
            # {'name': "ForumLastActiveDay", 'category_type': None},
            # {'name': "ForumLastActiveTime", 'category_type': None},
            # {'name': "ForumLastActiveDayOfWeek", 'category_type': None},
            # {'name': "ForumLastActiveIsWeekend", 'category_type': None},
            {'name': "HasVisitedForum", 'category_type': None},
            {'name': "HasVisitedSpectrum", 'category_type': None},
            {'name': "FirstFourLettersMatch", 'category_type': None},
            {'name': "MonikerHasNumber", 'category_type': None},
            {'name': "HandleHasNumber", 'category_type': None},
            {'name': "MonikerHasSpecial", 'category_type': None},
            {'name': "HandleHasSpecial", 'category_type': None},
            {'name': "NamesSimilarity", 'category_type': None},
            # {'name': "PhoneticSimilarity", 'category_type': None},
        ]
        assert(len(x[0]) == len(feature_info))
        return RFFinalLearnerInput(x, y, feature_info, country_categories, fluency_categories, random_seed)

    def __init__(self, x, y, feature_info, country_categories, fluency_categories, random_seed=0):
        self.__x = x
        self.__y = y
        self.__feature_info = feature_info
        self.__country_categories = country_categories
        self.__fluency_categories = fluency_categories
        self.__random_seed = random_seed

    def preprocess(self, equalize_target_classes=False):
        # Shrink data if needed
        if equalize_target_classes:
            self.equalize_target_classes_sample_sizes()

        boolean_encoder = BooleanEncoder()
        boolean_encoder.transform(self.__x)
        boolean_encoder.transform(self.__y)

        categorical_encoder = CategoricalEncoder()
        categorical_encoder.fit(self.__country_categories, self.__fluency_categories)
        categorical_encoder.transform(self)

        missing_value_encoder = MissingValueEncoder()
        missing_value_encoder.transform(self.__x)

        self.validate()

    def get_data_statistics(self):
        # Overview stats
        total_positives = 0
        total_negatives = 0
        for i in range(len(self.__x)):
            if self.__y[i][0]:
                total_positives += 1
            else:
                total_negatives += 1
        # Missing data stats
        total_missing = 0
        missing_positives = 0
        missing_negatives = 0
        for i in range(len(self.__x)):
            row = self.__x[i]
            for feature in row:
                if feature is None or feature == math.nan:
                    total_missing += 1
                    if self.__y[i][0]:
                        missing_positives += 1
                    else:
                        missing_negatives += 1
                    break
        stats = {
            'Total examples': len(self.__x),
            'Total positives': total_positives,
            'Total negatives': total_negatives,
            'Total missing': total_missing,
            'Missing positives': missing_positives,
            'Missing negatives': missing_negatives,
        }
        return stats

    def validate(self):
        for sample in self.__x:
            for feature in sample:
                if not isinstance(feature, (float, int, numpy.integer, numpy.float)):
                    raise Exception
        for sample in self.__y:
            for target in sample:
                if not isinstance(target, (float, int, numpy.integer, numpy.float)):
                    raise Exception

    def get_feature_category_type(self, index):
        return self.__feature_info[index]['category_type']

    def get_feature_count(self):
        # return numpy.array(self.__x).shape[1]
        return len(self.__x[0])

    @property
    def x_train(self):
        return self.__x

    @property
    def y_train(self):
        return self.__y

    def to_numpy_xy_arrays(self):
        return numpy.array(self.__x), numpy.array(self.__y)

    def get_x_train_data_frame(self, alternative_x=None):
        x = self.__x
        if alternative_x is not None:
            x = alternative_x
        x = numpy.array(x)
        table = {}
        data_frame = None
        for i in range(self.get_feature_count()):
            table[self.__feature_info[i]['name']] = x[:, i]
            data_frame = pandas.DataFrame(table)
        del table
        return data_frame

    def get_full_xy_data_frame(self, alternative_x=None):
        x = self.__x
        if alternative_x is not None:
            x = alternative_x
        x = numpy.array(x)
        y = numpy.array(self.__y)
        xy = numpy.append(x, y, axis=1)
        table = {}
        data_frame = None
        for i in range(self.get_feature_count()):
            table[self.__feature_info[i]['name']] = xy[:, i]
        table['target'] = xy[:, self.get_feature_count()]
        data_frame = pandas.DataFrame(table)
        del table
        return data_frame

    def index_to_feature_name(self, index):
        return self.__feature_info[index]['name']

    def get_data_counts(self):
        positives = 0
        negatives = 0
        for i in range(len(self.__x)):
            if self.__y[i][0]:
                positives += 1
            else:
                negatives += 1
        return len(self.__x), positives, negatives

    def get_missing_data_counts(self):
        total_missing = 0
        missing_positives = 0
        missing_negatives = 0
        for i in range(len(self.__x)):
            row = self.__x[i]
            for feature in row:
                if feature is None or feature == math.nan:
                    total_missing += 1
                    if self.__y[i][0]:
                        missing_positives += 1
                    else:
                        missing_negatives += 1
                    break
        return total_missing, missing_positives, missing_negatives

    def equalize_target_classes_sample_sizes(self):
        _, positives, negatives = self.get_data_counts()
        remove_positives = (positives > negatives)
        amount_to_remove = abs(positives - negatives)
        random.seed(self.__random_seed)
        while amount_to_remove > 0:
            index = random.randint(0, len(self.__x) - 1)
            if self.__y[index][0] and remove_positives or not self.__y[index][0] and not remove_positives:
                del self.__x[index]
                del self.__y[index]
                amount_to_remove -= 1

    def save_as_csv(self, filename):
        file = open(filename, 'w')
        wr = csv.writer(file, lineterminator='\n')
        wr.writerows(self.__x)
        file.close()
