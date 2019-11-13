import sys
from unittest import TestCase

from domain.feature_extractors.word_extractor import WordExtractor


class WordExtractorTests(TestCase):

    def setUp(self):
        self.test_names = [
            'Christian',
            'niccoheff10',
            'Andi3000',
            'Gaveron',
            'Sazzik',
            'Roik Belantor',
            'Serotos',
            'FlyingMustache',
            'GurglingMicrobe',
            'Tekhed',
            'Bolden',
            'Operator',
            'Razfu',
            'Achilles',
            'DarkHero',
            'Klosterabt',
            'Dagnabbit',
            'Yana',
            'Tschetan',
            'Dixie Flatline',
            'Sunya',
            'Oldmachine One',
            'Nep Nep',
            'RoyalPain',
            'Ruxiang',
            'jk',
            'bordon04',
            'xiaoliuze',
            'Badaboom85',
            'xanatea',
            'Neil Hardstrong',
            'Lacourse',
            'Hermiod',
            'Cosmoss',
            'Athine',
            'Blackorin',
            'JonDickson20',
            'bouvretron',
            'Sharp',
            'LLothos',
            'Cire',
            'DamonSoul',
        ]

    def test_damerau_levenshtein_distance(self):
        self.assertEqual(WordExtractor.damerau_levenshtein_distance('ZX', 'XYZ'), 2)
        self.assertEqual(WordExtractor.damerau_levenshtein_distance('BADC', 'ABCD'), 2)
        self.assertEqual(WordExtractor.damerau_levenshtein_distance('jellyifhs', 'jellyfish'), 2)
        self.assertEqual(WordExtractor.damerau_levenshtein_distance('ifhs', 'fish'), 2)

    def test_grapheme_to_phoneme_generators(self):
        print("{:20}{:20}{:20}{:20}{:20}".format('ORIGINAL',
                                                 'METAPHONE',
                                                 'SOUNDEX',
                                                 'NYSIIS',
                                                 'MATCH RATING CODEX',
                                                 ))
        for name in self.test_names:
            print("{:20}{:20}{:20}{:20}{:20}".format(name,
                                                     WordExtractor.metaphone(name),
                                                     WordExtractor.soundex(name),
                                                     WordExtractor.nysiis(name),
                                                     WordExtractor.match_rating_codex(name),
                                                     ))
        sys.stdout.flush()
