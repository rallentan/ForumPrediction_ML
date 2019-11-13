# Adapter for the Damerau-Levenshtein algorithm

import jellyfish._jellyfish


# Notes:
# - jellyfish expects unicode input. No action is required in Python 3.

class WordExtractor:

    @staticmethod
    def damerau_levenshtein_distance(string_a, string_b):
        string_a = WordExtractor.remove_unicode_chars(string_a)
        string_b = WordExtractor.remove_unicode_chars(string_b)
        return jellyfish.damerau_levenshtein_distance(string_a, string_b)

    @staticmethod
    def jaro_winkler_distance(string_a, string_b):
        return jellyfish.jaro_winkler(string_a, string_b)

    @staticmethod
    def metaphone(string):
        return jellyfish.metaphone(string)

    @staticmethod
    def soundex(string):
        return jellyfish.soundex(string)

    @staticmethod
    def nysiis(string):
        return jellyfish.nysiis(string)

    @staticmethod
    def match_rating_codex(string):
        return jellyfish.match_rating_codex(string)

    @staticmethod
    def first_four_letters_match(string_a, string_b):
        return string_a[:4] == string_b[:4]

    @staticmethod
    def has_number_in_first_four_letter(string):
        for char in string[:4]:
            n = ord(char)
            if 48 <= n <= 57:
                return True
        return False

    @staticmethod
    def has_special_characters(string):
        for char in string:
            n = ord(char)
            if not ((48 <= n <= 57) or (65 <= n <= 90) or (97 <= n <= 122)):
                return True
        return False

    @staticmethod
    def remove_unicode_chars(string):
        i = 0
        while True:
            if i >= len(string):
                break
            if ord(string[i]) > 255:
                string = string[:i] + string[i+1:]
                i -= 1
            i += 1
        return string
