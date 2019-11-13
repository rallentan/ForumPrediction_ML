
class FluencyContainer:

    @staticmethod
    def from_comma_separated_string(comma_separated_string):
        if comma_separated_string is None:
            return FluencyContainer(None)
        fluencies = comma_separated_string.split(',')
        return FluencyContainer(*fluencies)

    def __init__(self, *fluencies):
        if fluencies is None:
            self.__fluency_count = None
            self.__fluency_1 = None
            self.__fluency_2 = None
            self.__fluency_3 = None
            self.__fluency_4 = None
        else:
            self.__fluency_count = len(fluencies)
            self.__fluency_1 = ''
            self.__fluency_2 = ''
            self.__fluency_3 = ''
            self.__fluency_4 = ''
            if self.__fluency_count > 0:
                self.__fluency_1 = fluencies[0].strip()
            if self.__fluency_count > 1:
                self.__fluency_2 = fluencies[1].strip()
            if self.__fluency_count > 2:
                self.__fluency_3 = fluencies[2].strip()
            if self.__fluency_count > 3:
                self.__fluency_4 = fluencies[3].strip()

    @property
    def count(self):
        return self.__fluency_count

    @property
    def one(self):
        return self.__fluency_1

    @property
    def two(self):
        return self.__fluency_2

    @property
    def three(self):
        return self.__fluency_3

    @property
    def four(self):
        return self.__fluency_4
