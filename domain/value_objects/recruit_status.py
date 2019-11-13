
class RecruitStatus:

    def __init__(self, status):
        self.__status = status

    @property
    def joined(self):
        return self.__status != 0
