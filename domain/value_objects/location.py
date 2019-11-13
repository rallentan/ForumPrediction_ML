
class Location:

    @staticmethod
    def from_string(location_string):
        parts = location_string.split(',')
        country = parts[0].strip()
        province = None
        if len(parts) > 1:
            province = parts[1].strip()
        return Location(country, province)

    def __init__(self, country, province=None):
        if type(country) is not str:
            raise Exception
        if type(province) is not str and province is not None:
            raise Exception
        self.__country = country
        self.__province = province
        self.__gave_province = (province is not None)

    @property
    def country(self):
        return self.__country

    @property
    def province(self):
        return self.__province

    @property
    def gave_province(self):
        return self.__gave_province
